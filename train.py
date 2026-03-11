import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast
import numpy as np

from src.data.iu_xray import IUXrayDataset
from src.metrics.nlp import compute_bleu, compute_meteor, compute_rouge, compute_distinct
from src.metrics.clinical import precision_recall_f1
from src.models.model import XAIReportModel
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.tokenizer import Vocab, HFTokenizerWrapper
from src.utils.collate import collate_fn
from src.utils.labels import CHEXPERT_LABELS, extract_concepts


def label_smoothed_nll_loss(logits, target, pad_id, smoothing=0.1):
    # logits: BxTxV, target: BxT
    vocab_size = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)

    target = target[:, 1:].contiguous()
    logp = logp[:, :-1, :]

    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)

    mask = (target != pad_id).float()
    nll = (nll * mask).sum() / (mask.sum() + 1e-8)
    smooth = (smooth * mask).sum() / (mask.sum() + 1e-8)

    loss = (1 - smoothing) * nll + smoothing * smooth
    return loss


def evaluate(model, loader, vocab, device):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            gen_ids, _ = model.generate(images, vocab, max_len=input_ids.size(1), device=device)
            for i in range(gen_ids.size(0)):
                hyp = vocab.decode(gen_ids[i].tolist())
                ref = batch["texts"][i]
                hyps.append(hyp)
                refs.append(ref)

    metrics = {}
    metrics.update(compute_bleu(refs, hyps))
    metrics.update(compute_rouge(refs, hyps))
    metrics.update(compute_meteor(refs, hyps))
    metrics.update(compute_distinct(hyps))
    metrics.update(precision_recall_f1(refs, hyps))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    set_seed(cfg["seed"])

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json = Path(cfg["paths"]["processed_dir"]) / "iu_xray_train.jsonl"
    val_json = Path(cfg["paths"]["processed_dir"]) / "iu_xray_val.jsonl"

    decoder_type = cfg["model"].get("decoder_type", "basic")
    vocab_path = out_dir / "vocab.json"

    if decoder_type == "gpt2":
        tok = GPT2TokenizerFast.from_pretrained(cfg["model"].get("pretrained_lm", "gpt2"))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        vocab = HFTokenizerWrapper(tok, max_len=cfg["model"]["max_len"])
        with open(vocab_path, "w") as f:
            json.dump(
                {"type": "gpt2", "name": cfg["model"].get("pretrained_lm", "gpt2"), "max_len": cfg["model"]["max_len"]},
                f,
            )
        vocab_size = tok.vocab_size
    else:
        train_texts = [json.loads(line)["report"] for line in open(train_json)]
        vocab = Vocab(min_freq=cfg["model"]["vocab_min_freq"])
        vocab.build(train_texts)
        with open(vocab_path, "w") as f:
            json.dump({"itos": vocab.itos}, f)
        vocab_size = len(vocab.itos)

    # Build concept graph if requested and missing
    if cfg["model"].get("use_concept_graph", True):
        graph_path = cfg["model"].get("concept_graph_path")
        if graph_path:
            gp = Path(graph_path)
            if not gp.exists():
                n = len(CHEXPERT_LABELS)
                counts = np.zeros((n, n), dtype=np.float32)
                with open(train_json, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        concepts = extract_concepts(item["report"])
                        idxs = [CHEXPERT_LABELS.index(c) for c in concepts]
                        for i in idxs:
                            for j in idxs:
                                counts[i, j] += 1
                row_sums = counts.sum(axis=1, keepdims=True) + 1e-6
                adj = counts / row_sums
                gp.parent.mkdir(parents=True, exist_ok=True)
                np.save(gp, adj)

    train_ds = IUXrayDataset(train_json, vocab, max_len=cfg["model"]["max_len"])
    val_ds = IUXrayDataset(val_json, vocab, max_len=cfg["model"]["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = XAIReportModel(
        vocab_size=vocab_size,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_len=cfg["model"]["max_len"],
        encoder_name=cfg["model"]["encoder_name"],
        decoder_type=decoder_type,
        pretrained_lm=cfg["model"].get("pretrained_lm", "gpt2"),
        use_concept_graph=cfg["model"].get("use_concept_graph", True),
        concept_graph_path=cfg["model"].get("concept_graph_path"),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    bce = torch.nn.BCEWithLogitsLoss()

    epochs = 1 if args.smoke else cfg["train"]["epochs"]
    max_steps = 1 if args.smoke else cfg["train"]["max_steps"]

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            concept_labels = batch["concept_labels"].to(device)

            logits, concept_logits = model(images, input_ids)
            pad_id = vocab.stoi["<pad>"] if hasattr(vocab, "stoi") else vocab.pad_id
            ce_loss = label_smoothed_nll_loss(logits, input_ids, pad_id=pad_id, smoothing=cfg["train"]["label_smoothing"])
            c_loss = bce(concept_logits, concept_labels)
            loss = ce_loss + 0.2 * c_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if max_steps is not None and global_step >= max_steps:
                break

        ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        metrics = evaluate(model, val_loader, vocab, device)
        with open(out_dir / f"metrics_epoch{epoch}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if max_steps is not None and global_step >= max_steps:
            break


if __name__ == "__main__":
    main()
