import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.iu_xray import IUXrayDataset
from src.explainability.evidence import concept_evidence
from src.explainability.counterfactual import counterfactual_concept_influence
from src.explainability.graph_trace import top_concept_edges
from src.metrics.nlp import compute_bleu, compute_meteor, compute_rouge, compute_distinct
from src.metrics.clinical import precision_recall_f1
from src.models.model import XAIReportModel
from src.utils.config import load_config
from src.utils.collate import collate_fn
from src.utils.vocab_io import load_vocab
from src.visuals.report_card import save_report_card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qual_n", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(out_dir / "vocab.json")
    ds_path = Path(cfg["paths"]["processed_dir"]) / f"iu_xray_{args.split}.jsonl"
    ds = IUXrayDataset(ds_path, vocab, max_len=cfg["model"]["max_len"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XAIReportModel(
        vocab_size=getattr(vocab, "tok", None).vocab_size if hasattr(vocab, "tok") else len(vocab.itos),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_len=cfg["model"]["max_len"],
        encoder_name=cfg["model"]["encoder_name"],
        decoder_type=cfg["model"].get("decoder_type", "basic"),
        pretrained_lm=cfg["model"].get("pretrained_lm", "gpt2"),
        use_concept_graph=cfg["model"].get("use_concept_graph", True),
        concept_graph_path=cfg["model"].get("concept_graph_path"),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()

    refs, hyps = [], []
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(exist_ok=True)
    adj = None
    graph_path = cfg["model"].get("concept_graph_path")
    if cfg["model"].get("use_concept_graph", True) and graph_path and Path(graph_path).exists():
        adj = np.load(graph_path)

    sample_count = 0
    for batch in tqdm(loader, desc="eval"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        gen_ids, concept_logits = model.generate(images, vocab, max_len=input_ids.size(1), device=device)

        for i in range(gen_ids.size(0)):
            hyp = vocab.decode(gen_ids[i].tolist())
            ref = batch["texts"][i]
            refs.append(ref)
            hyps.append(hyp)

            if sample_count < args.qual_n:
                top_concepts, all_concepts = concept_evidence(concept_logits[i].cpu(), topk=cfg["explainability"]["topk_concepts"])
                deltas_top, _ = counterfactual_concept_influence(
                    model,
                    images[i : i + 1],
                    input_ids[i : i + 1],
                    vocab,
                    topk=cfg["explainability"]["counterfactual_k"],
                )
                edges_top = None
                if adj is not None:
                    probs = torch.sigmoid(concept_logits[i]).detach().cpu().numpy()
                    edges_top = top_concept_edges(adj, probs, topk=5, prob_thresh=0.3)
                image_path = ds.items[sample_count]["image"]
                out_path = qual_dir / f"sample_{sample_count}.png"
                save_report_card(image_path, ref, hyp, top_concepts, deltas_top, out_path, edges_top=edges_top)
                sample_count += 1

    metrics = {}
    metrics.update(compute_bleu(refs, hyps))
    metrics.update(compute_rouge(refs, hyps))
    metrics.update(compute_meteor(refs, hyps))
    metrics.update(compute_distinct(hyps))
    metrics.update(precision_recall_f1(refs, hyps))

    with open(out_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
