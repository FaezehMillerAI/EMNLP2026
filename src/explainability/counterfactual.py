import torch
import torch.nn.functional as F

from src.utils.labels import CHEXPERT_LABELS


def log_likelihood(logits, target_ids, pad_id):
    # logits: BxTxV, target_ids: BxT
    logp = F.log_softmax(logits, dim=-1)
    tgt = target_ids[:, 1:].contiguous()
    pred = logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != pad_id).float()
    return (pred * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)


@torch.no_grad()
def counterfactual_concept_influence(model, images, input_ids, vocab, topk=4):
    # Compute baseline log-likelihood
    device = images.device
    pad_id = vocab.stoi["<pad>"]

    memory = model.encoder(images)
    pooled = memory.mean(dim=1)
    concept_logits = model.concept_head(pooled)
    concept_ids = torch.arange(concept_logits.shape[-1], device=device)
    concept_tokens = model.concept_embed(concept_ids).unsqueeze(0).expand(images.size(0), -1, -1)
    memory_full = torch.cat([memory, concept_tokens], dim=1)
    logits_full = model.decoder(input_ids, memory_full)
    base_ll = log_likelihood(logits_full, input_ids, pad_id)

    # Remove each concept token in memory and compute delta
    deltas = []
    for i in range(concept_tokens.shape[1]):
        mask = torch.ones(concept_tokens.shape[1], dtype=torch.bool, device=device)
        mask[i] = False
        memory_cf = torch.cat([memory, concept_tokens[:, mask, :]], dim=1)
        logits_cf = model.decoder(input_ids, memory_cf)
        ll_cf = log_likelihood(logits_cf, input_ids, pad_id)
        deltas.append((CHEXPERT_LABELS[i], float((base_ll - ll_cf).mean().item())))

    deltas.sort(key=lambda x: x[1], reverse=True)
    return deltas[:topk], deltas

