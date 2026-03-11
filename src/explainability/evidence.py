import torch
from src.utils.labels import CHEXPERT_LABELS


def concept_evidence(concept_logits, topk=6):
    probs = torch.sigmoid(concept_logits).detach().cpu().tolist()
    items = []
    for i, p in enumerate(probs):
        items.append((CHEXPERT_LABELS[i], float(p)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:topk], items

