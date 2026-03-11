import torch

from src.utils.labels import CHEXPERT_LABELS

LABEL_TO_IDX = {l: i for i, l in enumerate(CHEXPERT_LABELS)}


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    texts = [b["text"] for b in batch]

    concept_labels = torch.zeros(len(batch), len(CHEXPERT_LABELS))
    for i, b in enumerate(batch):
        for c in b["concepts"]:
            if c in LABEL_TO_IDX:
                concept_labels[i, LABEL_TO_IDX[c]] = 1.0

    return {
        "images": images,
        "input_ids": input_ids,
        "texts": texts,
        "concept_labels": concept_labels,
    }

