import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.labels import extract_concepts


class IUXrayDataset(Dataset):
    def __init__(self, jsonl_path, vocab, max_len=300, image_size=224):
        self.items = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["image"]).convert("RGB")
        img = self.transform(img)
        text = item["report"]
        ids = torch.tensor(self.vocab.encode(text, self.max_len), dtype=torch.long)
        concepts = extract_concepts(text)
        return {
            "image": img,
            "text": text,
            "input_ids": ids,
            "concepts": concepts,
        }

