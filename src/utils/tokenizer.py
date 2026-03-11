import re
from collections import Counter

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


def basic_tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[.,;:!?()]", text)
    return tokens


class Vocab:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = []

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(basic_tokenize(t))
        self.itos = [
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["bos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["unk"],
        ]
        for tok, freq in counter.items():
            if freq >= self.min_freq:
                self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, text, max_len):
        tokens = [SPECIAL_TOKENS["bos"]] + basic_tokenize(text) + [SPECIAL_TOKENS["eos"]]
        ids = [self.stoi.get(t, self.stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
        if len(ids) < max_len:
            ids += [self.stoi[SPECIAL_TOKENS["pad"]]] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        tokens = []
        for i in ids:
            t = self.itos[i]
            if t in (SPECIAL_TOKENS["bos"], SPECIAL_TOKENS["eos"], SPECIAL_TOKENS["pad"]):
                continue
            tokens.append(t)
        return " ".join(tokens).replace(" ,", ",").replace(" .", ".")

