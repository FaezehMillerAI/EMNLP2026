import json
from src.utils.tokenizer import Vocab


def load_vocab(path):
    data = json.load(open(path))
    v = Vocab(min_freq=1)
    v.itos = data["itos"]
    v.stoi = {t: i for i, t in enumerate(v.itos)}
    return v

