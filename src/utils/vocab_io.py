import json
from transformers import GPT2TokenizerFast
from src.utils.tokenizer import Vocab, HFTokenizerWrapper


def load_vocab(path):
    data = json.load(open(path))
    if data.get("type") == "gpt2":
        tok = GPT2TokenizerFast.from_pretrained(data.get("name", "gpt2"))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return HFTokenizerWrapper(tok, max_len=data.get("max_len", 300))

    v = Vocab(min_freq=1)
    v.itos = data["itos"]
    v.stoi = {t: i for i, t in enumerate(v.itos)}
    return v
