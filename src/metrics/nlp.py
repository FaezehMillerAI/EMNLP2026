from collections import Counter

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


SMOOTH = SmoothingFunction().method3


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)


def compute_bleu(refs, hyps):
    # refs, hyps: list of strings
    bleu_scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}
    for r, h in zip(refs, hyps):
        r_tok = [r.split()]
        h_tok = h.split()
        bleu_scores["bleu1"].append(nltk.translate.bleu_score.sentence_bleu(r_tok, h_tok, weights=(1, 0, 0, 0), smoothing_function=SMOOTH))
        bleu_scores["bleu2"].append(nltk.translate.bleu_score.sentence_bleu(r_tok, h_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=SMOOTH))
        bleu_scores["bleu3"].append(nltk.translate.bleu_score.sentence_bleu(r_tok, h_tok, weights=(1/3, 1/3, 1/3, 0), smoothing_function=SMOOTH))
        bleu_scores["bleu4"].append(nltk.translate.bleu_score.sentence_bleu(r_tok, h_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SMOOTH))
    return {k: sum(v) / max(1, len(v)) for k, v in bleu_scores.items()}


def compute_meteor(refs, hyps):
    _ensure_nltk()
    scores = []
    for r, h in zip(refs, hyps):
        r_tok = nltk.word_tokenize(r)
        h_tok = nltk.word_tokenize(h)
        scores.append(meteor_score([r_tok], h_tok))
    return {"meteor": sum(scores) / max(1, len(scores))}


def compute_rouge(refs, hyps):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r, h in zip(refs, hyps):
        scores.append(scorer.score(r, h)["rougeL"].fmeasure)
    return {"rougeL": sum(scores) / max(1, len(scores))}


def compute_distinct(hyps):
    # Distinct-1/2 diversity
    unigrams = Counter()
    bigrams = Counter()
    total_unigrams = 0
    total_bigrams = 0
    for h in hyps:
        toks = h.split()
        total_unigrams += len(toks)
        total_bigrams += max(0, len(toks) - 1)
        unigrams.update(toks)
        bigrams.update(zip(toks, toks[1:]))
    d1 = len(unigrams) / max(1, total_unigrams)
    d2 = len(bigrams) / max(1, total_bigrams)
    return {"distinct1": d1, "distinct2": d2}
