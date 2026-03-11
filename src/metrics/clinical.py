from src.utils.labels import extract_concepts


def precision_recall_f1(refs, hyps):
    ps, rs, fs = [], [], []
    for r, h in zip(refs, hyps):
        r_set = set(extract_concepts(r))
        h_set = set(extract_concepts(h))
        if not h_set:
            p = 0.0
        else:
            p = len(r_set & h_set) / len(h_set)
        if not r_set:
            r_ = 0.0
        else:
            r_ = len(r_set & h_set) / len(r_set)
        if p + r_ == 0:
            f1 = 0.0
        else:
            f1 = 2 * p * r_ / (p + r_)
        ps.append(p)
        rs.append(r_)
        fs.append(f1)
    return {
        "precision": sum(ps) / max(1, len(ps)),
        "recall": sum(rs) / max(1, len(rs)),
        "f1": sum(fs) / max(1, len(fs)),
    }

