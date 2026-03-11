import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd


def pick_frontal(proj_df, uid):
    rows = proj_df[proj_df["uid"] == uid]
    # prefer PA/AP views
    frontal = rows[rows["projection"].isin(["PA", "AP", "AP Supine", "AP Semi erect"])]
    if len(frontal) == 0:
        frontal = rows
    if len(frontal) == 0:
        return None
    return frontal.iloc[0]["filename"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="IU-Xray raw folder (unzipped)")
    ap.add_argument("--out", default="data/processed", help="Output folder")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reports = pd.read_csv(raw / "indiana_reports.csv")
    projections = pd.read_csv(raw / "indiana_projections.csv")

    items = []
    for _, row in reports.iterrows():
        uid = row["uid"]
        findings = str(row.get("findings", ""))
        impression = str(row.get("impression", ""))
        report = (findings + " " + impression).strip()
        if not report:
            continue
        img = pick_frontal(projections, uid)
        if img is None:
            continue
        img_path = raw / "images" / img
        if not img_path.exists():
            continue
        items.append({"uid": uid, "image": str(img_path), "report": report})

    random.Random(args.seed).shuffle(items)
    n = len(items)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)

    splits = {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }

    for split, rows in splits.items():
        with open(out / f"iu_xray_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print("Wrote", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()

