import argparse
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    output_dir = args.output_dir or "outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_cmd = ["python", "train.py", "--config", args.config, "--smoke", "--output_dir", output_dir]
    print("Running:", " ".join(train_cmd))
    subprocess.check_call(train_cmd)

    ckpt = Path(output_dir) / "checkpoint_epoch1.pt"
    eval_cmd = [
        "python",
        "eval.py",
        "--config",
        args.config,
        "--output_dir",
        output_dir,
        "--checkpoint",
        str(ckpt),
        "--split",
        "val",
        "--qual_n",
        "2",
    ]
    print("Running:", " ".join(eval_cmd))
    subprocess.check_call(eval_cmd)


if __name__ == "__main__":
    main()

