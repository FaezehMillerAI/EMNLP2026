import argparse
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "raddar/iu-xray",
        "-p",
        args.out,
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

