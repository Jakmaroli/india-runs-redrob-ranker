#!/usr/bin/env python3
"""make_sample.py -- slice <=100 candidates for the §10.5 sandbox / Colab demo.

    python make_sample.py --candidates ./candidates.jsonl --n 100 --out ./sample_candidates.jsonl
"""
from __future__ import annotations
import argparse
import gzip


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="sample_candidates.jsonl")
    args = ap.parse_args()
    n = 0
    with _open(args.candidates) as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(line + "\n")
            n += 1
            if n >= args.n:
                break
    print(f"Wrote {n} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
