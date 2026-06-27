#!/usr/bin/env python3
"""
rank.py -- India Runs / Redrob Intelligent Candidate Discovery & Ranking
========================================================================
Single-command entrypoint -> top-100 CSV.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Optional offline-embedding upgrade (still CPU-only, NO network at rank time):

    python rank.py --candidates ./candidates.jsonl --cache ./cache/ --out ./submission.csv

Compute contract (Stage 3): CPU-only, no GPU, NO network, <=16 GB, <=5 min for 100K.
TF-IDF (or a precomputed local embedding index) + numpy + pure-python rules; no API calls.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import sys
import time
from typing import Dict, List

from ranker.textmatch import candidate_doc, semantic_scores, load_embedding_scores
from ranker.features import build_features
from ranker.traps import honeypot_report, keyword_stuffer_score
from ranker.scoring import score_candidate
from ranker.reasoning import build_reasoning


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_candidates(path: str) -> List[Dict]:
    out = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cache", default=None,
                    help="Optional dir with precomputed OFFLINE embedding index. "
                         "No network used to read it. Falls back to TF-IDF if absent.")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/6] Loading candidates from {args.candidates} ...", file=sys.stderr)
    cands = load_candidates(args.candidates)
    ids = [c["candidate_id"] for c in cands]
    print(f"      loaded {len(cands):,} ({time.time()-t0:.1f}s)", file=sys.stderr)

    print("[2/6] Semantic relevance to the JD intent ...", file=sys.stderr)
    sem = None
    if args.cache:
        sem = load_embedding_scores(args.cache, ids)
        if sem is not None:
            print(f"      using precomputed offline embedding index ({args.cache})",
                  file=sys.stderr)
    if sem is None:
        docs = [candidate_doc(c) for c in cands]
        sem = semantic_scores(docs)
        del docs
        import gc; gc.collect()
        print("      using TF-IDF semantic scores", file=sys.stderr)
    print(f"      semantic ready ({time.time()-t0:.1f}s)", file=sys.stderr)

    print("[3/6] features + [4/6] traps + [5/6] scoring ...", file=sys.stderr)
    rows = []
    n_honeypot = 0
    for i, c in enumerate(cands):
        feats = build_features(c)
        hp = honeypot_report(c)
        if hp["is_honeypot"]:
            n_honeypot += 1
        kw = keyword_stuffer_score(c, feats["has_engineering_career"])
        score = score_candidate(feats, float(sem[i]), kw, hp["is_honeypot"])
        rows.append({"candidate_id": c["candidate_id"], "score": score,
                     "_c": c, "_feats": feats, "_kw": kw, "_hp": hp["is_honeypot"]})
    print(f"      scored all; flagged {n_honeypot} honeypots ({time.time()-t0:.1f}s)",
          file=sys.stderr)

    print(f"[6/6] Top-{args.topk} -> {args.out} ...", file=sys.stderr)
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = rows[: args.topk]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, start=1):
            reasoning = build_reasoning(r["_c"], r["_feats"], r["score"], r["_kw"], r["_hp"])
            w.writerow([r["candidate_id"], rank, f"{r['score']:.4f}", reasoning])

    hp_top = sum(1 for r in top if r["_hp"])
    print(f"Done. Wrote {len(top)} rows in {time.time()-t0:.1f}s. "
          f"Honeypots in top-{args.topk}: {hp_top}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
