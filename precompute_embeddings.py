#!/usr/bin/env python3
"""
precompute_embeddings.py -- OPTIONAL Phase-1 offline embedding index builder.
The ONLY step allowed to use the network, and ONLY to download a local open-source
sentence-transformer (NO hosted LLM, NO GPT/Claude/Groq API, NO per-candidate calls).
Spec §10.3 permits pre-computation to exceed the 5-minute window.

Writes an index rank.py loads at Stage 3 with NO network:
    cache/candidate_ids.npy   (N,)   object
    cache/embeddings.npy      (N, D) float32, L2-normalised
    cache/jd_embedding.npy    (D,)   float32, L2-normalised

    pip install sentence-transformers
    python precompute_embeddings.py --candidates ./candidates.jsonl --out ./cache/
    python rank.py --candidates ./candidates.jsonl --cache ./cache/ --out ./submission.csv

Skip it entirely and rank.py uses TF-IDF, still satisfying every constraint.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import time

import numpy as np

from ranker.textmatch import candidate_doc, JD_QUERY


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="./cache/")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="Local open-source sentence-transformer. NOT a hosted LLM.")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # local, no API

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    ids, docs = [], []
    with _open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                ids.append(c["candidate_id"])
                docs.append(candidate_doc(c))
    print(f"Loaded {len(ids):,} candidates ({time.time()-t0:.1f}s)")

    model = SentenceTransformer(args.model, device="cpu")
    embs = model.encode(docs, batch_size=args.batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    jd = model.encode([JD_QUERY], normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)[0]

    np.save(os.path.join(args.out, "candidate_ids.npy"), np.array(ids, dtype=object))
    np.save(os.path.join(args.out, "embeddings.npy"), embs)
    np.save(os.path.join(args.out, "jd_embedding.npy"), jd)
    print(f"Wrote index to {args.out} ({time.time()-t0:.1f}s). shape={embs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
