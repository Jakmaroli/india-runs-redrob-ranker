"""
textmatch.py
============
The semantic half -- "understand the role, not the keywords".

DEFAULT: TF-IDF over candidate career narratives, cosine vs a JD-INTENT query.
Satisfies the compute contract out of the box (CPU, no network, no downloads).

OPTIONAL (no hosted LLM, no network at rank time): if a precomputed offline
sentence-transformer index is present, load_embedding_scores loads it from disk and
scores cosine vs the precomputed JD vector. rank.py prefers it when --cache is given,
else falls back to TF-IDF. Nothing here calls any API.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

JD_QUERY = (
    "senior ai engineer building production ranking retrieval and recommendation "
    "systems. embeddings based retrieval, vector search, hybrid search, semantic "
    "search, learning to rank, re-ranking. shipped end to end search recommendation "
    "system to real users at scale in production at a product company. information "
    "retrieval, nlp, language models, relevance. evaluation of ranking with ndcg mrr "
    "map offline online a/b testing. python, scalable inference, feature engineering, "
    "machine learning systems. matching candidates to jobs, talent intelligence."
)


def candidate_doc(c: Dict) -> str:
    p = (c.get("profile") or {})
    # FIX: .get(key, "") only substitutes when the KEY is missing, not when its
    # value is explicitly None -- a common pattern in real scraped/aggregated
    # data. Without the `or ""` guard, a single None field crashes " ".join()
    # with TypeError and aborts the entire run. Reproduced directly: a
    # candidate with current_title=None took down the whole 100k-row pipeline.
    parts = [p.get("headline") or "", p.get("summary") or "",
             p.get("current_title") or "", p.get("current_industry") or ""]
    for h in c.get("career_history", []) or []:
        seg = f'{h.get("title") or ""} {h.get("description") or ""} {h.get("industry") or ""}'
        parts += [seg, seg]
    parts.append(" ".join((s.get("name") or "") for s in (c.get("skills", []) or [])))
    return " ".join(parts)


def _minmax(sims: np.ndarray) -> np.ndarray:
    lo, hi = float(sims.min()), float(sims.max())
    if hi - lo < 1e-9:
        return np.zeros_like(sims)
    return (sims - lo) / (hi - lo)


def semantic_scores(docs: List[str]) -> np.ndarray:
    vec = TfidfVectorizer(
        sublinear_tf=True, max_features=30000, ngram_range=(1, 2),
        min_df=5, stop_words="english", dtype=np.float32,
    )
    cand_mat = vec.fit_transform(docs)
    query_vec = vec.transform([JD_QUERY])
    sims = linear_kernel(query_vec, cand_mat).ravel()
    return _minmax(sims)


def load_embedding_scores(cache_dir: str, candidate_ids: List[str]) -> Optional[np.ndarray]:
    ids_p = os.path.join(cache_dir, "candidate_ids.npy")
    emb_p = os.path.join(cache_dir, "embeddings.npy")
    jd_p = os.path.join(cache_dir, "jd_embedding.npy")
    if not all(os.path.exists(p) for p in (ids_p, emb_p, jd_p)):
        return None
    ids = np.load(ids_p, allow_pickle=True).tolist()
    embs = np.load(emb_p)
    jd = np.load(jd_p)
    sims = embs @ jd
    sim_map = {cid: float(s) for cid, s in zip(ids, sims)}
    aligned = np.array([sim_map.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
    return _minmax(aligned)
