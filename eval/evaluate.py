#!/usr/bin/env python3
"""
evaluate.py -- offline evaluation harness for the Redrob ranker.
================================================================
Why this exists: the JD lists "Hands-on experience designing evaluation frameworks for
ranking systems -- NDCG, MRR, MAP, offline-to-online correlation" as a MUST-HAVE skill,
and the spec hides the leaderboard. So we validate the ranker offline against a
JD-derived relevance oracle (eval/relevance_oracle.py) -- NOT an independent one, since
it reuses the ranker's own build_features()/traps logic restated as discrete tiers. This
is a self-consistency check, not blind validation against ground truth. We still report
the exact competition composite, run a feature-ABLATION study, and check trap robustness,
since those are informative regardless of the oracle's circularity.

    python -m eval.evaluate --candidates ./candidates.jsonl --out-dir ./eval_out/

It computes the expensive per-candidate features ONCE, caches them, then scores the
baseline and every ablation cheaply by reusing ranker.scoring.score_candidate with
toggled inputs -- so there is zero duplicated scoring logic and no chance of drift.

Outputs:
    eval_out/eval_metrics.json   machine-readable metrics + ablation deltas
    eval_out/eval_report.md      human-readable report (drop into the repo / deck)
"""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import math
import os
import time
from typing import Dict, List

import numpy as np

from ranker.textmatch import candidate_doc, semantic_scores, load_embedding_scores
from ranker.features import build_features
from ranker.traps import honeypot_report, keyword_stuffer_score
from ranker.scoring import score_candidate
from eval.relevance_oracle import tier_from

COMPOSITE_W = {"ndcg10": 0.50, "ndcg50": 0.30, "map": 0.15, "p10": 0.05}
RELEVANT_TIER = 3  # spec: "relevant" == tier 3+


# ----------------------------- metrics ------------------------------------
def _dcg(rels: List[float]) -> float:
    return sum((2.0 ** r - 1.0) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: List[float], all_rels: List[float], k: int) -> float:
    ideal = sorted(all_rels, reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(ranked_rels[:k]) / idcg


def average_precision(ranked_rels: List[float], total_relevant: int) -> float:
    """AP over the ranked list (binary relevance tier>=RELEVANT_TIER),
    normalised by total relevant in the pool (capped at len(list))."""
    if total_relevant == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, r in enumerate(ranked_rels):
        if r >= RELEVANT_TIER:
            hits += 1
            ap += hits / (i + 1)
    denom = min(total_relevant, len(ranked_rels))
    return ap / denom if denom else 0.0


def precision_at_k(ranked_rels: List[float], k: int) -> float:
    top = ranked_rels[:k]
    return sum(1 for r in top if r >= RELEVANT_TIER) / k if top else 0.0


def composite(metrics: Dict[str, float]) -> float:
    return (COMPOSITE_W["ndcg10"] * metrics["ndcg@10"] +
            COMPOSITE_W["ndcg50"] * metrics["ndcg@50"] +
            COMPOSITE_W["map"] * metrics["map"] +
            COMPOSITE_W["p10"] * metrics["p@10"])


def evaluate_ranking(ranked_ids: List[str], tier_map: Dict[str, int],
                     all_tiers: List[int]) -> Dict[str, float]:
    ranked_rels = [tier_map[i] for i in ranked_ids]
    total_relevant = sum(1 for t in all_tiers if t >= RELEVANT_TIER)
    m = {
        "ndcg@10": ndcg_at_k(ranked_rels, all_tiers, 10),
        "ndcg@50": ndcg_at_k(ranked_rels, all_tiers, 50),
        "map": average_precision(ranked_rels, total_relevant),
        "p@10": precision_at_k(ranked_rels, 10),
    }
    m["composite"] = composite(m)
    return m


# ----------------------------- data ---------------------------------------
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def load_candidates(path: str) -> List[Dict]:
    out = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ----------------------------- scoring variants ---------------------------
def score_variant(records: List[Dict], variant: str) -> List[str]:
    """Return the top-100 candidate_ids under a scoring variant.

    All variants reuse ranker.scoring.score_candidate with toggled inputs, so the
    scoring formula is never re-implemented (no logic drift).
    """
    scored = []
    for r in records:
        feats = r["feats"]
        sem = r["sem"]
        kw = r["kw"]
        hp = r["hp"]
        if variant == "baseline":
            sc = score_candidate(feats, sem, kw, hp)
        elif variant == "no_semantic":
            sc = score_candidate(feats, 0.0, kw, hp)
        elif variant == "no_must_skills":
            f = dict(feats); f["must_skills"] = 0.0
            sc = score_candidate(f, sem, kw, hp)
        elif variant == "no_behavioral":
            f = dict(feats); f["behavioral"] = 1.0
            sc = score_candidate(f, sem, kw, hp)
        elif variant == "no_location":
            f = dict(feats); f["location_fit"] = 1.0
            sc = score_candidate(f, sem, kw, hp)
        elif variant == "no_penalties":
            # Zero every penalty input via existing formula (no duplication).
            f = dict(feats)
            f["job_hop"] = f["consulting"] = f["research_only"] = f["langchain_only"] = 0.0
            f["has_engineering_career"] = True
            sc = score_candidate(f, sem, 0.0, hp)
        elif variant == "no_traps":
            # Do not zero honeypots -> show them leak into the top.
            sc = score_candidate(feats, sem, kw, False)
        else:
            raise ValueError(variant)
        scored.append((sc, r["candidate_id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cid for _, cid in scored[:100]]


def overlap_at_k(a: List[str], b: List[str], k: int) -> float:
    return len(set(a[:k]) & set(b[:k])) / k


# ----------------------------- main ---------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--cache", default=None, help="Optional offline embedding index dir")
    ap.add_argument("--out-dir", default="./eval_out/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    print("Loading candidates ...")
    cands = load_candidates(args.candidates)
    ids = [c["candidate_id"] for c in cands]
    print(f"  {len(cands):,} loaded ({time.time()-t0:.1f}s)")

    # Semantic (embedding index if provided, else TF-IDF) -- computed once.
    sem = None
    if args.cache:
        sem = load_embedding_scores(args.cache, ids)
    if sem is None:
        sem = semantic_scores([candidate_doc(c) for c in cands])
    print(f"  semantic ready ({time.time()-t0:.1f}s)")

    # Features + traps + oracle tier -- computed ONCE, cached per candidate.
    records = []
    tier_map = {}
    for i, c in enumerate(cands):
        feats = build_features(c)
        hp = honeypot_report(c)["is_honeypot"]
        kw = keyword_stuffer_score(c, feats["has_engineering_career"])
        tier = tier_from(c, feats, hp, kw)
        tier_map[c["candidate_id"]] = tier
        records.append({"candidate_id": c["candidate_id"], "feats": feats,
                        "sem": float(sem[i]), "kw": kw, "hp": hp, "tier": tier})
    all_tiers = [r["tier"] for r in records]
    print(f"  features + oracle tiers ready ({time.time()-t0:.1f}s)")

    # Tier distribution (sanity: how many real tier-5/4 exist in the pool).
    dist = {t: sum(1 for x in all_tiers if x == t) for t in range(6)}

    # Baseline + ablations.
    variants = ["baseline", "no_semantic", "no_must_skills", "no_behavioral",
                "no_location", "no_penalties", "no_traps"]
    rankings = {v: score_variant(records, v) for v in variants}
    print(f"  scored {len(variants)} variants ({time.time()-t0:.1f}s)")

    base = rankings["baseline"]
    base_metrics = evaluate_ranking(base, tier_map, all_tiers)

    hp_ids = {r["candidate_id"] for r in records if r["hp"]}
    kw_ids = {r["candidate_id"] for r in records if r["kw"] > 0.3}
    consult_ids = {r["candidate_id"] for r in records if r["feats"]["consulting"] > 0.5}

    def rate(ranked, idset, k):
        return sum(1 for i in ranked[:k] if i in idset) / k

    robustness = {
        "honeypot_rate@10": rate(base, hp_ids, 10),
        "honeypot_rate@100": rate(base, hp_ids, 100),
        "keyword_stuffer_rate@100": rate(base, kw_ids, 100),
        "consulting_heavy_rate@100": rate(base, consult_ids, 100),
    }

    ablation = {}
    for v in variants:
        if v == "baseline":
            continue
        m = evaluate_ranking(rankings[v], tier_map, all_tiers)
        ablation[v] = {
            "composite": round(m["composite"], 4),
            "delta_composite": round(m["composite"] - base_metrics["composite"], 4),
            "overlap@10": round(overlap_at_k(base, rankings[v], 10), 3),
            "overlap@100": round(overlap_at_k(base, rankings[v], 100), 3),
            "honeypots_in_top100": sum(1 for i in rankings[v][:100] if i in hp_ids),
        }

    # Top-10 audit.
    top10_audit = []
    for rank, cid in enumerate(base[:10], 1):
        c = next(x for x in cands if x["candidate_id"] == cid)
        p = c.get("profile") or {}  # FIX: guard against missing/null profile, same as the ranker
        top10_audit.append({
            "rank": rank, "candidate_id": cid, "oracle_tier": tier_map[cid],
            "title": p.get("current_title") or "",
            "company": p.get("current_company") or "",
            "yoe": p.get("years_of_experience") or "",
            "location": p.get("location") or "",
        })

    out = {
        "n_candidates": len(cands),
        "oracle_tier_distribution": dist,
        "baseline_metrics": {k: round(v, 4) for k, v in base_metrics.items()},
        "trap_robustness": {k: round(v, 4) for k, v in robustness.items()},
        "ablation": ablation,
        "top10_audit": top10_audit,
        "runtime_seconds": round(time.time() - t0, 1),
        "note": ("Metrics are computed against a JD-derived proxy oracle that reuses the "
                 "ranker's own feature extraction (build_features/traps) in discrete-tier "
                 "form. This is a self-consistency / sanity check, NOT independent validation "
                 "-- it measures whether the score and the JD rules agree with each other, not "
                 "whether either agrees with the hidden ground truth. Use deltas/ablations for "
                 "relative insight into which components matter, not the absolute numbers."),
    }
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    _write_report(out, os.path.join(args.out_dir, "eval_report.md"))
    print(f"\nDone in {out['runtime_seconds']}s. Wrote eval_metrics.json + eval_report.md")
    print(f"Baseline proxy-composite: {base_metrics['composite']:.4f} | "
          f"honeypots@100: {robustness['honeypot_rate@100']*100:.1f}%")
    return 0


def _write_report(out: Dict, path: str) -> None:
    b = out["baseline_metrics"]
    r = out["trap_robustness"]
    lines = []
    lines.append("# Offline Evaluation Report -- Redrob Ranker\n")
    lines.append("> Metrics use a **JD-derived proxy oracle** (`eval/relevance_oracle.py`) "
                 "that shares feature-extraction code with the ranker itself -- this is a "
                 "**self-consistency check**, not independent validation against ground truth. "
                 "It confirms the score behaves the way the JD's stated rules say it should; "
                 "it does not prove the ranking is correct. Read the **ablation deltas** for "
                 "what each component contributes.\n")
    lines.append(f"- Candidates evaluated: **{out['n_candidates']:,}**")
    lines.append(f"- Runtime: **{out['runtime_seconds']}s**\n")

    lines.append("## Proxy oracle tier distribution")
    lines.append("| Tier | Count |\n|---|---|")
    for t in range(5, -1, -1):
        lines.append(f"| {t} | {out['oracle_tier_distribution'][t]:,} |")
    lines.append("")

    lines.append("## Baseline metrics (competition composite formula)")
    lines.append("| Metric | Value |\n|---|---|")
    lines.append(f"| proxy-NDCG@10 | {b['ndcg@10']:.4f} |")
    lines.append(f"| proxy-NDCG@50 | {b['ndcg@50']:.4f} |")
    lines.append(f"| proxy-MAP | {b['map']:.4f} |")
    lines.append(f"| proxy-P@10 | {b['p@10']:.4f} |")
    lines.append(f"| **proxy-composite** | **{b['composite']:.4f}** |")
    lines.append("\n_composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10_\n")

    lines.append("## Trap robustness")
    lines.append("| Check | Rate |\n|---|---|")
    lines.append(f"| honeypots in top-10 | {r['honeypot_rate@10']*100:.1f}% |")
    lines.append(f"| honeypots in top-100 | {r['honeypot_rate@100']*100:.1f}% (DQ if >10%) |")
    lines.append(f"| keyword-stuffers in top-100 | {r['keyword_stuffer_rate@100']*100:.1f}% |")
    lines.append(f"| consulting-heavy in top-100 | {r['consulting_heavy_rate@100']*100:.1f}% |")
    lines.append("")

    lines.append("## Ablation study (remove one component, re-rank)")
    lines.append("| Variant | proxy-composite | Δ vs baseline | overlap@10 | overlap@100 | honeypots@100 |")
    lines.append("|---|---|---|---|---|---|")
    for v, a in out["ablation"].items():
        lines.append(f"| {v} | {a['composite']:.4f} | {a['delta_composite']:+.4f} | "
                     f"{a['overlap@10']:.2f} | {a['overlap@100']:.2f} | {a['honeypots_in_top100']} |")
    lines.append("\n_A negative Δ means removing that component HURTS quality -> it is "
                 "pulling its weight. `no_traps` should leak honeypots into the top-100._\n")

    lines.append("## Top-10 audit (baseline)")
    lines.append("| Rank | ID | Tier | Title | Company | YoE | Location |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in out["top10_audit"]:
        lines.append(f"| {a['rank']} | {a['candidate_id']} | {a['oracle_tier']} | "
                     f"{a['title']} | {a['company']} | {a['yoe']} | {a['location']} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
