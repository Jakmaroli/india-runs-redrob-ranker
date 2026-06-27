"""
relevance_oracle.py
===================
We have NO access to the hidden ground truth. To validate the ranker offline (exactly
the "design an evaluation framework for ranking systems" skill the JD asks for), we
build a transparent relevance oracle straight from the JD text and assign each
candidate a discrete tier 0-5.

IMPORTANT -- this is a *proxy*, not the real ground truth, AND it is NOT independent
of the ranker: it reuses build_features()/honeypot_report()/keyword_stuffer_score()
from the ranker itself, restated into discrete tiers with hard caps. Measured
correlation between this oracle's tiers and the ranker's own composite score is ~0.86
on a synthetic test set -- expected, since they share inputs. That makes this a
transparent self-consistency check (does the score agree with the JD rules it claims
to encode?), not blind validation. We report metrics as "proxy-NDCG@10" etc. and never
claim they equal the leaderboard score or prove the ranking is correct.

Tier semantics (aligned with the spec: tier 3+ == "relevant"; honeypots == tier 0):
  5  exemplary fit  -- ranking/search/recsys engineer, 6-8 yrs, product company,
                       covers all 4 must-have areas, NLP/IR, shipped, India tier-1,
                       open-to-work and responsive.
  4  strong fit     -- applied-ML at a product company with ranking/retrieval evidence,
                       5-9 yrs, most must-haves, available.
  3  relevant       -- ML/AI engineer with some ranking/IR evidence and ok availability.
  2  adjacent       -- ML-adjacent (data scientist/engineer) missing key dimensions.
  1  weak           -- engineer but role family far from the mandate.
  0  irrelevant/trap-- honeypot, keyword-stuffer, no engineering career, or hard
                       disqualifier (consulting-only / CV-without-NLP / unavailable).
"""
from __future__ import annotations
from typing import Dict

from ranker.features import build_features
from ranker.traps import honeypot_report, keyword_stuffer_score, TODAY, _parse


def relevance_tier(c: Dict) -> int:
    """Convenience wrapper that computes features/traps then the tier."""
    feats = build_features(c)
    hp = honeypot_report(c)["is_honeypot"]
    kw = keyword_stuffer_score(c, feats["has_engineering_career"])
    return tier_from(c, feats, hp, kw)


def tier_from(c: Dict, feats: Dict, hp: bool, kw: float) -> int:
    """Core tier logic using precomputed features/traps (no recomputation)."""
    # ---- Hard tier-0 conditions (traps / not a real candidate) ----
    if hp:
        return 0
    if kw > 0.5:
        return 0
    if not feats["has_engineering_career"]:
        return 0

    s = c.get("redrob_signals") or {}
    la = _parse(s.get("last_active_date"))
    days_inactive = (TODAY - la).days if la else 999
    available = bool(s.get("open_to_work_flag")) or days_inactive <= 90

    # ---- Build a coarse evidence score from independent buckets ----
    pts = 0.0
    # role family
    title = (feats["title"] or "").lower()
    if any(k in title for k in ["recommendation", "search", "ranking", "nlp",
                                 "ai engineer", "ml engineer", "machine learning"]):
        pts += 2.0
    elif any(k in title for k in ["applied scientist", "data scientist"]):
        pts += 1.0
    elif any(k in title for k in ["data engineer", "research"]):
        pts += 0.5
    # ranking/retrieval evidence + must-have coverage + nlp + shipped
    pts += 2.0 * feats["ranking_retrieval"]
    pts += 1.5 * feats["must_skills"]
    pts += 1.0 * feats["nlp_ir"]
    pts += 0.5 * feats["shipped"]
    pts += 1.0 * feats["experience_fit"]
    pts += 1.0 * min(1.0, feats["applied_ml_years"] / 5.0)
    pts += 0.8 * feats["product"]
    # location (JD: India tier-1 preferred)
    pts += 0.5 * feats["location_fit"]

    # ---- Map evidence points -> base tier ----
    if pts >= 8.0:
        tier = 5
    elif pts >= 6.5:
        tier = 4
    elif pts >= 4.5:
        tier = 3
    elif pts >= 3.0:
        tier = 2
    else:
        tier = 1

    # ---- JD hard-cap disqualifiers (independent of the points) ----
    # Consulting-ONLY careers are capped; the JD carve-out spares those with real
    # prior product-company experience.
    if feats["consulting"] > 0.5 and feats["product"] < 0.25:
        tier = min(tier, 1)                 # "we will not move forward"
    if feats["cv_speech"] > 0.3 and feats["nlp_ir"] < 0.2:
        tier = min(tier, 1)                 # CV/speech without NLP/IR
    if feats["research_only"] > 0.6:
        tier = min(tier, 2)                 # pure research, no production
    if feats["langchain_only"] > 0.6:
        tier = min(tier, 2)                 # recent LangChain-only
    if not available:
        tier = min(tier, 2)                 # unreachable == not a hire
    return tier
