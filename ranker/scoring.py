"""
scoring.py
==========
    final_score = base_fit  x  behavioral_multiplier  x  location_fit  -  penalties

base_fit is a weighted blend (weights sum to 1.0, asserted at import). Penalties
capture the JD's explicit "do NOT want" list. Honeypots are zeroed. Every contribution
is explainable for the Stage-5 interview.
"""
from __future__ import annotations
from typing import Dict

ROLE_FIT = {
    "senior ai engineer": 1.0, "lead ai engineer": 1.0, "ai engineer": 0.97,
    "applied ml engineer": 0.98, "machine learning engineer": 0.96,
    "senior machine learning engineer": 0.98, "staff machine learning engineer": 0.97,
    "senior ml engineer": 1.0, "ml engineer": 0.93,
    "recommendation systems engineer": 0.99, "search engineer": 0.97,
    "ranking engineer": 0.98, "nlp engineer": 0.95, "senior nlp engineer": 0.97,
    "senior software engineer (ml)": 0.93, "senior applied scientist": 0.9,
    "applied scientist": 0.88,
    "data scientist": 0.78, "senior data scientist": 0.82,
    "ai specialist": 0.6, "ai research engineer": 0.62,
    "research scientist": 0.5,
    "computer vision engineer": 0.45,
    "junior ml engineer": 0.55,
    "data engineer": 0.62, "senior data engineer": 0.66, "analytics engineer": 0.58,
    "backend engineer": 0.5, "senior software engineer": 0.55, "software engineer": 0.5,
    "full stack developer": 0.4, "cloud engineer": 0.38, "devops engineer": 0.35,
    "data analyst": 0.4,
}
DEFAULT_ROLE_FIT = 0.12

WEIGHTS = {
    "semantic": 0.22,
    "role": 0.18,
    "must_skills": 0.14,
    "ranking_retrieval": 0.12,
    "applied_ml": 0.09,
    "experience_fit": 0.08,
    "evaluation": 0.05,
    "nlp_ir": 0.04,
    "shipped": 0.04,
    "product": 0.02,
    "assessment": 0.01,
    "education": 0.01,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"weights sum to {sum(WEIGHTS.values())}"


def role_fit(title: str) -> float:
    return ROLE_FIT.get((title or "").strip().lower(), DEFAULT_ROLE_FIT)


def base_fit(feats: Dict, semantic: float) -> float:
    applied = min(1.0, feats["applied_ml_years"] / 5.0)
    components = {
        "semantic": semantic,
        "role": role_fit(feats["title"]),
        "must_skills": feats["must_skills"],
        "ranking_retrieval": feats["ranking_retrieval"],
        "applied_ml": applied,
        "experience_fit": feats["experience_fit"],
        "evaluation": feats["evaluation"],
        "nlp_ir": feats["nlp_ir"],
        "shipped": feats["shipped"],
        "product": feats["product"],
        "assessment": feats["assessment"],
        "education": feats["education_fit"],
    }
    score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    if feats["cv_speech"] > 0.3 and feats["nlp_ir"] < 0.2:
        score *= 0.7
    return score


def penalties(feats: Dict, kw_stuffer: float) -> float:
    pen = 0.0
    pen += 0.45 * kw_stuffer
    pen += 0.18 * feats["job_hop"]
    # Consulting penalty, gated by the JD carve-out: "currently at one of these
    # companies but have prior product-company experience -- that's fine." So
    # product-company depth shrinks the services penalty.
    pen += 0.20 * feats["consulting"] * (1.0 - feats["product"])
    pen += 0.22 * feats["research_only"]
    pen += 0.15 * feats["langchain_only"]
    if not feats["has_engineering_career"]:
        pen += 0.30
    return pen


def score_candidate(feats: Dict, semantic: float, kw_stuffer: float,
                    is_honeypot: bool) -> float:
    if is_honeypot:
        return 0.0
    bf = base_fit(feats, semantic)
    modified = bf * feats["behavioral"] * feats["location_fit"]
    final = modified - penalties(feats, kw_stuffer)
    return max(0.0, min(1.0, final))
