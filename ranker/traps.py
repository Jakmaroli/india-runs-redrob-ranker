"""
traps.py
========
Trap & integrity detection for the Redrob candidate pool.

  1. Honeypots (~80)       -- subtly *impossible* profiles, forced to tier 0 in the
                              hidden ground truth. A honeypot rate > 10% in the top-100
                              is an instant disqualification, so we zero any profile
                              whose internal arithmetic cannot be true.
  2. Keyword stuffers      -- non-engineering titles whose skills list is packed with AI
                              buzzwords but whose career has nothing to do with ML/IR.
  3. Plain-language Tier-5  -- strong engineers who DON'T use buzzwords (handled
                              positively in features/scoring).
  4. Behavioral twins      -- near-identical profiles differing only in redrob_signals
                              (disambiguated by the behavioral multiplier in scoring).

Every flag points at a concrete field -- no black box decides "impossible".
"""
from __future__ import annotations
from datetime import date
from typing import Dict, List

TODAY = date(2026, 6, 23)

NON_ENGINEERING_TITLES = {
    "hr manager", "human resources", "accountant", "content writer",
    "graphic designer", "marketing manager", "sales executive",
    "customer support", "operations manager", "business analyst",
    "project manager", "civil engineer", "mechanical engineer",
    "recruiter", "talent acquisition", "ui designer", "ux designer",
}

AI_BUZZWORDS = {
    "machine learning", "deep learning", "nlp", "natural language processing",
    "llm", "fine-tuning llms", "fine-tuning", "rag", "transformers", "pytorch",
    "tensorflow", "computer vision", "image classification", "speech recognition",
    "generative ai", "langchain", "vector databases", "embeddings",
    "recommendation systems", "reinforcement learning", "neural networks",
}


def _parse(d) -> date | None:
    if not d:
        return None
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def honeypot_report(c: Dict) -> Dict:
    """Honeypot signals + boolean `is_honeypot`. Requires an unambiguous wall-clock
    violation OR two independent softer impossibilities (avoids false-positives)."""
    signals: List[str] = []
    p = (c.get("profile") or {})
    yoe = float(p.get("years_of_experience", 0) or 0)
    skills = c.get("skills", []) or []
    hist = c.get("career_history", []) or []

    for h in hist:
        sd, ed = _parse(h.get("start_date")), _parse(h.get("end_date")) or TODAY
        dm = int(h.get("duration_months", 0) or 0)
        if sd and dm > _months_between(sd, ed) + 3:
            signals.append("tenure_exceeds_elapsed_time")
            break

    if any(int(h.get("duration_months", 0) or 0) > yoe * 12 + 24 for h in hist):
        signals.append("single_role_longer_than_career")

    span_years = sum(int(h.get("duration_months", 0) or 0) for h in hist) / 12.0
    if span_years > yoe + 6:
        signals.append("career_months_far_exceed_yoe")

    n_expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and int(s.get("duration_months", 0) or 0) == 0
    )
    if n_expert_zero >= 5:
        signals.append("many_expert_skills_zero_months")

    if any(
        s.get("proficiency") == "expert"
        and int(s.get("duration_months", 0) or 0) > yoe * 12 + 36
        for s in skills
    ):
        signals.append("skill_tenure_exceeds_career")

    starts = [_parse(h.get("start_date")) for h in hist if _parse(h.get("start_date"))]
    if starts:
        earliest = min(starts)
        career_span_yrs = _months_between(earliest, TODAY) / 12.0
        if yoe > career_span_yrs + 5:
            signals.append("yoe_exceeds_time_since_first_job")

    is_hp = ("tenure_exceeds_elapsed_time" in signals) or (len(signals) >= 2)
    return {"is_honeypot": is_hp, "signals": signals}


def keyword_stuffer_score(c: Dict, has_engineering_career: bool) -> float:
    """0..1 -- non-engineering title + buzzword skills + no real engineering career."""
    title = ((c.get("profile") or {}).get("current_title") or "").strip().lower()
    skills = c.get("skills", []) or []
    buzz = sum(1 for s in skills if (s.get("name") or "").strip().lower() in AI_BUZZWORDS)
    if title not in NON_ENGINEERING_TITLES:
        return 0.0
    if has_engineering_career:
        return 0.0
    if buzz >= 4:
        return min(1.0, 0.6 + 0.1 * (buzz - 4))
    if buzz >= 1:
        return 0.4
    return 0.15
