"""
reasoning.py
============
1-2 sentence factual, extractive `reasoning` per candidate. Every clause is assembled
from fields that exist in the record -- no hallucination, no network, no LLM, so it
reproduces inside the Stage-3 sandbox. Clauses depend on each candidate's own signals
and concerns (so rows differ), and tone tracks the rank.
"""
from __future__ import annotations
from typing import Dict, List

from .traps import _parse, TODAY


def _years(c: Dict) -> float:
    return float((c.get("profile") or {}).get("years_of_experience", 0) or 0)


def _top_relevant_skills(c: Dict, limit: int = 3) -> List[str]:
    want = ["search", "rank", "recommend", "retrieval", "embedding", "nlp",
            "ml", "machine learning", "llm", "vector", "python", "faiss",
            "elasticsearch", "opensearch", "pinecone", "ndcg"]
    picked = []
    for s in c.get("skills", []) or []:
        nm = (s.get("name") or "")
        if any(w in nm.lower() for w in want) and s.get("proficiency") in ("advanced", "expert"):
            picked.append(nm)
        if len(picked) >= limit:
            break
    return picked


def _strongest_role_phrase(c: Dict) -> str:
    best = None
    for h in c.get("career_history", []) or []:
        d = (h.get("description") or "").lower()
        t = (h.get("title") or "").lower()
        score = sum(1 for k in ["rank", "search", "recommend", "retrieval",
                                "embedding", "relevance", "personaliz"] if k in d)
        score += 2 if any(k in t for k in ["ml", "ai", "search", "data scien"]) else 0
        if best is None or score > best[0]:
            best = (score, h)
    if best and best[0] > 0:
        h = best[1]
        return f'{h.get("title") or ""} at {h.get("company") or ""}'
    return ""


def build_reasoning(c: Dict, feats: Dict, score: float, kw_stuffer: float,
                    is_honeypot: bool) -> str:
    p = (c.get("profile") or {})
    yoe = _years(c)
    title = p.get("current_title") or "Unspecified role"
    bits: List[str] = []

    lead = f"{title} with {yoe:.1f} yrs"
    if feats["applied_ml_years"] >= 1:
        lead += f" (~{feats['applied_ml_years']:.0f} in applied ML/data roles)"
    bits.append(lead)

    role_phrase = _strongest_role_phrase(c)
    if feats["ranking_retrieval"] > 0.25 and role_phrase:
        bits.append(f"built ranking/retrieval work as {role_phrase}")
    elif role_phrase and feats["nlp_ir"] > 0.2:
        bits.append(f"NLP/IR experience as {role_phrase}")
    elif role_phrase:
        bits.append(f"closest relevant role: {role_phrase}")

    cov = int(round(feats["must_skills"] * 4))
    if cov >= 3:
        bits.append(f"covers {cov}/4 must-have skill areas")

    skills = _top_relevant_skills(c)
    if skills:
        bits.append("strong on " + ", ".join(skills))

    s = c.get("redrob_signals") or {}
    rr = float(s.get("recruiter_response_rate", 0) or 0)
    la = _parse(s.get("last_active_date"))
    days = (TODAY - la).days if la else None
    avail = []
    if days is not None and days <= 45:
        avail.append("recently active")
    avail.append(f"{rr:.0%} recruiter response")
    if s.get("open_to_work_flag"):
        avail.append("open to work")
    bits.append(", ".join(avail))

    loc = p.get("location") or "location unspecified"
    if feats["location_fit"] >= 1.0:
        bits.append(f"based in {loc}")
    elif feats["location_fit"] < 0.7:
        reloc = "willing to relocate" if s.get("willing_to_relocate") else "relocation unclear"
        bits.append(f"{loc} ({reloc})")

    concerns = []
    if kw_stuffer > 0.3:
        concerns.append("AI skills listed but career is non-engineering")
    if feats["consulting"] > 0.5:
        concerns.append("career largely at services firms")
    if feats["research_only"] > 0.4:
        concerns.append("research-leaning, light on production")
    if feats["langchain_only"] > 0.4:
        concerns.append("recent LLM-wrapper work, limited pre-LLM ML depth")
    if feats["job_hop"] > 0.5:
        concerns.append("frequent short stints")
    if feats["cv_speech"] > 0.3 and feats["nlp_ir"] < 0.2:
        concerns.append("CV/speech background, light NLP/IR")
    npd = s.get("notice_period_days")
    if isinstance(npd, (int, float)) and npd > 60:
        concerns.append(f"{int(npd)}-day notice")
    if days is not None and days > 120:
        concerns.append("inactive for months")

    sentence = "; ".join(bits) + "."
    if concerns:
        sentence += " Concern: " + ", ".join(concerns[:2]) + "."
    return sentence
