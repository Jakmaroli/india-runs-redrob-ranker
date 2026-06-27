"""
features.py
===========
Turns a raw candidate record into the interpretable features the scorer consumes.
Read the JD for what it MEANS, not what it SAYS.

The 4 must-have skill groups are checked two ways:
  * `terms`    -- explicit/buzzword terminology, matched anywhere in the profile.
  * `evidence` -- plain-language descriptions of the SAME work, matched specifically
                  against career_history descriptions (harder to buzzword-stuff).
So a genuine engineer who never writes "RAG" still gets credit.

DEFENSIVE NULL HANDLING: real-world aggregated/scraped data (100k rows) routinely has
fields that are present-but-null, not just missing -- e.g. {"current_title": null} or
{"redrob_signals": null} or even a missing "profile" key entirely. `.get(key, default)`
ONLY substitutes the default when the KEY is absent, NOT when its value is explicitly
None -- so every nested-field access below uses `x.get(k) or fallback` / `c.get("k") or
{}` patterns instead. This was reproduced directly: a candidate with current_title=None
crashed the entire 100k-row pipeline with a single TypeError. One malformed row out of
100,000 should never abort the whole run.
"""
from __future__ import annotations
from typing import Dict, List

from .traps import TODAY, _parse

RANKING_RETRIEVAL_TERMS = [
    "ranking", "rank ", "re-rank", "rerank", "learning to rank", "ltr",
    "retrieval", "search engine", "search relevance", "semantic search",
    "recommendation", "recommender", "recsys", "personalization",
    "embedding", "embeddings", "vector search", "vector database", "ann ",
    "nearest neighbor", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "bm25", "information retrieval",
]
EVAL_TERMS = ["ndcg", "mrr", "map@", "mean average precision", "a/b test",
              "ab test", "offline evaluation", "online evaluation", "precision@",
              "recall@", "ctr", "engagement metric"]
NLP_TERMS = ["nlp", "natural language", "text", "language model", "llm",
             "transformer", "bert", "sentence-transformer", "named entity",
             "summarization", "question answering", "rag", "tokeniz"]
CV_SPEECH_TERMS = ["computer vision", "image classification", "object detection",
                   "segmentation", "ocr", "speech recognition", "asr ", "tts ",
                   "robotics", "lidar", "pose estimation"]
PRELLM_ML_TERMS = ["xgboost", "lightgbm", "random forest", "logistic regression",
                   "gradient boost", "scikit", "feature engineering", "svm",
                   "matrix factorization", "collaborative filtering", "word2vec",
                   "tf-idf", "spark mllib"]
LANGCHAIN_TERMS = ["langchain", "llamaindex", "openai api", "prompt engineering",
                   "gpt-4", "chatgpt wrapper"]
SHIPPED_TERMS = ["shipped", "deployed", "in production", "production", "served",
                 "real users", "at scale", "launched", "rolled out", "live "]

MUST_SKILL_GROUPS = {
    "embeddings_retrieval": {
        "terms": ["sentence-transformers", "sentence transformers", "openai embeddings",
                  "bge", "e5 embedding", "embeddings", "dense retrieval", "semantic search",
                  "bi-encoder", "cross-encoder", "rag", "vector search", "embedding model",
                  "text embedding", "dense passage retrieval"],
        "evidence": ["recommendation system", "recommender system", "recommendation engine",
                     "relevance scoring", "relevance ranking", "content ranking",
                     "ranking system", "ranking model", "ranking algorithm",
                     "matching algorithm", "similarity matching", "content matching",
                     "personalization engine", "collaborative filtering",
                     "search relevance", "connect them to the most relevant",
                     "what to show", "what users are looking for"],
    },
    "vector_db": {
        "terms": ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
                  "elasticsearch", "vector database", "vector db", "hybrid search",
                  "approximate nearest neighbor", "ann search", "hnsw", "ivf"],
        "evidence": ["search index", "search infrastructure", "retrieval infrastructure",
                     "indexing pipeline", "nearest neighbor search", "similarity index",
                     "across a large dataset", "large dataset"],
    },
    "python_engineering": {
        "terms": ["python", "pytorch", "tensorflow", "keras", "numpy", "pandas",
                  "fastapi", "flask", "pyspark"],
        "evidence": [],
    },
    "ranking_eval": {
        "terms": ["ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
                  "precision@", "recall@", "a/b testing", "ab testing",
                  "offline evaluation", "ranking evaluation", "ir metrics",
                  "information retrieval", "learning to rank", "ltr", "evaluation framework"],
        "evidence": ["evaluation pipeline", "model comparison", "compare model versions",
                     "before rollout", "ranking quality", "click-through rate",
                     "click through rate", "offline testing", "version comparison",
                     "measured impact", "online experiment"],
    },
}

# JD names TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini explicitly; we extend with
# other pure services / BPM / staffing firms of the same genre.
CONSULTING_FIRMS = {"tcs", "tata consultancy", "infosys", "wipro", "accenture",
                    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
                    "mphasis", "ltimindtree", "lti mindtree", "deloitte",
                    "ibm global services", "hexaware", "persistent systems",
                    "genpact", "wns", "exl service", "exl ", "concentrix",
                    "teleperformance", "dxc", "conduent", "ntt data", "sutherland",
                    "firstsource", "wpro", "birlasoft", "coforge", "zensar"}

RESEARCH_MARKERS = ["research scientist", "phd", "postdoc", "research fellow",
                    "academic", "university", "institute", "publication",
                    "research engineer", "research lab"]

PRODUCT_INDUSTRY_HINTS = ["product", "saas", "internet", "consumer", "e-commerce",
                          "fintech", "edtech", "marketplace", "technology", "software",
                          "media", "healthtech ai", "ai"]

INDIA_TIER1 = {"pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi",
               "gurgaon", "gurugram", "bengaluru", "bangalore", "ncr", "navi mumbai",
               "thane", "greater noida"}

ENGINEERING_TITLE_TOKENS = ["engineer", "developer", "scientist", "programmer",
                            "architect", "sde", "ml ", "ai ", "data "]

EDU_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45}


def _candidate_text(c: Dict) -> str:
    p = (c.get("profile") or {})
    # FIX: same None-vs-missing-key issue as textmatch.candidate_doc -- .get(k, "")
    # doesn't guard against an explicit None value. Guarded with `or ""`.
    parts = [p.get("headline") or "", p.get("summary") or "", p.get("current_title") or ""]
    for h in c.get("career_history", []) or []:
        parts += [h.get("title") or "", h.get("description") or "", h.get("industry") or ""]
    for s in c.get("skills", []) or []:
        parts.append(s.get("name") or "")
    return " ".join(parts).lower()


def _career_text(c: Dict) -> str:
    return " ".join((h.get("description", "") or "") + " " + (h.get("title", "") or "")
                    for h in (c.get("career_history", []) or [])).lower()


def _skills_text(c: Dict) -> str:
    return " ".join((s.get("name", "") or "") for s in (c.get("skills", []) or [])).lower()


def _count_terms(text: str, terms: List[str]) -> int:
    return sum(1 for t in terms if t in text)


def has_engineering_career(c: Dict) -> bool:
    for h in c.get("career_history", []) or []:
        t = (h.get("title") or "").lower()
        if any(tok in t for tok in ENGINEERING_TITLE_TOKENS):
            return True
    return False


def must_skill_coverage(c: Dict) -> float:
    skills = _skills_text(c)
    prof = _candidate_text(c)
    career = _career_text(c)
    satisfied = 0
    for g in MUST_SKILL_GROUPS.values():
        term_hit = any(t in skills or t in prof for t in g["terms"])
        ev_hit = any(e in career for e in g["evidence"])
        if term_hit or ev_hit:
            satisfied += 1
    return satisfied / len(MUST_SKILL_GROUPS)


def applied_ml_years(c: Dict) -> float:
    yrs = 0.0
    for h in c.get("career_history", []) or []:
        t = (h.get("title") or "").lower()
        d = (h.get("description") or "").lower()
        is_ml = any(k in t for k in ["ml", "ai", "machine learning", "data scien",
                                     "applied scien", "nlp", "search", "recommend",
                                     "research engineer"])
        is_ml = is_ml or _count_terms(d, RANKING_RETRIEVAL_TERMS + NLP_TERMS) >= 2
        if is_ml:
            yrs += int(h.get("duration_months", 0) or 0) / 12.0
    return yrs


def job_hop_score(c: Dict) -> float:
    hist = c.get("career_history", []) or []
    completed = [h for h in hist if not h.get("is_current")]
    if not completed:
        return 0.0
    short = sum(1 for h in completed if int(h.get("duration_months", 0) or 0) < 18)
    return min(1.0, short / max(3, len(completed)))


def consulting_career_score(c: Dict) -> float:
    hist = c.get("career_history", []) or []
    total = sum(int(h.get("duration_months", 0) or 0) for h in hist) or 1
    consult = 0
    for h in hist:
        comp = (h.get("company") or "").lower()
        if any(f in comp for f in CONSULTING_FIRMS):
            consult += int(h.get("duration_months", 0) or 0)
    return consult / total


def product_company_score(c: Dict) -> float:
    hist = c.get("career_history", []) or []
    total = sum(int(h.get("duration_months", 0) or 0) for h in hist) or 1
    product = 0
    for h in hist:
        comp = (h.get("company") or "").lower()
        ind = (h.get("industry") or "").lower()
        # Any services / consulting / BPM employer is NOT a product company,
        # regardless of buzzwords in the industry string (e.g. "AI Services").
        if any(f in comp for f in CONSULTING_FIRMS) or "services" in ind or "consulting" in ind:
            continue
        if any(hint in ind for hint in PRODUCT_INDUSTRY_HINTS) or ind == "":
            product += int(h.get("duration_months", 0) or 0)
    return product / total


def research_only_score(c: Dict, text: str) -> float:
    research_hits = _count_terms(text, RESEARCH_MARKERS)
    shipped_hits = _count_terms(text, SHIPPED_TERMS)
    if research_hits >= 2 and shipped_hits == 0:
        return 1.0
    if research_hits >= 1 and shipped_hits == 0:
        return 0.5
    return 0.0


def langchain_only_score(c: Dict, text: str) -> float:
    lc = _count_terms(text, LANGCHAIN_TERMS)
    prellm = _count_terms(text, PRELLM_ML_TERMS)
    if lc >= 1 and prellm == 0:
        return min(1.0, 0.5 + 0.2 * lc)
    return 0.0


def location_fit(c: Dict) -> float:
    p = (c.get("profile") or {})
    loc = (p.get("location") or "").lower()
    country = (p.get("country") or "").lower()
    reloc = bool((c.get("redrob_signals") or {}).get("willing_to_relocate"))
    if any(city in loc for city in INDIA_TIER1):
        return 1.0
    if country == "india":
        return 0.9 if reloc else 0.8
    return 0.6 if reloc else 0.4


def experience_fit(c: Dict) -> float:
    yoe = float((c.get("profile") or {}).get("years_of_experience", 0) or 0)
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe <= 9:
        return 0.9
    if 4 <= yoe < 5 or 9 < yoe <= 11:
        return 0.7
    if 3 <= yoe < 4 or 11 < yoe <= 13:
        return 0.45
    return 0.25


def education_fit(c: Dict) -> float:
    best = 0.5
    for e in c.get("education", []) or []:
        t = (e.get("tier") or "").lower()
        best = max(best, EDU_TIER_SCORE.get(t, 0.5))
    return best


def concept_features(text: str) -> Dict[str, float]:
    rr = _count_terms(text, RANKING_RETRIEVAL_TERMS)
    ev = _count_terms(text, EVAL_TERMS)
    nlp = _count_terms(text, NLP_TERMS)
    cv = _count_terms(text, CV_SPEECH_TERMS)
    prellm = _count_terms(text, PRELLM_ML_TERMS)
    shipped = _count_terms(text, SHIPPED_TERMS)
    return {
        "ranking_retrieval": min(1.0, rr / 4.0),
        "evaluation": min(1.0, ev / 2.0),
        "nlp_ir": min(1.0, nlp / 3.0),
        "cv_speech": min(1.0, cv / 3.0),
        "prellm_ml": min(1.0, prellm / 3.0),
        "shipped": min(1.0, shipped / 2.0),
        "_raw": {"rr": rr, "ev": ev, "nlp": nlp, "cv": cv,
                 "prellm": prellm, "shipped": shipped},
    }


def behavioral_multiplier(c: Dict) -> float:
    """0.45 .. 1.15 -- availability/quality modifier from redrob_signals."""
    s = c.get("redrob_signals") or {}
    m = 1.0
    la = _parse(s.get("last_active_date"))
    if la:
        days = (TODAY - la).days
        if days <= 30:
            m *= 1.06
        elif days <= 90:
            m *= 1.0
        elif days <= 180:
            m *= 0.85
        else:
            m *= 0.6
    rr = float(s.get("recruiter_response_rate", 0) or 0)
    m *= 0.8 + 0.35 * rr
    if s.get("open_to_work_flag"):
        m *= 1.06
    else:
        m *= 0.88
    npd = s.get("notice_period_days")
    if isinstance(npd, (int, float)):
        if npd <= 30:
            m *= 1.03
        elif npd <= 60:
            m *= 0.98
        elif npd <= 90:
            m *= 0.92
        else:
            m *= 0.85
    ic = s.get("interview_completion_rate", None)
    if ic is not None and ic >= 0 and ic < 0.4:
        m *= 0.9
    gh = s.get("github_activity_score", None)
    if isinstance(gh, (int, float)) and gh >= 60:
        m *= 1.02
    if s.get("verified_email"):
        m *= 1.01
    return max(0.45, min(1.15, m))


def assessment_alignment(c: Dict) -> float:
    s = c.get("redrob_signals") or {}
    scores = s.get("skill_assessment_scores", {}) or {}
    relevant = {k: v for k, v in scores.items()
                if any(w in k.lower() for w in
                       ["nlp", "ml", "machine", "search", "rank", "recommend",
                        "retrieval", "embedding", "llm", "data", "python"])}
    if not relevant:
        return 0.5
    avg = sum(relevant.values()) / len(relevant)
    return max(0.0, min(1.0, avg / 100.0))


def build_features(c: Dict) -> Dict:
    text = _candidate_text(c)
    eng_career = has_engineering_career(c)
    feats = {
        "candidate_id": c["candidate_id"],
        "title": (c.get("profile") or {}).get("current_title", ""),
        "yoe": float((c.get("profile") or {}).get("years_of_experience", 0) or 0),
        "location": (c.get("profile") or {}).get("location", ""),
        "country": (c.get("profile") or {}).get("country", ""),
        "has_engineering_career": eng_career,
        "applied_ml_years": applied_ml_years(c),
        "must_skills": must_skill_coverage(c),
        "experience_fit": experience_fit(c),
        "education_fit": education_fit(c),
        "location_fit": location_fit(c),
        "job_hop": job_hop_score(c),
        "consulting": consulting_career_score(c),
        "product": product_company_score(c),
        "research_only": research_only_score(c, text),
        "langchain_only": langchain_only_score(c, text),
        "behavioral": behavioral_multiplier(c),
        "assessment": assessment_alignment(c),
    }
    feats.update(concept_features(text))
    return feats
