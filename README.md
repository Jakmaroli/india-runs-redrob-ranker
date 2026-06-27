# India Runs · Track 1 — Intelligent Candidate Discovery & Ranking

A hybrid, **fully-offline** candidate ranker for the Redrob *"Senior AI Engineer —
Founding Team"* job description. It ranks the top 100 of the 100,000-candidate pool by
**understanding what the role needs**, not by matching keywords — and it survives the
dataset's traps (keyword stuffers, honeypots, behavioral mismatches).

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That single command reproduces `submission.csv`. **CPU-only, no GPU, no network, no
hosted LLM** — verified at ~150s for the full 100K pool (well under the 5-min / 16-GB
Stage-3 contract), **0 honeypots in the top 100**, and passes every format rule.

---

## TL;DR — the idea

> The right answer is **not** "find candidates whose skills section has the most AI
> keywords." That's a trap. The right answer is reasoning about the gap between what
> the JD *says* and what it *means*. — the JD itself.

We never trust a skill list on its own. We score the **career narrative** (what a
person actually built), the **structured fit** (experience, role family, product-vs-
services, NLP/IR-vs-CV, the JD's 4 must-have skills), the **integrity** (is the profile
even arithmetically possible?), and the **behavioral availability** (are they reachable
and hireable?).

```
final_score = base_fit × behavioral_multiplier × location_fit − trap_penalties
```

Honeypots are zeroed. Keyword stuffers, job-hoppers, consulting-only careers,
research-without-production, and recent-LangChain-only profiles are down-weighted —
exactly per the JD's "do NOT want" section.

---

## Architecture

```
candidates.jsonl
      │
      ▼
 textmatch.py   ── semantic relevance: TF-IDF over career narratives, cosine vs a
      │             JD-INTENT query (default). OPTIONAL: a precomputed offline
      │             sentence-transformer index (precompute_embeddings.py) loaded with
      │             NO network at rank time. Surfaces plain-language strong fits;
      │             sinks buzzword-only profiles whose narrative doesn't match.
      ▼
 features.py    ── interpretable structured features: role-family fit · experience band
      │             (5–9, ideal 6–8) · applied-ML-at-product-company years · the JD's
      │             4 must-have skill groups (term OR plain-language career evidence) ·
      │             ranking/retrieval · evaluation rigor · NLP/IR vs CV/speech ·
      │             shipped-to-prod · job-hop · consulting · research-only ·
      │             langchain-only · education tier · behavioral multiplier · location
      ▼
 traps.py       ── honeypots (impossible arithmetic) → zeroed; keyword stuffers
      │             (non-eng title + buzzwords + no eng career) → heavily penalised
      ▼
 scoring.py     ── weighted blend (Σ weights = 1.0, asserted) → × behavioral × location
      │             → minus penalties → honeypots forced to 0
      ▼
 reasoning.py   ── factual, extractive, per-candidate reasoning (no hallucination,
      │             tone tracks rank, honest concerns included)
      ▼
   submission.csv  (top 100, unique ranks, non-increasing scores)
```

### Why TF-IDF (and an optional offline embedding upgrade), not a hosted LLM
The ranking step is contractually **CPU-only, no network, ≤5 min for 100K**. A sparse
TF-IDF index meets that with zero downloads and is fully reproducible. For a top-10
quality boost you can run `precompute_embeddings.py` **once, offline** to build a
sentence-transformer index (an open-source local model — *not* a hosted LLM/API); then
`rank.py --cache ./cache/` loads it with **no network**. The two paths are
interchangeable and both honour the contract.

---

## Scoring signals (single source of truth = `scoring.WEIGHTS`)

| Signal | Weight | Source |
|---|---|---|
| Semantic similarity (narrative ↔ JD intent) | 0.22 | `textmatch` (TF-IDF or offline embeddings) |
| Role / title family fit | 0.18 | `scoring.ROLE_FIT` |
| Must-have skill coverage (4 JD groups) | 0.14 | `features.must_skill_coverage` (term **or** career evidence) |
| Ranking / retrieval / recsys evidence | 0.12 | `features.concept_features` |
| Applied-ML-at-product-company depth | 0.09 | `features.applied_ml_years` |
| Experience band (5–9, ideal 6–8) | 0.08 | `features.experience_fit` |
| Evaluation rigor (NDCG/MRR/MAP/A-B) | 0.05 | `features.concept_features` |
| NLP / IR grounding | 0.04 | `features.concept_features` |
| Shipped-to-production evidence | 0.04 | `features.concept_features` |
| Product-company experience | 0.02 | `features.product_company_score` |
| Assessment-verified competence | 0.01 | `redrob_signals.skill_assessment_scores` |
| Education tier | 0.01 | `features.education_fit` |
| **Total** | **1.00** | `assert` at import |

Then `× behavioral_multiplier × location_fit − penalties`; honeypots → 0.

**Penalties** (JD "do NOT want"): keyword stuffer (≤0.45), consulting-only career
(≤0.20), research-without-production (≤0.22), job-hopper (≤0.18), LangChain-only
(≤0.15), no real engineering career (0.30).

---

## How the traps are handled

| Trap | Detector | Effect |
|------|----------|--------|
| **Honeypots** (impossible profiles) | `traps.honeypot_report` — tenure > elapsed wall-clock time, a role longer than the whole career, career-months ≫ YOE, "expert in N skills, 0 months used", YOE > time since first job | score forced to **0** |
| **Keyword stuffers** | `traps.keyword_stuffer_score` — non-engineering title + buzzword skills + no real engineering career | heavy penalty + the 0.30 "no engineering career" penalty |
| **Plain-language Tier-5s** | semantic match on the *narrative* + must-skill *evidence* phrases checked against career history | surfaced (not keyword-gated) |
| **Behavioral twins** | `features.behavioral_multiplier` | the more available/responsive twin ranks higher |

Verified on the released pool: **0 honeypots and 0 keyword-stuffers in the top 100.**

---

## Repository layout

```
india_runs_best/
├── rank.py                  # single-command entrypoint → submission.csv
├── precompute_embeddings.py # OPTIONAL offline embedding index (no hosted LLM)
├── validate_submission.py   # local format + honeypot-rate validator
├── make_sample.py           # slice ≤100 candidates for the sandbox/demo
├── requirements.txt         # pinned, CPU-only deps
├── submission_metadata.yaml # portal metadata (fill the TODOs)
├── SUBMISSION_CHECKLIST.md  # every spec requirement + LLM / Colab guidance
└── ranker/
    ├── textmatch.py         # TF-IDF semantic layer + offline-embedding hook
    ├── features.py          # interpretable structured features
    ├── traps.py             # honeypot + keyword-stuffer detection
    ├── scoring.py           # weighted blend + penalties + modifiers
    └── reasoning.py         # factual extractive reasoning
```

## Reproduce / validate / demo

```bash
# Reproduce the submission (Stage-3 single command)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate format + honeypot rate before uploading
python validate_submission.py --submission ./submission.csv --candidates ./candidates.jsonl

# Build the ≤100-candidate sample for the hosted sandbox / Colab
python make_sample.py --candidates ./candidates.jsonl --n 100 --out ./sample_candidates.jsonl

# OPTIONAL offline-embedding upgrade (Phase 1 once, then rank with no network)
pip install sentence-transformers
python precompute_embeddings.py --candidates ./candidates.jsonl --out ./cache/
python rank.py --candidates ./candidates.jsonl --cache ./cache/ --out ./submission.csv
```
