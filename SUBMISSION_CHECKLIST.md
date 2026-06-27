# India Runs / Redrob — Submission Checklist (spec v4)

Everything the organizers require, what's done, and what **you** must still do.
Tick every box before you upload. You get **only 3 submissions total** — the last
valid one counts.

---

## 1. The three required deliverables (§10)

- [x] **CSV file** — top-100 ranking (`submission.csv`, produced by `rank.py`).
- [ ] **Deck / PPT converted to PDF** explaining your approach. (Reuse / adapt
      `Redrob_Ranker_Approach.pdf` from the original package — update the weights table
      and the "4 must-have skill groups + offline-embedding option" so it matches this
      code.)
- [ ] **GitHub repo** — clean, complete, working code (this folder) with real commit
      history.

---

## 2. CSV format rules (§2–§3) — all enforced by `validate_submission.py` ✅

- [x] Columns exactly: `candidate_id,rank,score,reasoning` (in this order).
- [x] Exactly **100** data rows (+1 header). UTF-8 encoded. `.csv` (not xlsx/json).
- [x] Ranks 1..100, each exactly once. Candidate_ids unique and all exist in the pool.
- [x] `score` **non-increasing** with rank (ties allowed; broken by candidate_id asc).
- [x] Scores not all identical.
- [x] Reasoning: non-empty, varied, factual, no hallucination, tone matches rank.
- [ ] **Filename = your registered participant/team ID** + `.csv` (e.g. `team_xxx.csv`).
      Rename `submission.csv` before upload.

Run before every upload:
```bash
python validate_submission.py --submission ./submission.csv --candidates ./candidates.jsonl
```

---

## 3. Compute constraints (§3) — the ranking step ✅

| Constraint | Limit | This ranker |
|---|---|---|
| Runtime | ≤ 5 min wall-clock (100K) | ~150 s ✅ |
| Memory | ≤ 16 GB | well under ✅ |
| Compute | CPU only, no GPU | ✅ |
| **Network** | **OFF during ranking** | ✅ no API calls |
| Disk | ≤ 5 GB intermediate | ✅ |

Stage-3 reproduction: organizers re-run **`rank.py`** in a sandboxed Docker
(5 min / 16 GB / no GPU / no network). This package reproduces with **no setup beyond
`pip install -r requirements.txt`**.

---

## 4. Honeypot filter (§7) ✅

- [x] Honeypot rate in top 100 must be **≤ 10%** (else instant DQ). This ranker zeroes
      impossible profiles → **0% in the top 100** (verified).

---

## 5. SHOULD YOU USE AN LLM? — short answer: **NOT at rank time. Yes, allowed in dev.**

- **At rank time (the code that makes the CSV): NO.** §3 bans network/hosted-LLM calls
  (OpenAI/Anthropic/Cohere/Gemini/Groq). The JD's own note says the right answer is
  *not* an LLM-per-candidate system — "a system that calls GPT-4/Claude per candidate
  cannot scale to 200K in production." A per-candidate LLM **cannot fit the 5-min CPU
  budget** and would fail Stage 3. **This ranker uses no LLM at rank time.** ✅
- **As a pre-computation step: allowed but optional and risky.** §10.3 lets
  pre-computation exceed 5 min. You *could* cache LLM re-rank scores offline, but it
  adds reproduction fragility (API keys, rate limits, can't re-run in the sandbox) and
  invites Stage-4/5 scrutiny. **We deliberately don't.** If you want a quality bump,
  use the **offline open-source embedding** path (`precompute_embeddings.py`) instead —
  it's a local model, **not a hosted LLM**, and `rank.py --cache` reads it with no
  network.
- **As a dev tool (writing/reviewing code): YES, allowed and expected.** §10.4 — just
  **declare it honestly** in `submission_metadata.yaml`. It is *not* penalised; only
  contradicting your declaration in the interview is.

---

## 6. SANDBOX / DEMO LINK — yes, a website/Colab is **mandatory** (§10.5)

- [ ] You **must** provide a working hosted environment that runs the ranker on a
      ≤100-candidate sample end-to-end and outputs a ranked CSV in ≤5 min CPU.
      Without it your submission is flagged at Stage 1.
- **Acceptable platforms (pick ONE):** Google **Colab** (notebook that runs end-to-end)
  · HuggingFace Spaces (free tier) · Streamlit Cloud (free tier) · Replit (public) ·
  Binder · or a `docker pull`/`docker run` recipe in the README.
- **Easiest = Colab.** A minimal notebook should:
  1. `!git clone <your repo>` and `!pip install -r requirements.txt`
  2. upload or load `sample_candidates.jsonl` (use `make_sample.py`)
  3. `!python rank.py --candidates sample_candidates.jsonl --out out.csv`
  4. display `out.csv`
  Paste that Colab share link into `submission_metadata.yaml` → `sandbox_demo_link`.
- It does **not** need to handle the full 100K — small-sample reproducibility is the point.

---

## 7. Portal metadata (§10.2) — fill `submission_metadata.yaml`

- [ ] Team name · primary contact name / email / phone · team member list
- [ ] **GitHub repository URL** (reachable; private OK if you can grant org access)
- [ ] **Sandbox / demo link** (see §6 above)
- [ ] AI tools declared (honest) · compute environment one-liner · methodology ≤200 words

---

## 8. Evaluation stages (§5) — what actually decides the win

| Stage | What happens | Your status |
|---|---|---|
| 1. Format validation | auto-validator | ✅ passes |
| 2. Scoring | composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10 (top-10 = half) | depends on tier alignment |
| 3. Code reproduction + honeypot | re-run `rank.py` (5 min/16 GB/no GPU/no net); honeypot ≤10% | ✅ reproducible, 0% honeypots |
| 4. Manual review | reasoning quality (10 sampled rows), methodology, **git history authenticity**, code quality | ⚠️ needs real commit history |
| 5. Defend-your-work interview | 30-min call: walk through architecture, defend choices | ⚠️ you must own the code |

---

## 9. Your remaining TO-DO (in priority order)

1. **Rename CSV** to your registered team/participant ID, re-run the validator.
2. **Push to GitHub with REAL, incremental commit history** — not one "initial commit"
   dump (Stage 4 explicitly filters "flat git history"). Suggested commit sequence:
   data exploration → traps.py → features.py → scoring.py → reasoning.py → rank.py →
   weight tuning → README/checklist.
3. **Create the Colab sandbox** and paste the link into the metadata.
4. **Fill every TODO** in `submission_metadata.yaml`.
5. **Convert the approach deck to PDF** (update it to match this code: 12 weights, the
   4 must-have groups with plain-language evidence, the optional offline-embedding path).
6. **Prep for the interview** — be able to explain: why TF-IDF over a hosted LLM (the
   compute contract), the honeypot arithmetic, why behavioral signals multiply rather
   than add, and how a plain-language Tier-5 (e.g. a "Niramai/Netflix" ranking engineer
   who never writes "RAG") still reaches the top 10.

### Optional, only if you have time and will test the full Stage-3 run
- Build the offline embedding index (`precompute_embeddings.py`), commit it (or its
  build script), and `rank.py --cache` for a likely NDCG@10 bump. Re-validate runtime.
