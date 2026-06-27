# Defend Your Work — Stage 5 Interview Prep

The spec says Stage 5 is a 30-minute video call where you "walk through architecture, defend design choices, demonstrate familiarity with your own code." Since you didn't write this codebase from scratch, you must read this and understand the *why*.

## Core Design Philosophy
We read the JD for what it **means**, not what it says. We didn't build a keyword matcher; we built a system that models the actual candidate traits the JD demands, penalizes explicitly stated disqualifiers, and survives the dataset traps.

## Likely Questions & Your Answers

**1. Why didn't you use an LLM (GPT-4 / Claude / Llama) to rank the candidates?**
*Answer:* "Compute limits. The Stage-3 contract restricts the ranking step to 5 minutes on CPU with no network. We can't hit an API, and running a local LLM over 100K candidates on CPU in 5 minutes is mathematically impossible. The JD itself says 'a system that calls GPT-4 per candidate cannot scale to a 200K pool in production'. So we built a fast, rigorous semantic + feature blend instead."

**2. How do you find semantic relevance without an LLM?**
*Answer:* "We use a TF-IDF vector space over the candidate's *career narratives* (where real work is described), scored against a query we wrote to capture the JD's *intent* (ranking, retrieval, production deployment, evaluation). This is fast, fits the CPU budget, and surfaces strong engineers who don't use trendy buzzwords."
*(If you did the offline embedding upgrade: "We also built an offline sentence-transformer embedding index during the allowed pre-compute phase. `rank.py` loads it locally with no network for a quality bump.")*

**3. How did you catch the honeypots?**
*Answer:* "In `traps.py`. Honeypots are arithmetically impossible profiles. We check if tenure exceeds the elapsed wall-clock time since their start date, if a single role is longer than their whole claimed career, or if they claim 'expert' in 5+ skills they've used for 0 months. If they fail the math, we zero their score."

**4. How do you handle keyword stuffers (e.g. an HR Manager who lists RAG/Pinecone)?**
*Answer:* "Also in `traps.py` (`keyword_stuffer_score`). We look for non-engineering current titles (HR, Marketing, Accountant) paired with AI buzzwords *and* no genuine engineering career history. The scoring penalizes them heavily."

**5. How are behavioral signals (Redrob signals) used?**
*Answer:* "As a multiplier, not an additive feature (`behavioral_multiplier`). The JD says a perfect candidate who hasn't logged in for 6 months and ignores recruiters isn't hireable. If `last_active` is recent, `open_to_work` is true, and response rate is high, they get a boost (up to 1.15x). If they ghost, their entire score is shrunk (down to 0.45x)."

**6. Why is consulting/services background penalized?**
*Answer:* "The JD explicitly says 'People who have only worked at consulting firms (TCS, Infosys, Wipro...) - we will not move forward'. So in `features.py` we measure the fraction of their career at those firms. BUT we included the JD's carve-out: if they have prior product-company experience, the consulting penalty shrinks."

**7. How do you validate your ranking without a leaderboard?**
*Answer:* "We built an offline harness (`eval.evaluate`) using a JD-derived proxy oracle
that assigns a relevance tier (0-5) per candidate, then computes NDCG@10, NDCG@50, MAP,
P@10, plus an ablation study removing features one at a time. Important honesty point if
asked: the oracle reuses the ranker's own `build_features`/`traps` code, restated as
discrete tiers -- so this is a *self-consistency check*, not independent validation. It
tells us the score behaves the way our stated JD rules say it should, and the ablation
tells us which components actually matter. It does NOT prove the ranking matches Redrob's
hidden ground truth -- nothing offline can prove that without labeled data. Say this
directly if asked rather than overselling the proxy numbers as 'proof it works.'"

## Your prep tasks
1. **Read `scoring.py`**: know what the 12 weights in `WEIGHTS` are.
2. **Read `traps.py`**: understand the 6 `honeypot_report` rules.
3. **Read `eval/relevance_oracle.py`**: understand how we mapped the JD to proxy tiers 0-5 so we could calculate NDCG without the ground truth.
