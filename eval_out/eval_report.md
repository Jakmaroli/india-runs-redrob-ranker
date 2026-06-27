# Offline Evaluation Report -- Redrob Ranker

> Metrics use a **JD-derived proxy oracle** (`eval/relevance_oracle.py`) that shares feature-extraction code with the ranker itself -- this is a **self-consistency check**, not independent validation against ground truth. It confirms the score behaves the way the JD's stated rules say it should; it does not prove the ranking is correct. Read the **ablation deltas** for what each component contributes.

- Candidates evaluated: **100,000**
- Runtime: **145.5s**

## Proxy oracle tier distribution
| Tier | Count |
|---|---|
| 5 | 509 |
| 4 | 304 |
| 3 | 984 |
| 2 | 10,954 |
| 1 | 47,980 |
| 0 | 39,269 |

## Baseline metrics (competition composite formula)
| Metric | Value |
|---|---|
| proxy-NDCG@10 | 1.0000 |
| proxy-NDCG@50 | 0.9574 |
| proxy-MAP | 0.9612 |
| proxy-P@10 | 1.0000 |
| **proxy-composite** | **0.9814** |

_composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10_

## Trap robustness
| Check | Rate |
|---|---|
| honeypots in top-10 | 0.0% |
| honeypots in top-100 | 0.0% (DQ if >10%) |
| keyword-stuffers in top-100 | 0.0% |
| consulting-heavy in top-100 | 1.0% |

## Ablation study (remove one component, re-rank)
| Variant | proxy-composite | Δ vs baseline | overlap@10 | overlap@100 | honeypots@100 |
|---|---|---|---|---|---|
| no_semantic | 0.9894 | +0.0080 | 0.80 | 0.91 | 0 |
| no_must_skills | 0.9814 | -0.0000 | 1.00 | 0.92 | 0 |
| no_behavioral | 0.7683 | -0.2131 | 0.40 | 0.60 | 0 |
| no_location | 0.9787 | -0.0027 | 0.40 | 0.65 | 0 |
| no_penalties | 0.9215 | -0.0599 | 0.70 | 0.84 | 0 |
| no_traps | 0.9814 | +0.0000 | 1.00 | 1.00 | 0 |

_A negative Δ means removing that component HURTS quality -> it is pulling its weight. `no_traps` should leak honeypots into the top-100._

## Top-10 audit (baseline)
| Rank | ID | Tier | Title | Company | YoE | Location |
|---|---|---|---|---|---|---|
| 1 | CAND_0006567 | 5 | Senior AI Engineer | Meta | 7.9 | Noida, Uttar Pradesh |
| 2 | CAND_0018499 | 5 | Senior Machine Learning Engineer | Zomato | 7.2 | Noida, Uttar Pradesh |
| 3 | CAND_0081846 | 5 | Lead AI Engineer | Razorpay | 6.7 | Jaipur, Rajasthan |
| 4 | CAND_0006418 | 5 | Machine Learning Engineer | Verloop.io | 5.7 | Gurgaon, Haryana |
| 5 | CAND_0041669 | 5 | Recommendation Systems Engineer | CRED | 8.0 | Noida, Uttar Pradesh |
| 6 | CAND_0064326 | 5 | Search Engineer | Sarvam AI | 7.6 | Gurgaon, Haryana |
| 7 | CAND_0052328 | 5 | Recommendation Systems Engineer | Amazon | 6.5 | Pune, Maharashtra |
| 8 | CAND_0093193 | 5 | Senior Machine Learning Engineer | Niramai | 7.9 | Bangalore, Karnataka |
| 9 | CAND_0050454 | 5 | AI Engineer | Rephrase.ai | 6.8 | Delhi, Delhi |
| 10 | CAND_0066999 | 5 | Recommendation Systems Engineer | Microsoft | 5.9 | Delhi, Delhi |
