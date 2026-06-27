#!/usr/bin/env python3
"""
validate_submission.py -- local format + honeypot validator. Run BEFORE uploading.

    python validate_submission.py --submission ./submission.csv --candidates ./candidates.jsonl
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json

from ranker.traps import honeypot_report


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()

    with _open(args.submission) as f:
        rows = list(csv.DictReader(f))

    cand_ids, cand_by_id = set(), {}
    skipped_malformed = 0
    with _open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            # FIX: a row missing "candidate_id" entirely (malformed real-world data)
            # used to crash this validator with KeyError before it could check anything.
            cid = c.get("candidate_id")
            if not cid:
                skipped_malformed += 1
                continue
            cand_ids.add(cid)
            cand_by_id[cid] = c
    if skipped_malformed:
        print(f"  [WARN] {skipped_malformed} candidate row(s) had no candidate_id -- skipped, not counted as valid pool members.")

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("Format checks:")
    check("header == candidate_id,rank,score,reasoning",
          list(rows[0].keys()) == ["candidate_id", "rank", "score", "reasoning"] if rows else False)
    check("exactly 100 data rows", len(rows) == 100)

    ids = [r["candidate_id"] for r in rows]
    try:
        ranks = [int(r["rank"]) for r in rows]
    except (ValueError, KeyError) as e:
        print(f"  [FAIL] could not parse 'rank' column as integers: {e}")
        return 1
    try:
        scores = [float(r["score"]) for r in rows]
    except (ValueError, KeyError) as e:
        print(f"  [FAIL] could not parse 'score' column as floats: {e}")
        return 1

    check("ranks 1..100 each exactly once", sorted(ranks) == list(range(1, 101)))
    check("candidate_ids unique", len(set(ids)) == len(ids))
    check("every candidate_id exists in pool", all(i in cand_ids for i in ids))
    check("score non-increasing with rank",
          all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
    check("scores not all identical", len(set(scores)) > 1)
    check("no empty reasoning", all(r["reasoning"].strip() for r in rows))
    check("reasonings not all identical", len(set(r["reasoning"] for r in rows)) > 1)

    print("Honeypot check (Stage-3 DQ if >10% of top-100):")
    hp = sum(1 for i in ids if i in cand_by_id and honeypot_report(cand_by_id[i])["is_honeypot"])
    rate = hp / max(1, len(ids))
    check(f"honeypot rate = {rate:.1%} (<= 10%)", rate <= 0.10)

    print("\n" + ("ALL CHECKS PASSED -- safe to upload." if ok
                  else "VALIDATION FAILED -- fix the above before uploading."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
