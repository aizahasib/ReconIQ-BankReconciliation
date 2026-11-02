# ReconIQ harness design

ReconIQ splits reconciliation into work that deterministic code can prove
and work that a language model is better at judging. Code searches dates and
amounts; the model chooses among a small set using vendor, memo, invoice, and
description text; code then checks the assembled result.

This page describes the implementation in `recon/shortlist.py` and
`recon/slm.py`. The Fireworks runner in `fireworks_gpt_oss_harness.py` uses the
same shortlist and assembly code.

## Data flow

```text
One bank line + the month's ledger
                |
                v
     Deterministic candidate builder
   (date window, exact sums, partial targets)
                |
                v
       At most 10 aliased candidates
         + already-computed options
                |
                v
       One structured model decision
                |
                v
  Assemble identical proposals into groups
                |
                v
       Sum + date + uniqueness checks
                |
        +-------+-------+
        |               |
      pass             fail
   auto-accept     unresolved/review
```

## 1. Deterministic candidate generation

For each bank line, `build_shortlist` performs the following operations:

1. **Date filtering.** The normal code window is `DATE_WINDOW = (-2, 7)`,
   measured as `bank_date - ledger_date`. This permits a ledger date up to two
   days after the bank date or up to seven days before it. Partial-payment
   discovery uses the wider `PARTIAL_WINDOW = (-2, 20)`.
2. **Exact-sum search.** Amounts are stored as integer cents. The filter finds
   one- through four-entry ledger subsets whose total exactly equals the bank
   amount. This also represents a processor payout as gross sales plus a
   negative fee entry.
3. **Partial-payment targets.** For a larger same-sign ledger entry, the filter
   looks for one or two sibling bank lines that, together with the current
   line, exactly equal the ledger amount.
4. **Ranking and cap.** Exact-sum sets are ranked by fewer entries, tighter date
   spread, and stable ID order. Candidate IDs from those sets and partial
   targets are collected and capped at ten.
5. **Aliasing.** The prompt exposes candidates only as `C1` through `C10`.
   Real ledger IDs are mapped back only after a valid answer is received.

The internal exact-sum search keeps at most 60 sets before the final ten-entry
candidate cap. A candidate can be present even when it belongs to more than one
arithmetically valid option; this is the ambiguity the model must resolve.

## 2. Bounded model decision

The model sees:

- one bank line;
- up to ten ledger rows with date, amount, vendor/customer, account, and memo;
- the candidate sets that already sum exactly to the bank amount; and
- any ledger entries identified as possible partial-payment targets.

It returns one structured answer:

```json
{"ledger": ["C3", "C7"]}
```

An empty list means no match. Any malformed response or candidate outside the
shortlist is invalid. The Fireworks runner requests JSON-schema output and can
make up to two validation-repair calls. It saves one JSONL row per bank line,
so a stopped run resumes without repeating completed decisions.

The primary run used GPT-OSS-120B with medium reasoning and eight
concurrent workers. Concurrency changes throughput, not the candidate sets or
scoring. A separate 500-line ablation used low reasoning; its results are
reported in `BENCHMARKS.md` and must not be mixed with the primary run.

## 3. Group assembly and checks

`assemble_groups` merges bank lines that selected the same ledger set. This is
how several partial payments can form one many-bank-to-one group even though no
model call sees the whole month.

`run_checks` evaluates each non-empty assembled cluster:

- **Sum:** the total of all bank lines in the cluster equals the total of its
  ledger entries exactly, to the cent.
- **Date:** every bank/ledger relationship falls within the applicable normal
  or partial-payment window.
- **Uniqueness:** no ledger entry is claimed by two different proposed groups.

The raw benchmark scores all assembled proposals with no escalation. The
safety-oriented operating tier auto-accepts only groups whose constituent
proposals pass all three checks; failures should be labeled **unresolved** and
sent to review, not silently declared unmatched.

## What the checks do and do not prove

The checks establish arithmetic and structural consistency. They do not prove
that the text-based choice is the only valid business interpretation. An
equal-amount decoy can still satisfy sum and date checks. That is why false
accept rate is measured empirically and why production deployment still needs
human review, access controls, and testing on real data.

On the primary 10,000-line run, the checked tier accepted 8,121 groups with
only four false groups: **99.95% group precision and 91.76% recall**. On the
separate 500-line low-reasoning run, 83.0% of line proposals were admitted with
no wrong admitted proposals in that sample.

## Known limitation

Shortlist coverage is the upper bound on per-line model accuracy. It was 97.1%
on the full Phi-4 run and 97.4% on the fixed 500-line comparison, but it declines
at the largest month size. Many-bank-to-one partial-payment groups are the
primary remaining weakness because every sibling bank line must independently
select the same ledger target before assembly succeeds.

## Code map

| Component | Location |
|---|---|
| Candidate generation and prompt rendering | `recon/shortlist.py` |
| Exact-sum search and date-window helpers | `recon/solver.py` |
| Local backends and shared line evaluation | `recon/slm.py` |
| Fireworks GPT-OSS runner | `fireworks_gpt_oss_harness.py` |
| Exact group and unmatched-item scoring | `recon/score.py` |

