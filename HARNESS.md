# ReconIQ harness design

ReconIQ decomposes reconciliation into a constraint-satisfaction stage, solved deterministically, and a
disambiguation stage, delegated to a language model. Deterministic code enumerates every ledger subset that
is consistent with the bank amount and date. The model selects among those candidates using unstructured
text (counterparty, memo, invoice and description fields). Deterministic code then verifies the assembled
groups. The model never performs arithmetic and never sees more than one bank line and ten ledger rows.

This document specifies the implementation in `recon/shortlist.py`, `recon/solver.py` and `recon/slm.py`.
The Fireworks runner (`fireworks_gpt_oss_harness.py`) calls the same shortlist construction, prompt
rendering, group assembly and check functions.

## Pipeline

    bank line b, month ledger L
            │
            ▼
    build_shortlist(m, b)          date-window filter → exact-sum enumeration (k ≤ 4)
            │                      → partial-payment targets → rank → cap at 10 → alias C1..Cn
            ▼
    render_prompt(sl)              ≤10 candidate rows + exact-sum sets + partial targets
            │
            ▼
    model.complete(...)            one constrained decision: {"ledger": ["C3", "C7"]}
            │
            ▼
    parse_answer(sl, text)         alias → ledger id; any id outside the shortlist ⇒ invalid
            │
            ▼
    assemble_groups(proposals)     bank lines with an identical ledger set form one group
            │
            ▼
    run_checks(m, proposals)       sum ∧ date ∧ uniqueness, evaluated per group
            │
     ┌──────┴───────┐
     pass           fail
     auto-accept    unresolved → review queue

## 1. Candidate generation (`build_shortlist`)

All amounts are integer cents, so every equality below is exact.

**Date filter.** A ledger entry l is eligible for bank line b if
`DATE_WINDOW[0] ≤ day(b) − day(l) ≤ DATE_WINDOW[1]`, with `DATE_WINDOW = (−2, 7)` (`recon/gen.py`). A
ledger entry may be dated up to 2 days after the bank line or up to 7 days before it. Partial-payment
discovery uses `PARTIAL_WINDOW = (−2, 20)` (`recon/solver.py`).

**Exact-sum enumeration.** `subset_sums(b.amount, pool, max_k=4, cap=60)` returns every subset of size 1–4
of the date-filtered pool whose sum equals the bank amount.
- Sizes 1 and 2 are enumerated completely, through an amount-indexed hash map.
- Size 3 is a nested scan over the hash map.
- Size 4 is a meet-in-the-middle join over pair sums, with index ordering i<j<k<l so each set appears once.
- Sizes 3 and 4 stop once 60 sets have been found, so enumeration is complete for k ≤ 2 and bounded for
  k ≥ 3.

Fee-netted payouts need no special case: the gross-sale entry and the negative fee entry form a size-2 set.

**Partial-payment targets.** A ledger entry l becomes a target for b when three conditions hold:
- it has the same sign as b and |l| > |b|;
- it lies within `PARTIAL_WINDOW` of b;
- one or two other same-sign bank lines, also within the window, satisfy
  `b + Σ siblings = l` (from `subset_sums(l − b, others, max_k=2, cap=5)`).

The first sibling set found is recorded. This enumerates many-to-one groups without any single call seeing
the whole month.

**Ranking and cap.** Exact-sum sets are sorted by (cardinality, maximum date distance from b, lexicographic
ids). Candidate ids are collected in that order, and partial targets are appended after all exact-sum ids.
The list is then truncated to 10 (`MAX_CANDIDATES`). Two consequences:
- Sets that are not fully contained in the kept candidates are dropped.
- Partial-payment targets have the lowest priority, so on months dense with exact-sum sets they are the
  first candidates the cap removes.

**Aliasing and order.** Kept candidates are re-sorted by (date, id) and exposed only as `C1..Cn`. The model
therefore sees no real ledger ids, and the display order carries no signal from the ranking. `parse_answer`
maps aliases back to ledger ids and returns `None` for malformed output or any alias outside `C1..Cn`.

## 2. Model decision

**Input:**
- the bank line (date, amount, description);
- up to 10 candidate rows (date, amount, counterparty, account, memo);
- the exact-sum sets, already restricted to the kept candidates;
- the partial-payment targets.

Every option the model is offered has already been verified arithmetically. Its task is a single-label
selection over at most `|sets| + |partials| + 1` alternatives, the last being "none" (`{"ledger": []}`).

**Output contract:** `{"ledger": ["C…", …]}`. The Fireworks runner requests JSON-schema-constrained
output and makes up to two validation-repair calls on an invalid response.

**Logging:** each bank line writes one JSONL row containing the proposal, the raw output, the gold answer,
shortlist statistics and token usage. Runs are idempotent and resumable at line granularity.

**Run configuration.** The primary run used GPT-OSS-120B at `reasoning_effort=medium` with 8 concurrent
workers. Concurrency affects throughput only: shortlists, prompts and scoring are deterministic functions
of the month. The 500-line ablation in `BENCHMARKS.md` used `reasoning_effort=low` and is reported
separately.

## 3. Assembly and verification

**Assembly.** `assemble_groups` clusters bank lines by their exact proposed ledger set. A cluster
`({b₁…bₙ}, S)` becomes one predicted group, which is how a many-to-one group emerges from independent
per-line decisions.

**Checks.** `run_checks` evaluates each non-empty cluster:

| Check | Condition |
|---|---|
| Sum | Σ bank amounts in the cluster = Σ ledger amounts in S, exactly |
| Date | every (bᵢ, l ∈ S) pair lies in `PARTIAL_WINDOW` when n > 1, otherwise in `DATE_WINDOW` |
| Uniqueness | no l ∈ S is claimed by any other cluster |

A line is **admitted** when its cluster passes all three checks. False-accept rate (FAR) is the fraction
of admitted lines whose proposal differs from gold.

**Scoring modes.**
- The raw benchmark scores every assembled proposal, with no escalation.
- The operating tier auto-accepts only admitted groups. Rejected groups are labelled unresolved and
  routed to review, never silently treated as unmatched.

## Soundness of the checks

The checks are **necessary, not sufficient**. They guarantee arithmetic consistency, date feasibility and
one-to-one use of each ledger entry. They cannot prove that the textual choice is right: an equal-amount
decoy within the date window passes all three. FAR is therefore measured empirically, not guaranteed.

| Run | Admitted | Errors among admitted | Precision / recall |
|---|---|---|---|
| Primary run, 10,000 lines, medium reasoning | 8,121 groups | 4 false groups | 99.95% / 91.76% |
| 500-line ablation, low reasoning | 83.0% of lines | none observed | — |

## Limitations

- **Coverage ceiling.** Per-line accuracy is bounded by shortlist recall: the fraction of lines whose
  gold set lies inside the kept candidates. That is 97.1% over all 10,000 lines (Phi-4 run) and 97.4% on
  the 500-line set. It falls to 95.2% on 300-line months, where 62% of shortlists hit the 10-candidate cap.
- **Many-to-one groups.** A group of n partial payments is recovered only if all n sibling bank lines
  independently select the same target. Per-line accuracy p therefore compounds to roughly pⁿ for the
  group. Because partial targets are ranked last, the cap also removes them first. This is the dominant
  remaining error class.

## Code map

| Component | Location |
|---|---|
| Shortlist construction, prompt rendering, answer parsing, assembly, checks | `recon/shortlist.py` |
| Exact-sum enumeration (`subset_sums`), date windows (`in_window`, `PARTIAL_WINDOW`) | `recon/solver.py` |
| `DATE_WINDOW`, synthetic month generator | `recon/gen.py` |
| Per-line runner, local backends, line-level evaluation | `recon/slm.py` |
| Fireworks GPT-OSS runner | `fireworks_gpt_oss_harness.py` |
| Exact group and unmatched-item scoring | `recon/score.py` |
