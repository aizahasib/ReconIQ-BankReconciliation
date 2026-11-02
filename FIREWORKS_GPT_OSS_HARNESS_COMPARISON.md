# Fireworks GPT-OSS-120B: Whole-Month vs Shortlist Harness

**Run date:** November 2, 2026  
**Dataset:** `data/phase0_hard.jsonl`  
**Coverage:** 60 synthetic reconciliation months and 10,000 bank lines  
**Model:** `accounts/fireworks/models/gpt-oss-120b`  
**Reasoning effort:** medium

## Executive conclusion

The shortlist harness is the decisive winner. It raises exact group F1 from
**49.31% to 96.06%** and bank-line accuracy from **35.76% to 92.95%** on the
same 60 months. More importantly, it removes the whole-month system's severe
size collapse: at 300 bank lines, group F1 rises from **21.17% to 94.21%**.

The whole-month run is strong on 50-line months but becomes an incomplete,
high-precision matcher at larger sizes. It omits most valid groups and labels
far too many items unmatched. The harness converts the task into bounded
per-line choices, where GPT-OSS performs well and consistently.

The recommended operating design is:

1. Generate bounded candidate sets with the deterministic shortlist.
2. Let GPT-OSS select among those candidates.
3. Auto-accept only proposals that pass the sum, date, and uniqueness checks.
4. Treat everything else as unresolved and escalate it; do not automatically
   label rejected proposals unmatched.
5. Improve the many-bank-to-one candidate path, which is now the dominant
   remaining quality bottleneck.

## Methodology

Both systems used exactly the same hard-regime dataset:

- 20 months with 50 bank lines each;
- 20 months with 150 bank lines each;
- 20 months with 300 bank lines each.

The whole-month system sent every bank line and ledger entry for one month in
one request and asked the model to return all match groups. The harness made
one model decision per bank line, exposing at most ten ledger candidates plus
the arithmetic candidate sets found by the deterministic filter.

Scoring uses exact group equality: every bank and ledger ID in a group must
match gold. Bank-line accuracy also requires the complete containing group to
be correct.

## Primary results

### Exact reconciliation quality

| Month size | Whole-month group F1 | Harness group F1 | Change | Whole-month line accuracy | Harness line accuracy | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 96.45% | 99.60% | +3.16 pp | 89.50% | 99.30% | +9.80 pp |
| 150 | 70.64% | 98.60% | +27.96 pp | 57.63% | 97.57% | +39.93 pp |
| 300 | 21.17% | 94.21% | +73.04 pp | 15.87% | 89.58% | +73.72 pp |
| **All** | **49.31%** | **96.06%** | **+46.74 pp** | **35.76%** | **92.95%** | **+57.19 pp** |

Overall group counts make the difference concrete:

| System | Correct groups | Predicted groups | Gold groups | Group precision | Group recall |
|---|---:|---:|---:|---:|---:|
| Whole month | 3,065 | 3,585 | 8,846 | 85.50% | 34.65% |
| Harness, raw assembled | 8,512 | 8,877 | 8,846 | 95.89% | 96.22% |
| Harness, checks admitted only | 8,117 | 8,121 | 8,846 | **99.95%** | 91.76% |

The raw assembled harness result is the best balanced result. Applying all
deterministic checks produces a safety-oriented result with only four false
groups among 8,121 accepted groups, at the cost of leaving more correct groups
unresolved.

### Uncertainty and paired consistency

A paired bootstrap resampling entire months (20,000 samples, fixed seed) gives
a 95% interval of **+38.70 to +54.06 percentage points** for the harness's
group-F1 improvement and **+50.73 to +62.43 points** for its bank-line-accuracy
improvement. The harness improves line accuracy on 56 of 60 months, ties on
three, and loses on one. An exact paired sign test on those month-level outcomes
gives **p = 8.05e-16** (two-sided, ties excluded). The bootstrap estimates the
size of the gain; the sign test asks whether month-level wins and losses are
consistent with equal performance. Run `python3 verify_submission.py` to
recompute these figures from the included raw artifacts.

## Why the whole-month run fails

The whole-month model becomes increasingly incomplete as month size grows:

| Month size | Predicted groups as a share of gold | Predicted unmatched bank lines | Truly unmatched bank lines |
|---:|---:|---:|---:|
| 50 | 94.23% | 140 | 40 |
| 150 | 69.03% | 1,179 | 160 |
| 300 | 17.49% | 5,069 | 300 |

At 300 lines, the model predicts nearly 17 times too many bank lines as
unmatched. Its group precision remains 71.11%, so the primary failure is not
random matching; it is failure to return an exhaustive answer.

The prompts fit inside GPT-OSS-120B's context window. The observed behavior is
therefore a task-scaling and exhaustive-generation failure rather than a simple
context overflow. The original runner did not retain `finish_reason`, so this
run cannot distinguish deliberate early completion from an output-budget
effect with certainty.

## Harness error decomposition

The per-line harness results are:

- **10,000** total decisions;
- **9,554** correct decisions (95.54% raw accuracy);
- **9,707** decisions where the gold answer was present in the shortlist;
- **293** unavoidable errors caused by the gold answer missing from the
  shortlist;
- **153** additional model errors when the filter could have supported the
  correct answer;
- **98.42%** model accuracy conditional on the gold answer being available;
- **94.48%** accuracy on lines with two or more arithmetic options;
- **0%** parse failures.

This means roughly two-thirds of the remaining raw errors are filter/shortlist
errors, not model-selection errors. Improving candidate generation should now
have a larger payoff than changing the model prompt.

### Deterministic checks

The harness checks admitted 84.33% of all bank-line proposals. Only four of
8,433 admitted line proposals were wrong, for a **0.047% false-accept rate**.

When assembled strictly from admitted proposals, exact group precision is
99.95%, recall is 91.76%, and F1 is 95.68%. This is suitable as a high-precision
automatic tier, provided rejected items are routed to review or escalation
rather than declared unmatched.

## Match-type analysis

Exact gold-group recall by relationship type:

| Match type | Whole month | Harness |
|---|---:|---:|
| Equal-amount decoy | 39.12% | **99.71%** |
| Processor fee netting | 20.88% | **100.00%** |
| One bank to many ledger | 16.24% | **99.55%** |
| One-to-one | 41.17% | **97.14%** |
| Many bank to one ledger | 4.47% | **50.20%** |

The harness solves nearly all decoy, fee-netting, one-to-many, and one-to-one
groups. Many-bank-to-one remains weak:

| Month size | Harness many-bank-to-one recall |
|---:|---:|
| 50 | 96.55% |
| 150 | 86.62% |
| 300 | 19.86% |

This sharp 300-line decline is the most important next engineering target. The
candidate builder needs better partial-payment grouping at larger month sizes.

## Unmatched-item behavior

| Metric, all sizes | Whole month | Harness raw |
|---|---:|---:|
| Unmatched bank precision | 7.58% | **65.59%** |
| Unmatched bank recall | **96.80%** | 93.80% |
| Unmatched ledger precision | 7.53% | **84.54%** |
| Unmatched ledger recall | **97.54%** | 96.30% |

The whole-month system finds almost every truly unmatched item only because it
incorrectly labels thousands of matched items unmatched. The harness is much
better calibrated. Even so, a missing or rejected harness match should be
treated as unresolved, not automatically unmatched, especially for 300-line
months.

## Token use, cost, and latency

| Resource | Whole month | Harness | Ratio |
|---|---:|---:|---:|
| Logical model tasks | 60 months | 10,000 bank lines | 166.7x |
| Total tokens | 2,190,965 | 7,931,207 | 3.62x |
| Prompt tokens | 1,345,759 | 5,716,332 | 4.25x |
| Cached prompt tokens | 448,195 | 2,760,053 | 6.16x |
| Completion tokens | 845,206 | 2,214,875 | 2.62x |
| Reasoning tokens | 751,319 | 1,993,888 | 2.65x |
| Estimated cost | **$0.65** | **$1.81** | 2.80x |

The whole-month runner made 84 billed successful calls for 60 logical month
tasks because 16 months needed output-validation retries. Every harness line
passed response parsing on its first successful call; transient HTTP retries
that produced no response are not represented in the saved usage records.

Cost is estimated using Fireworks' listed rates on November 2, 2026 for
GPT-OSS-120B: $0.15 per million uncached input tokens, $0.015 per million cached
input tokens, and $0.60 per million output tokens. The model page is
[here](https://fireworks.ai/models/fireworks/gpt-oss-120b).

The whole-month run accumulated 6,287 seconds of request time, averaging 104.8
seconds per month. The harness accumulated 12,298 request-seconds, averaging
1.23 seconds per bank line (median 0.99 seconds, 95th percentile 2.65 seconds).
Because the harness used eight concurrent workers, summed request time is not
wall-clock runtime; the runner did not persist start and end timestamps for an
exact wall-clock comparison.

The harness used almost two million reasoning tokens—90% of its completion
tokens—for a constrained selection task. A low-reasoning ablation is likely the
best next cost experiment.

## Context against existing Phase 0 baselines

| System | 50-line F1 | 150-line F1 | 300-line F1 | Overall F1 |
|---|---:|---:|---:|---:|
| GPT-OSS-120B whole month | 96.45% | 70.64% | 21.17% | 49.31% |
| GPT-4.1 whole month | 89.55% | 78.40% | 71.50% | 75.30% |
| Deterministic solver | 69.68% | 70.60% | 67.30% | 68.52% |
| Phi-4 14B shortlist harness | 99.21% | 97.43% | 90.35% | 93.32% |
| **GPT-OSS-120B shortlist harness** | **99.60%** | **98.60%** | **94.21%** | **96.06%** |

GPT-OSS with the harness is the strongest full-dataset result currently stored
in the repository. GPT-5.1's recorded 300-line-only whole-month result remains
higher on that slice (97.17% F1 versus 94.21%), but it was not run across all
three sizes in the stored results.

## Supporting low-reasoning ablation

A teammate-run fixed sample covers `m50_00`, `m150_00`, and `m300_00`: 500 bank
lines and 445 gold groups. This is a separate experiment and should not be
pooled with the medium-reasoning full run above.

On those 500 lines, low-reasoning GPT-OSS achieves **96.4% harness group F1**
and **92.2% line accuracy**, versus **53.6% F1** and **45.0% line accuracy**
when the same low-reasoning model processes whole months. Per-line raw accuracy
is 94.2%; 83.0% of proposals pass every check, with no wrong admitted proposals
observed in this sample. Reported cost is $0.00012 per line.

This result makes a full low-reasoning run the leading cost-reduction
experiment, but the sample is too small to replace the primary 60-month claim.
It also does not yet have all raw Fireworks logs in this checkout. The full
cross-model table and caveats are in `BENCHMARKS.md`.

## Final recommendation

Adopt the shortlist harness, not whole-month GPT-OSS, as the working design.
Use deterministic admission checks for the automatic tier and escalate the
remainder. The quality gain is very large for a modest absolute cost increase
of about **$1.17 across 10,000 lines** in this experiment.

Prioritize the next work in this order:

1. Improve many-bank-to-one/partial-payment candidate generation at 300-line
   scale.
2. Extend the promising 500-line low-reasoning ablation to all 60 months to
   measure full-coverage quality, cost, and uncertainty.
3. Persist API status, `finish_reason`, actual run timestamps, and transient
   retry counts in future runners.
4. Add an explicit escalation evaluation so rejected proposals are measured as
   unresolved work rather than unmatched predictions.

## Source artifacts

- Whole-month raw results:
  `results/lines/fireworks_gpt_oss_120b_whole_month.jsonl`
- Harness per-line results:
  `results/lines/fireworks_gpt_oss_120b_harness.jsonl`
- Harness aggregate and equivalent-month comparison:
  `results/fireworks_gpt_oss_120b_harness.json`
