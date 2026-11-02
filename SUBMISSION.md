# ReconProof — submission kit

This file is formatted for copying into the HackerSquad project form. Replace
the bracketed placeholders and confirm the disclosure section before
submitting.

## Project name

ReconProof

## One-line description

An auditable bank-reconciliation harness that uses deterministic accounting
constraints to turn Fireworks GPT-OSS from a 49% F1 whole-month matcher into a
96% F1 verified reconciliation system.

## Project overview

**Task:** Match bank statement lines to general-ledger entries, including
one-to-many invoice settlement, many-to-one partial payments, processor-fee
netting, date lag, equal-amount decoys, and unmatched items.

**Intended user:** Finance and accounting teams that need to reconcile large
transaction batches without blindly trusting plausible LLM output.

**Approach:** ReconProof generates an arithmetic shortlist for each bank line,
asks GPT-OSS-120B on Fireworks Serverless to select the correct candidates, and
checks every proposal for amount, date, and uniqueness consistency. Checked
matches can be accepted automatically; failures remain unresolved for review.
The model sees at most ten aliased candidates and never has to search the full
ledger or perform the arithmetic itself.

**Why it matters:** Whole-month LLM reconciliation becomes incomplete as the
month grows. At 300 lines, GPT-OSS whole-month group F1 falls to 21.17% and
marks almost 17 times too many bank lines unmatched. ReconProof reaches 94.21%
F1 at the same size and provides deterministic evidence for each accepted
match.

## Who benefits and why it is valuable

The intended user is a finance or accounting team reconciling transaction
batches. Their expensive work is not just finding obvious one-to-one matches;
it is investigating exceptions, partial payments, processor fees, duplicate
amounts, and plausible but incorrect proposals.

ReconProof's checked tier admits 84.33% of line proposals on the benchmark. An
illustrative 10,000-line workload would therefore reduce the review queue from
10,000 lines to 1,567. Assuming 30 seconds of analyst review per line, that is
83.3 hours versus 13.1 hours, or 70.3 hours shifted from routine inspection to
exceptions. At an illustrative fully loaded rate of $50/hour, that is roughly
$3,500 of analyst capacity per batch versus $1.81 of measured API cost.

Those labor assumptions are a value hypothesis, not customer-validated ROI.
Before production, the team would measure actual handling time, validate on
real de-identified data, and sample accepted matches in addition to reviewing
every rejected proposal.

## What we tried and learned

| Experiment | Hypothesis | Result | Decision |
|---|---|---|---|
| Whole-month GPT-OSS | A capable open model can return the complete reconciliation directly | 49.31% F1 overall; collapses to 21.17% at 300 lines | Rejected as the operating design |
| Deterministic amount/date solver | Arithmetic alone can solve the task | 68.5% F1; strong arithmetic, weak on decoys | Use as candidate generation, not final selection |
| Shortlist + GPT-OSS | Let code search and calculate; let the model resolve text ambiguity | 96.06% F1 and 92.95% line accuracy | Selected architecture |
| Checked admission tier | Arithmetic and uniqueness checks can make automation safer | 99.95% precision at 91.76% recall | Auto-accept passes; review failures |
| Local small models | The architecture may transfer across proposers | Phi-4 93.7%, Qwen3-8B 90.1%, Llama 3.1 61.5% F1 on the fixed sample | Architecture transfers, but model quality matters |
| First Qwen3-8B LoRA | Task examples may improve a small base model | Fell to 54.7% F1 | Negative result; do not use this adapter |

## Before-and-after evidence

- **Baseline model:** OpenAI GPT-OSS-120B through Fireworks Serverless, whole
  month in one request.
- **Improved system:** The same model and reasoning effort inside the ReconProof
  shortlist and verification harness.
- **Evaluation dataset:** The same 60 fixed synthetic months, containing
  10,000 bank lines at 50-, 150-, and 300-line sizes.
- **Primary metric:** Exact group F1. Every bank and ledger ID in a group must
  match gold.
- **Result:** F1 improves from 49.31% to 96.06%; bank-line accuracy improves
  from 35.76% to 92.95%.
- **Uncertainty:** Paired month bootstrap, 20,000 samples: the 95% interval for
  the F1 improvement is +38.70 to +54.06 percentage points. The harness wins
  on line accuracy in 56 of 60 months, ties on three, and loses on one. An exact
  paired sign test on month-level wins and losses gives p = 8.05e-16
  (two-sided, ties excluded).
- **Safety tier:** Requiring sum, date, and uniqueness checks yields 99.95%
  group precision at 91.76% recall.
- **Tradeoff:** Estimated API cost increases from $0.65 to $1.81 for all 10,000
  lines. The harness uses more requests but makes each model decision bounded,
  parallelizable, resumable, and auditable.
- **Low-reasoning ablation:** On a separate fixed 500-line set, GPT-OSS with low
  reasoning improves from 53.6% whole-month F1 to 96.4% harness F1 and costs a
  reported $0.00012 per line. This is supporting evidence; the full 60-month
  low-reasoning run has not been completed.
- **Cross-model evidence:** On that same 500-line harness sample, GPT-4.1 reaches
  97.9% F1, Phi-4 14B reaches 93.7%, Qwen3-8B 4-bit reaches 90.1%, and Llama
  3.1 8B reaches 61.5%. The architecture helps across model sizes, while the
  proposer still materially affects quality.
- **Limitation:** Many-bank-to-one matching remains the main weakness (50.2%
  recall overall; 19.9% on 300-line months). The dataset is synthetic and real
  financial data requires additional privacy and operational validation.

## Evaluation integrity

- Both primary systems use the same GPT-OSS-120B model, medium reasoning,
  dataset, month IDs, and exact scoring code.
- The 60 months are generated ledger-first, then bank lines are derived, so the
  gold groups are exact by construction.
- The model was not trained on these months. However, shortlist parameters were
  developed during Phase 0 while studying this synthetic regime. This is a
  controlled benchmark, not a fully untouched external test set.
- Statistical resampling uses whole months because lines within a month are
  correlated. Resampling individual lines would overstate certainty.
- Raw primary outputs are saved per month or per line and can be independently
  rescored with `python3 verify_submission.py`.

## Disclosure — confirm before submission

**Pre-existing work:** `[State what, if anything, existed before the hackathon.
If none, write: “The submitted system and evaluation were built during the
event.”]`

**AI coding assistance:** OpenAI Codex was used to help implement and review
scripts, analyze saved results, and prepare repository documentation. `[Add any
other assistants used by you or your teammates.]`

**Team contributions:** `[List each teammate and their concrete contribution,
including who ran the additional model experiments.]`

## Technologies

- Fireworks Serverless API
- OpenAI GPT-OSS-120B (`accounts/fireworks/models/gpt-oss-120b`)
- Python 3.11 standard library
- Structured JSON generation
- Concurrent and resumable inference
- Deterministic subset-sum candidate generation
- Sum, date-window, and uniqueness verification
- Paired month-level bootstrap evaluation
- Exact paired month-level sign test

## Links

- **Public repository:** `[REPOSITORY_URL]`
- **Demo video:** `[VIDEO_URL]`
- **Full report:** `FIREWORKS_GPT_OSS_HARNESS_COMPARISON.md` in the repository
- **Extended benchmarks:** `BENCHMARKS.md` in the repository
- **Harness design:** `HARNESS.md` in the repository

## 75-second demo outline

**0–10 seconds — Problem**

“Bank reconciliation looks like text matching, but the output must also satisfy
accounting arithmetic. Whole-month models often return plausible but incomplete
answers as transaction volume grows.”

**10–25 seconds — Baseline failure**

Show the results table. “On the same 60 months, GPT-OSS whole-month drops from
96% F1 at 50 lines to 21% at 300 lines, primarily by leaving valid matches
unmatched.”

**25–45 seconds — Approach**

Show `recon/shortlist.py` or one rendered prompt. “ReconProof enumerates small
candidate sets using date and exact-sum constraints. GPT-OSS chooses among at
most ten candidates, then deterministic sum, date, and uniqueness checks decide
whether the proposal can be automatically accepted.”

**45–62 seconds — Measured result**

Run `python3 verify_submission.py`. “The same model on the same 10,000 lines
improves from 49.3% to 96.1% group F1. Checked proposals achieve 99.95%
precision.”

**62–75 seconds — Value and honesty**

“The checked tier automatically admits 84% of proposals and routes the rest to
review. The experiment cost $1.81. The remaining weakness is partial-payment
grouping at large month sizes, and the result still needs validation on real
bank data.”

## Rubric map

| Criterion | Evidence | Honest weakness |
|---|---|---|
| Benchmark evaluation, 40% | Same open model before and after on 60 paired months and 10,000 lines; +46.74 F1 points; month bootstrap and sign test; negative results reported | Synthetic regime influenced shortlist development; no frozen external dataset |
| Business value, 25% | Named finance/accounting user; checked automation tier; explicit review-time and cost assumptions | No customer interview or measured handling-time study |
| Technical execution, 20% | Resumable Fireworks runners, structured output, deterministic candidate search, exact group assembly, three admission checks, reproducible scorer | Many-to-one candidate recall declines at 300 lines |
| Presentation and demo, 10% | Before/after table, one-command verification, 75-second script | Video and public links still need to be added |
| Documentation, 5% | README, harness design, benchmark report, raw artifacts, tests, security notes | Some teammate cross-model raw logs are not yet in this checkout |

## Pre-submission checklist

- [ ] Create the public repository and replace `[REPOSITORY_URL]` above.
- [ ] Confirm `python3 verify_submission.py` passes from a clean checkout.
- [ ] Confirm README links work on GitHub.
- [ ] If the supporting teammate runs are highlighted in the demo, add their
      missing raw logs and report-generation script to the repository.
- [ ] Replace every bracketed disclosure placeholder and confirm it with all
      teammates.
- [ ] Replace the illustrative labor assumptions if the team has better
      customer or domain evidence.
- [ ] Record and upload a demo under 120 seconds.
- [ ] Replace `[VIDEO_URL]` above and test the video link without being signed in.
- [ ] Add project description, repository link, technologies, and required team fields.
- [ ] Save the project form, then explicitly submit it for judging.
- [ ] Confirm the project displays as submitted before the deadline.

The event guide states that saving a draft does not submit it and that judges
see the latest saved version after submission.
