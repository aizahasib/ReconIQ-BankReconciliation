# Phase 0 / Gate G0 report

**Status: G0 not passed.** Five of six runs pass (R0.2 on GPT-4.1). R0.6 failed its first attempt.

| Run | Go condition | Result | Verdict |
|---|---|---|---|
| R0.1 Generator | 60/60 gold self-check, ≥30% amount-unresolvable | 60/60; 70.4% unresolvable (hard regime) | **Go** |
| R0.2 Frontier strong form | F1@300 ≤ 90% and falling with size | GPT-4.1: 89.5% → 78.4% → 71.5% (50/150/300); Opus not run (no key) | **Go (GPT-4.1)** |
| R0.3 Solver only | F1 ≤ 80% | default 89.7% (fail), then hard 68.5% | **Go after branch** |
| R0.4 Shortlist + base SLM | raw acc ≥ 50% (better model) | phi4:14b 94.0%, qwen3:8b 82.6% | **Go** |
| R0.5 Check FAR | FAR ≤ 10%, admission ≥ 40% | phi4 0.0% / 81.6%; qwen3 1.4% / 57.8% | **Go** |
| R0.6 SFT smoke | +10 pp raw acc, FAR up ≤ 2 pp | −23.0 pp, FAR +4.1 pp | **Fail (attempt 1)** |

## Run details

### R0.1: generator (`recon/gen.py`)
Each month is minted ledger-first, and the bank lines are derived from the ledger, so gold is exact by construction. The generator covers six perturbations:
- 1:many
- many:1 partials
- fee netting
- 1–5 day lag
- equal-amount decoys
- 5% unmatched lines on each side

Results by regime:

| Regime | Self-check | Amount-unresolvable | Needs text (≥2 options in shortlist) |
|---|---|---|---|
| default | 60/60 | 51.3% | 71.7% |
| hard | 60/60 | 70.4% | 74.8% |

### R0.2: frontier strong form, GPT-4.1 (runner not included in this export)
Setup: hard regime, all 60 months, whole month in one prompt, one repair turn fed by the checker. GPT-4.1 has no reasoning mode.

| Size | Group F1 | Precision | Recall | $/month | s/month |
|---|---|---|---|---|---|
| 50 | 89.5% | 93.7% | 85.7% | 0.03 | 8 |
| 150 | 78.4% | 79.3% | 77.5% | 0.09 | 21 |
| 300 | **71.5%** | 71.9% | 71.1% | 0.17 | 40 |
| all | 75.3% | 76.2% | 74.5% | 0.10 | 23 |

Total cost was $5.71 at list price.
- **Repair turn:** it fired on all 60 months. Checker problems fell from 38.8 to 29.9 per month on average, so the repair turn helps but doesn't converge.
- **Recall by group type:** 1:1 91%, decoy 87%, fee-netted 37%, 1:many 18%, many:1 1%. GPT-4.1 matches 1:1 pairs well, including decoys, but rarely finds groups that need summing.
- **Same three months as R0.4/R0.5:**

| Size | GPT-4.1 | Solver | phi4 shortlist pipeline |
|---|---|---|---|
| 50 | 92.7% | 74.4% | 100% |
| 150 | 69.7% | 66.9% | 93.5% |
| 300 | 71.0% | 72.4% | 92.8% |
| all | 72.7% | 71.0% | 93.7% |

### R0.3: solver only (`recon/solver.py`)
The solver matches on amount and date only: an exact 1:1 greedy pass, then 1→many subset-sum for k ≤ 4, then many→1. Group F1 by month size:

| Regime | 50 | 150 | 300 | All |
|---|---|---|---|---|
| default | 93.8% | 91.2% | 88.2% | **89.7%** |
| hard | 69.7% | 70.6% | 67.3% | **68.5%** |

The default regime failed the ≤ 80% condition, so I applied the plan's branch. The hard regime (`HARD_PARAMS`) puts more weight on decoys, places decoys on the same day as the real entry, and allows 3-way decoy sets. It is used for every later run. The plan's branch also says to re-run R0.2 on the hard regime, and that happens once R0.2 runs at all.

### R0.4: shortlist + base SLM (`recon/shortlist.py`, `recon/slm.py`)
Setup: one month per size from the hard regime, 500 bank lines in total, at most 10 candidates per line. The shortlist contains the gold answer 97.4% of the time, so that is the ceiling for any model.

| Model | Raw acc | Ambiguous-line acc | Parse fail | s/line |
|---|---|---|---|---|
| phi4:14b (Ollama) | **94.0%** | 92.9% | 0.2% | 2.1 |
| qwen3:8b (Ollama, think off) | 82.6% | 77.4% | 0% | 1.4 |

### R0.5: check FAR
There are three checks:
- **Sum:** proposals with an identical ledger set form a cluster, and the cluster's bank total must equal its ledger total.
- **Date window.**
- **Uniqueness.**

| Model | Admission | FAR (all checks) | FAR sum / date / unique | Assembled F1, no escalation (all / 300) |
|---|---|---|---|---|
| phi4:14b | 81.6% | **0.0%** | 1.3 / 5.8 / 2.5% | 93.7% / 92.8% |
| qwen3:8b | 57.8% | 1.4% | 6.1 / 18.0 / 3.3% | 82.2% / 76.0% |

### R0.6: SFT smoke (`recon/sft.py`)
Setup:
- **Training:** MLX LoRA, rank 16, all layers, learning rate 1e-4, scale 20. One epoch of 2,000 examples, 500 iterations at batch size 4, which took 2.9 h.
- **Data:** the hard regime, with seeds disjoint from the eval set.
- **Evaluation:** the same 500 lines, with base and LoRA both run on MLX.

| | Raw acc | Ambiguous acc | Admission | FAR |
|---|---|---|---|---|
| Qwen3-8B-4bit base (MLX) | 90.2% | 87.4% | 72.4% | 1.7% |
| + LoRA 2k | 67.2% | 57.1% | 41.8% | 5.7% |

**Format check (the plan's first branch step).** Inference prompts are byte-for-byte prefixes of the training sequences. The empty `<think></think>` block is present in both, and round-trip parse failures are 0, so training and inference formats match.

One quirk: mlx_lm's prompt mask stops before the empty think block, so those four tokens are trained on. That is harmless, but it inflates the first validation loss (3.97).

**Why it got worse.**
- Training loss fell from 3.97 to 0.035 by iteration 250, with validation loss at 0.035. Almost all of that loss is JSON boilerplate; the decision itself is one candidate token.
- Of the LoRA's errors, 131 are lines the base model got right where LoRA picked a different candidate *of the same size*. In other words, it now picks the wrong one among equal-amount decoys.
- In the training targets, the gold answer is the first-listed exact set 49% of the time. The adapter looks to have learned that position prior instead of reading the text.
- These are aggressive settings (lr 1e-4 × scale 20 on all 36 layers) for a model that was already at 90%.

## Phase 0b (gate G0b)

### 0b.1: frontier-in-harness (runner not included in this export)
GPT-4.1 is the proposer on the same shortlists, the same checks and the same 500 lines as R0.4. The shortlist ceiling is 97.4%.

| Proposer | Raw acc | Ambiguous acc | Admission | FAR | Assembled F1 (300 / all) | $/line | s/line |
|---|---|---|---|---|---|---|---|
| GPT-4.1 | **96.4%** | 95.5% | 88.8% | 0.0% | 97.4% / 97.9% | 0.0010 | 1.0 |
| phi4:14b | 94.0% | 92.9% | 81.6% | 0.0% | 92.8% / 93.7% | local | 2.1 |
| qwen3:8b | 82.6% | 77.4% | 57.8% | 1.4% | 76.0% / 82.2% | local | 1.4 |

- **Near the ceiling:** GPT-4.1 gets only 5 lines wrong where the gold answer was on the shortlist. Its other 13 errors are lines the shortlist can't reach.
- **Line by line against phi4:** both right on 468; GPT-4.1 only on 14 (7 unmatched, 3 fee-netted, 2 decoy, 2 one-to-many); phi4 only on 2; both wrong on 16.
- **Cost:** in the harness, GPT-4.1 costs about $0.30 for a 300-line month (300 calls). Whole-month mode costs $0.17, so the harness spends more frontier tokens.

**G0b reading:** frontier-in-harness is 96.4%, above the plan's 96% line. That points to the **parity headline**: the harness lifts the frontier by about 26 pp at 300 lines (71.5% whole-month vs 97.4% assembled), and a 14B inside it comes within 2.4 pp raw / 4.6 pp assembled at no API cost. The 94–96% band between the two branches no longer applies.

### 0b.2: reasoning frontier whole-month (runner not included in this export)
GPT-5.1 (`gpt-5.1-2025-11-13`) at medium reasoning effort, run on the 20 hard-regime 300-line months, with up to 3 repair turns.

| | Group F1 | Precision | Recall | $/month | s/month |
|---|---|---|---|---|---|
| GPT-5.1 medium, 300 lines | **97.2%** (min 94.9, median 97.2, max 99.4) | 99.5% | 94.9% | 0.34 | 210 |
| GPT-4.1 whole-month, 300 lines | 71.5% | 71.9% | 71.1% | 0.17 | 40 |

- **Repairs:** the checker found nothing on the first turn in 15 of 20 months. The other 5 were fixed in one repair turn (1–8 problems each), so turns 3 and 4 never ran.
- **Recall by group type:** decoy 99.6%, fee-netted 99.1%, 1:1 98.2%, 1:many 89.1%, many:1 **33.6%**. Almost everything it proposes is right (99.5% precision). It mostly misses by leaving partial payments unmatched.
- **Cost:** $6.89 for the 20 months. About 26k reasoning tokens per month account for most of the bill.

**G0b reading:** reasoning whole-month is 97.2%, above the 90% branch line, so the plan says to harden the generator (vendor aliases, near-duplicate memos, same-vendor decoys) and re-run this step once. On the current generator, a reasoning frontier model given the whole month roughly equals GPT-4.1 inside the harness (97.4% assembled on m300_00) and beats phi4 inside it (92.8%).

### 0b.3: phi4 on all 60 hard months (`phase0.py r04 --model phi4:14b --all-months`)
phi4:14b in the harness on all 60 hard-regime months: 10,000 bank lines, 20 months at each size. The 500 lines from R0.4 are reused. Results are in `results/r04_all.json`.

| Size | Raw acc | Gold in shortlist | Assembled group F1 | Line acc (assembled) | Unmatched-bank recall |
|---|---|---|---|---|---|
| 50 | 99.3% | 100% | 99.2% | 99.3% | 100% |
| 150 | 97.7% | 99.9% | 97.4% | 96.5% | 85.0% |
| 300 | 90.9% | 95.2% | **90.4%** | 86.2% | 67.7% |
| all | 93.8% | 97.1% | 93.3% | 90.6% | 75.8% |

Across all lines, 81.1% are admitted, with a FAR of **0.11%** (9 wrong out of 8,110 admitted lines) and 0.04% parse failures, at 2.1 s per line. Raw accuracy and admission match the 3-month sample (94.0% / 81.6%), so the sample was representative.

- **Errors grow with month size.** At 300 lines, 88% of lines have two or more exact-sum options (47% at 50 lines), and the average shortlist holds 7.8 candidates (1.9 at 50).
- **The 544 errors at 300 lines break down as follows:**
  - 290 (53%): the gold answer isn't on the shortlist.
  - 152: the model picked the wrong set.
  - 99: the model matched a line that should stay unmatched.
  - 3: the model missed a match.

  **More than half the 300-line gap is the filter, not the model.**
- **Against GPT-5.1 whole-month on the same 20 300-line months:** phi4 in the harness scores 90.4% and GPT-5.1 scores 97.2%, a 6.8 pp gap. GPT-5.1's precision is higher (99.5% vs 88.8%). phi4's recall is also lower (92.0% vs 94.9%).
- **Comparison with step 1:** the step 1 parity reading rests on m300_00 alone. On all 20 300-line months, phi4 lands at 90.4%, not 92.8%.

**G0b reading:** on the current generator, phi4 in the harness does not match a reasoning frontier model on 300-line months. The specialized-model case depends on two things: step 7 raising shortlist recall at 300 lines (worth up to about 4.8 pp of raw accuracy), and the hardened generator from step 2 narrowing GPT-5.1's lead.

## Deviations from the plan

1. **R0.6 used Qwen3-8B, not the better R0.4 model.** phi4 scored 94.0% against a 97.4% ceiling, so +10 pp was arithmetically impossible for it. Qwen3 had about 15 pp of headroom.
2. **R0.6 compares base and LoRA on MLX, not Ollama**, so that both use the same engine and template. On the same 500 lines, Qwen3-8B-4bit scored 90.2% on MLX and 82.6% on Ollama q4_K_M. The 4-bit quantisation and runtime matter by about 8 pp.

## Headroom concern

The plan expected the base SLM to score 50–65% on this unit of work. It scored 83–94%. Once the filter has done exact-sum enumeration, choosing among the shortlisted options is easy. As a result:
- SFT has very little room to improve.
- The thesis now rests on R0.2. If a frontier model also reaches the high 90s at 300 lines, there is no gap to close.

The generator may need harder text disambiguation, such as near-duplicate memos, vendor aliases, or decoys that share a vendor. That would raise the plan's ≥40%-needs-text bar rather than just meeting it.

## What's needed to finish G0

1. **R0.6 second attempt.** The plan's branch is to re-run once with 5,000 examples:
   ```
   .venv/bin/python phase0.py r06 --n-train 5000
   ```
   At the current speed that takes about 7 h of training. By the plan's rule, a second R0.6 failure means G0 fails twice on the same run, so I'd stop and pick the categorization task. With the current hyperparameters, 5,000 examples will probably reproduce the same collapse. A run at lr 1e-5 with scale 1 is a fairer test, but it isn't the plan's literal branch.

## Files

- Code: `phase0.py` (subcommands r01–r06) and `recon/`, which holds `gen`, `solver`, `score`, `shortlist`, `slm` and `sft`.
- Data: `data/phase0_{default,hard}.jsonl` and `data/sft_hard_2000/`.
- The frontier runners (R0.2, 0b.1, 0b.2) are not included in this export. Their summary results are kept in `results/r02_*.json` and in the `hard_gpt41` entries of `results/r04.json` and `results/r05.json`.
- Results: `results/r0*.json`, with per-line outputs for the local models in `results/lines/`. The LoRA adapter isn't included (167 MB); `phase0.py r06` retrains it.
