# ReconIQ benchmark results

This document covers the comprehensive benchmarks across a range of models.

## Evidence tiers

### Primary comparison: full 60-month Fireworks run

The submission headline uses the same GPT-OSS-120B model, medium reasoning,
same 60 months, and same 10,000 bank lines for both approaches. The raw
artifacts are included in this repository and `python3 verify_submission.py` recomputes the
metrics and paired bootstrap intervals.

| Approach | Coverage | Exact group F1 | Bank-line accuracy | Estimated API cost |
|---|---:|---:|---:|---:|
| Whole month | 60 months / 10,000 lines | 49.31% | 35.76% | $0.65 |
| **ReconIQ harness** | **60 months / 10,000 lines** | **96.06%** | **92.95%** | **$1.81** |

This is the result to use for the main before/after claim.

The paired month bootstrap places the group-F1 improvement between +38.70 and
+54.06 percentage points at 95% confidence. The harness wins month-level line
accuracy on 56 months, ties on three, and loses on one; an exact two-sided sign
test gives p = 8.05e-16 with ties excluded. Both analyses preserve the month as
the evidence unit so correlated lines are not treated as independent.

### Supporting comparison: fixed 500-line set

The teammate report evaluates the first month of each size: `m50_00`,
`m150_00`, and `m300_00`. Together they contain 500 bank lines and 445 gold
groups. Cells below are micro-averaged across those three months.

| System | Approach | Exact group F1 | Bank-line accuracy | Execution |
|---|---|---:|---:|---|
| Deterministic amount/date solver | No model | 71.0% | 74.2% | Local |
| GPT-4.1 | Whole month | 72.7% | 65.6% | Hosted |
| GPT-OSS-120B, low reasoning | Whole month | 53.6% | 45.0% | Fireworks |
| Phi-4 14B | Whole month | 22.0% | 20.5% | Ollama; 300 lines did not fit |
| Llama 3.1 8B | Whole month | 4.6% | 4.8% | Ollama |
| GPT-4.1 | Harness | **97.9%** | **95.4%** | Hosted |
| GPT-OSS-120B, low reasoning | Harness | 96.4% | 92.2% | Fireworks |
| Phi-4 14B | Harness | 93.7% | 91.6% | Ollama |
| Qwen3-8B 4-bit, base | Harness | 90.1% | 88.4% | MLX |
| Qwen3-8B Q4_K_M | Harness | 82.2% | 80.6% | Ollama |
| Llama 3.1 8B | Harness | 61.5% | 59.6% | Ollama |

The fixed-sample comparison reinforces the architecture result: the harness
helps every completed model with both whole-month and harness results. It also
shows that the model still matters after candidate generation; the harness is
not merely a deterministic solver with an ornamental model call.

The GPT-OSS low-reasoning row is a useful ablation, not a replacement for the
primary result. On the same 500 lines, the harness raises F1 from 53.6% to
96.4% and line accuracy from 45.0% to 92.2%.

## Fixed-sample line-level behavior

The shortlist contained the gold answer on 97.4% of the 500 lines. That is the
maximum possible raw per-line accuracy without changing candidate generation.

| Proposer | Raw line accuracy | Ambiguous-line accuracy | Parse failures | Admitted by all checks | Wrong among admitted | Median / p90 latency | Cost per line |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4.1 | 96.4% | 95.5% | 0.0% | 88.8% | 0.0% | 1.04 s wall | $0.00099 |
| **GPT-OSS-120B, low reasoning** | **94.2%** | **94.7%** | **0.0%** | **83.0%** | **0.0%** | **0.56 / 0.78 s** | **$0.00012** |
| Phi-4 14B | 94.0% | 92.9% | 0.2% | 81.6% | 0.0% | 2.14 s wall | Local |
| Qwen3-8B 4-bit, base | 90.2% | 87.4% | 0.0% | 72.4% | 1.7% | 1.37 s wall | Local |
| Qwen3-8B Q4_K_M | 82.6% | 77.4% | 0.0% | 57.8% | 1.4% | 1.44 s wall | Local |
| Llama 3.1 8B | 60.8% | 54.2% | 0.0% | 45.4% | 14.1% | 1.20 s wall | Local |

“Admitted” means that the assembled proposal passed sum, date, and uniqueness
checks. The wrong-among-admitted column is the observed false-accept rate for
this sample, not a universal guarantee.

At $0.00012 per line, the reported low-reasoning GPT-OSS sample cost about
$0.06 for 500 line decisions. A full 10,000-line low-reasoning run is still
needed before claiming the same quality or cost at full coverage.

## Fixed-sample recall by match type

The 500-line set contains 115 one-to-one, 230 equal-amount decoy, 32 fee-netted,
43 one-bank-to-many-ledger, and 25 many-bank-to-one-ledger gold groups.

| System | Approach | One-to-one | Decoy | Fee-netted | One-to-many | Many-to-one |
|---|---|---:|---:|---:|---:|---:|
| GPT-OSS-120B, low reasoning | Whole month | 84.3% | 48.3% | 6.2% | 2.3% | 0.0% |
| **GPT-OSS-120B, low reasoning** | **Harness** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **36.0%** |
| GPT-4.1 | Harness | 100.0% | 100.0% | 100.0% | 100.0% | 60.0% |
| Phi-4 14B | Harness | 96.5% | 97.8% | 87.5% | 95.3% | 64.0% |
| Qwen3-8B 4-bit, base | Harness | 99.1% | 90.0% | 93.8% | 95.3% | 60.0% |
| Llama 3.1 8B | Harness | 96.5% | 44.8% | 56.2% | 58.1% | 44.0% |

The pattern agrees with the full primary run: the harness largely solves
one-to-one, decoy, fee-netted, and one-to-many matching, while many-to-one
partial payments remain the dominant weakness.

## Other full-run context

The teammate report also includes these runs over all 60 hard months:

| System | Approach | Exact group F1 | Bank-line accuracy |
|---|---|---:|---:|
| Deterministic amount/date solver | No model | 68.5% | 71.8% |
| GPT-4.1 | Whole month | 75.3% | 69.0% |
| Phi-4 14B | Harness | 93.3% | 90.6% |
| GPT-OSS-120B, medium reasoning | Whole month | 49.31% | 35.76% |
| **GPT-OSS-120B, medium reasoning** | **Harness** | **96.06%** | **92.95%** |

The Phi-4 full harness run processed 10,000 lines with 93.8% raw line
accuracy, 92.6% ambiguous-line accuracy, 97.1% shortlist coverage, 0.0% parse
failures, an 81.1% admission rate, and a 0.1% observed false-accept rate.

GPT-5.1 has a strong but non-equivalent result on only the twenty 300-line
months: 97.2% group F1 and 91.2% line accuracy. It should not be presented as a
full-dataset comparison.

## Reporting caveats

- The primary GPT-OSS medium-reasoning result is reproducible from the included
  raw artifacts. Some teammate-reported cross-model rows do not yet have their
  raw logs in this checkout; they are supporting evidence until those logs and
  the report-generation script are added.
- Nemotron 3.5 Lightning is excluded from completed-result tables. The reported
  harness run covered only 451 of 500 lines with mixed settings, and its
  whole-month output was invalid after reasoning exhausted the token budget.
- Phi-4 whole-month does not cover a 300-line month because roughly 21,000
  prompt tokens exceed its 16,000-token context window.
- The first Qwen3-8B LoRA attempt degraded quality. It is a negative result, not
  evidence that fine-tuning cannot work.
- The harness scores shown here use raw assembled proposals with no escalation.
  A production workflow should send failed checks to review.

