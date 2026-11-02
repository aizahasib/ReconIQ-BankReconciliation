# ReconProof

**Verified bank reconciliation with Fireworks GPT-OSS**

ReconProof combines deterministic accounting constraints with an LLM. Instead
of asking a model to reconcile an entire month in one response, it generates a
small arithmetic shortlist for each bank line, asks GPT-OSS to choose among the
candidates, and verifies the proposal with sum, date, and uniqueness checks.

The result is a large, measured improvement on the same 60 fixed synthetic
months:

| Approach | Exact group F1 | Bank-line accuracy | Estimated API cost |
|---|---:|---:|---:|
| GPT-OSS-120B, whole month | 49.31% | 35.76% | $0.65 |
| **GPT-OSS-120B + ReconProof harness** | **96.06%** | **92.95%** | $1.81 |

At 300 bank lines—the hardest setting—the harness raises group F1 from 21.17%
to 94.21%. If only proposals passing all three checks are automatically
accepted, group precision reaches **99.95%** with 91.76% recall.

## Why this matters

Bank reconciliation is not just semantic matching. A deposit can settle
several invoices, several partial payments can settle one ledger entry, and
processor fees can net against gross sales. A plausible but arithmetically
wrong match creates downstream accounting risk.

ReconProof separates the work:

```text
Bank line + ledger
        │
        ▼
Deterministic candidate generation
(date windows, exact sums, partial-payment targets)
        │
        ▼
Fireworks GPT-OSS selection
        │
        ▼
Sum + date + uniqueness checks
        │
        ├── pass ──► auto-accept
        └── fail ──► unresolved / review
```

This makes the model's task bounded and auditable while retaining text-based
judgment for vendor names, memos, descriptions, invoices, and decoys.

The implementation is documented step by step in [HARNESS.md](HARNESS.md),
including the exact date windows, subset limits, group assembly, and admission
semantics.

### Illustrative business value

The practical value is reducing the queue that a finance analyst must inspect,
not eliminating financial oversight. On the synthetic benchmark, the checks
admit 84.33% of line proposals at a 0.047% observed false-accept rate.

For an illustrative 10,000-line monthly workload:

| Assumption | Manual workflow | ReconProof review tier |
|---|---:|---:|
| Lines requiring review | 10,000 | 1,567 |
| Review time at 30 seconds per line | 83.3 hours | 13.1 hours |
| Potential analyst time shifted to exceptions | — | 70.3 hours |

The 30-second review time is an explicit planning assumption, not a measured
customer result. At a hypothetical fully loaded analyst cost of $50/hour, the
time difference would be about $3,500 per 10,000-line batch, versus $1.81 in
measured API cost. Real deployment requires customer timing studies and must
include review of unresolved and sampled accepted matches.

## Reproduce the stored evaluation

The repository includes the dataset and raw result logs. Recompute the primary
metrics and a paired, month-level bootstrap confidence interval without an API
key or third-party package:

```bash
python3 verify_submission.py
```

Expected headline output:

```text
Whole month:  group F1 0.4931, line accuracy 0.3576
Harness:      group F1 0.9606, line accuracy 0.9295
Delta:        group F1 +0.4674, line accuracy +0.5719
```

The harness improves line accuracy on 56 of 60 months, ties on three, and
loses on one. The 95% paired bootstrap interval for the group-F1 improvement is
**+38.70 to +54.06 percentage points**. An exact paired sign test at the month
level gives **p = 8.05e-16** for the 56-to-1 win/loss split, excluding three
ties. Both tests keep each reconciliation month as the unit of evidence rather
than treating correlated lines as independent.

## Additional teammate runs

A separate fixed sample of the first 50-, 150-, and 300-line months provides a
500-line cross-model comparison. These are supporting experiments, not the
primary headline above.

| Model | Approach | Reasoning | Exact group F1 | Bank-line accuracy |
|---|---|---|---:|---:|
| GPT-OSS-120B | Whole month | Low | 53.6% | 45.0% |
| **GPT-OSS-120B** | **ReconProof harness** | **Low** | **96.4%** | **92.2%** |
| GPT-4.1 | ReconProof harness | — | 97.9% | 95.4% |
| Phi-4 14B | ReconProof harness | — | 93.7% | 91.6% |
| Qwen3-8B 4-bit | ReconProof harness | — | 90.1% | 88.4% |
| Llama 3.1 8B | ReconProof harness | — | 61.5% | 59.6% |

The low-reasoning GPT-OSS harness achieved 94.2% raw per-line selection
accuracy, admitted 83.0% of proposals with no observed false accepts in this
sample, and cost $0.00012 per line. A full low-reasoning run is still needed
before extrapolating that cost-quality result to all 10,000 lines. See
[BENCHMARKS.md](BENCHMARKS.md) for sample definitions, line-level results,
match-type recall, other full runs, and caveats.

## Run the Fireworks experiments

Requirements:

- Python 3.11+
- A Fireworks API key with serverless inference access
- No Python packages for the primary Fireworks experiments

Set the key in your shell; it is never saved by the runners:

```bash
export FIREWORKS_API_KEY="your-key"
```

### Whole-month baseline

Start with one month, then resume across all 60:

```bash
python3 fireworks_gpt_oss.py --limit 1
python3 fireworks_gpt_oss.py
```

### ReconProof harness

Start with 25 lines, then resume across all 10,000:

```bash
python3 fireworks_gpt_oss_harness.py --limit-lines 25
python3 fireworks_gpt_oss_harness.py
```

The harness defaults to eight concurrent requests. If Fireworks returns
frequent transient rate-limit errors, lower concurrency; completed lines remain
saved:

```bash
python3 fireworks_gpt_oss_harness.py --workers 2
```

Both runners use `accounts/fireworks/models/gpt-oss-120b`, medium reasoning,
structured JSON output, exponential retry for transient API failures, output
validation, and resumable JSONL logs.

## Dataset and evaluation

`data/phase0_hard.jsonl` contains 60 synthetic months generated ledger-first,
with bank lines derived afterward so the gold groups are exact by construction.
It contains 10,000 bank lines across three sizes:

| Month size | Months | Bank lines |
|---:|---:|---:|
| 50 | 20 | 1,000 |
| 150 | 20 | 3,000 |
| 300 | 20 | 6,000 |

The generator covers:

- one-to-one matches;
- one bank line to several ledger entries;
- several partial bank payments to one ledger entry;
- processor gross-sale and fee netting;
- equal-amount textual decoys;
- date lag and unmatched items.

The primary metric is exact group F1. A prediction is correct only when every
bank and ledger ID in the group matches gold. Bank-line accuracy likewise
requires the line's complete group to be correct.

The GPT-OSS model was not trained or tuned on this evaluation set. The
deterministic shortlist parameters were developed during Phase 0; results
should therefore be treated as a controlled synthetic benchmark, not a claim
of production accuracy on real bank data. Because the harness parameters were
developed while studying this synthetic regime, a frozen external dataset is
the appropriate next validation step.

## Results and evidence

- [Full whole-month vs harness report](FIREWORKS_GPT_OSS_HARNESS_COMPARISON.md)
- [Extended cross-model benchmarks](BENCHMARKS.md)
- [Harness design and checks](HARNESS.md)
- [Machine-readable comparison](results/fireworks_gpt_oss_120b_harness.json)
- [Whole-month raw results](results/lines/fireworks_gpt_oss_120b_whole_month.jsonl)
- [Harness raw results](results/lines/fireworks_gpt_oss_120b_harness.jsonl)
- [Phase 0 research log](G0_REPORT.md)

Key tradeoffs:

- The harness costs about 2.8× more than whole-month inference, but the absolute
  measured difference is about $1.17 over 10,000 lines.
- The remaining bottleneck is many-bank-to-one matching: 50.2% group recall
  overall and 19.9% on 300-line months.
- Medium reasoning accounts for roughly 90% of harness completion tokens. A
  teammate's 500-line low-reasoning ablation retained 96.4% group F1 at a
  reported $0.00012 per line, making a full low-reasoning run the most
  promising cost experiment.
- The benchmark uses synthetic data. Real-bank validation, privacy controls,
  and human-review workflows are required before production use.

## Repository map

| Path | Purpose |
|---|---|
| `fireworks_gpt_oss.py` | Whole-month Fireworks baseline |
| `fireworks_gpt_oss_harness.py` | Fireworks GPT-OSS shortlist harness |
| `verify_submission.py` | Offline completeness, metric, and confidence-interval verifier |
| `HARNESS.md` | Exact candidate, prompt, assembly, and check design |
| `BENCHMARKS.md` | Primary and supporting cross-model results with caveats |
| `recon/gen.py` | Synthetic month generator |
| `recon/shortlist.py` | Candidate generation, prompt, checks, and group assembly |
| `recon/score.py` | Exact group and unmatched-item metrics |
| `recon/slm.py` | Shared harness runner and local backends |
| `phase0.py` | Earlier solver, local-model, and LoRA experiments |
| `data/` | Versioned synthetic evaluation and SFT data |
| `results/` | Versioned summaries and raw evidence |

## Optional local experiments

The historical Phase 0 experiments include Ollama baselines and an Apple
Silicon MLX LoRA experiment. They are not required for the Fireworks result.

```bash
# Ollama Llama 3.1 8B on the fixed 500-line sample
python3 ollama_llama31_8b_harness.py

# Optional MLX dependencies for phase0.py r06
python3 -m pip install -r requirements-mlx.txt
```

## Submission materials

`SUBMISSION.md` contains copy-ready project text, the technology list, evidence
summary, repository checklist, and a concise demo-video outline.

## Security

- API keys are read only from `FIREWORKS_API_KEY`.
- `.env`, virtual environments, model adapters, caches, and local logs are
  excluded by `.gitignore`.
- The included dataset is synthetic and contains no customer or bank data.

## License

MIT. See [LICENSE](LICENSE).
