#!/usr/bin/env python3
"""Recompute the submission's primary metrics from included raw artifacts.

No API key or third-party package is required. The bootstrap resamples entire
months so bank lines from the same reconciliation month stay clustered.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from recon.data import Month, load_months
from recon.score import score_month, summarize
from recon.slm import assembled_month_scores


DEFAULT_DATA = Path("data/phase0_hard.jsonl")
DEFAULT_WHOLE = Path("results/lines/fireworks_gpt_oss_120b_whole_month.jsonl")
DEFAULT_HARNESS = Path("results/lines/fireworks_gpt_oss_120b_harness.jsonl")
EXPECTED_MONTHS = 60
EXPECTED_LINES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--whole", type=Path, default=DEFAULT_WHOLE)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_261_102)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def assert_complete(
    months: list[Month], whole_rows: list[dict[str, Any]], harness_rows: list[dict[str, Any]]
) -> None:
    if len(months) != EXPECTED_MONTHS:
        raise SystemExit(f"Expected {EXPECTED_MONTHS} months, found {len(months)}")
    month_ids = {month.id for month in months}
    whole_ids = [row.get("month") for row in whole_rows]
    if len(whole_rows) != EXPECTED_MONTHS or set(whole_ids) != month_ids:
        raise SystemExit("Whole-month results are incomplete or contain unexpected month IDs")
    if len(whole_ids) != len(set(whole_ids)):
        raise SystemExit("Whole-month results contain duplicate month IDs")

    expected = {(month.id, bank.id) for month in months for bank in month.bank}
    actual = [(row.get("month"), row.get("bank")) for row in harness_rows]
    if len(expected) != EXPECTED_LINES:
        raise SystemExit(f"Expected {EXPECTED_LINES} dataset lines, found {len(expected)}")
    if len(harness_rows) != EXPECTED_LINES or set(actual) != expected:
        raise SystemExit("Harness results are incomplete or contain unexpected bank-line IDs")
    if len(actual) != len(set(actual)):
        raise SystemExit("Harness results contain duplicate bank-line IDs")


def aggregate(scores: list[dict[str, Any]]) -> tuple[float, float]:
    tp = sum(score["tp"] for score in scores)
    n_pred = sum(score["n_pred"] for score in scores)
    n_gold = sum(score["n_gold"] for score in scores)
    line_ok = sum(score["line_ok"] for score in scores)
    n_lines = sum(score["n_lines"] for score in scores)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, line_ok / n_lines


def percentile(values: list[float], fraction: float) -> float:
    values.sort()
    index = min(int(fraction * len(values)), len(values) - 1)
    return values[index]


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    """Exact paired sign-test p-value, excluding ties.

    The unit is one reconciliation month, so correlated bank lines within a
    month are not incorrectly treated as independent observations.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * lower_tail)


def bootstrap_delta(
    paired_scores: list[tuple[dict[str, Any], dict[str, Any]]], samples: int, seed: int
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    group_deltas: list[float] = []
    line_deltas: list[float] = []
    for _ in range(samples):
        sample = [paired_scores[rng.randrange(len(paired_scores))] for _ in paired_scores]
        whole = aggregate([pair[0] for pair in sample])
        harness = aggregate([pair[1] for pair in sample])
        group_deltas.append(harness[0] - whole[0])
        line_deltas.append(harness[1] - whole[1])
    return {
        "group_f1_95pct": [percentile(group_deltas, 0.025), percentile(group_deltas, 0.975)],
        "line_accuracy_95pct": [percentile(line_deltas, 0.025), percentile(line_deltas, 0.975)],
    }


def main() -> None:
    args = parse_args()
    for path in (args.data, args.whole, args.harness):
        if not path.exists():
            raise SystemExit(f"Missing required artifact: {path}")

    months = load_months(args.data)
    whole_rows = load_jsonl(args.whole)
    harness_rows = load_jsonl(args.harness)
    assert_complete(months, whole_rows, harness_rows)

    whole_groups = {row["month"]: row["groups"] for row in whole_rows}
    harness_groups = assembled_month_scores(months, harness_rows)
    whole_scores = [score_month(month, whole_groups[month.id]) for month in months]
    harness_scores = [score_month(month, harness_groups[month.id]) for month in months]
    whole_summary = summarize(
        [{"size": month.size, **score} for month, score in zip(months, whole_scores)]
    )
    harness_summary = summarize(
        [{"size": month.size, **score} for month, score in zip(months, harness_scores)]
    )

    whole_overall = aggregate(whole_scores)
    harness_overall = aggregate(harness_scores)
    wins = ties = 0
    for whole, harness in zip(whole_scores, harness_scores):
        whole_acc = whole["line_ok"] / whole["n_lines"]
        harness_acc = harness["line_ok"] / harness["n_lines"]
        wins += harness_acc > whole_acc
        ties += harness_acc == whole_acc
    intervals = bootstrap_delta(
        list(zip(whole_scores, harness_scores)), args.bootstrap_samples, args.seed
    )
    losses = len(months) - wins - ties
    sign_p = exact_two_sided_sign_test(wins, losses)
    result = {
        "complete": True,
        "months": len(months),
        "bank_lines": sum(len(month.bank) for month in months),
        "whole_month": {"group_f1": whole_overall[0], "line_accuracy": whole_overall[1]},
        "harness": {"group_f1": harness_overall[0], "line_accuracy": harness_overall[1]},
        "delta": {
            "group_f1": harness_overall[0] - whole_overall[0],
            "line_accuracy": harness_overall[1] - whole_overall[1],
        },
        "paired_bootstrap": {
            "unit": "month",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            **intervals,
        },
        "month_line_accuracy": {"wins": wins, "ties": ties, "losses": losses},
        "paired_sign_test": {
            "unit": "month",
            "metric": "line_accuracy",
            "ties_excluded": ties,
            "two_sided_p": sign_p,
        },
        "by_size": {
            size: {
                "whole_group_f1": whole_summary[size]["group_f1"],
                "harness_group_f1": harness_summary[size]["group_f1"],
                "whole_line_accuracy": whole_summary[size]["line_accuracy"],
                "harness_line_accuracy": harness_summary[size]["line_accuracy"],
            }
            for size in ("50", "150", "300")
        },
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Verified: {result['months']} months / {result['bank_lines']:,} bank lines")
    print(
        f"Whole month:  group F1 {whole_overall[0]:.4f}, "
        f"line accuracy {whole_overall[1]:.4f}"
    )
    print(
        f"Harness:      group F1 {harness_overall[0]:.4f}, "
        f"line accuracy {harness_overall[1]:.4f}"
    )
    print(
        f"Delta:        group F1 {result['delta']['group_f1']:+.4f}, "
        f"line accuracy {result['delta']['line_accuracy']:+.4f}"
    )
    group_ci = intervals["group_f1_95pct"]
    line_ci = intervals["line_accuracy_95pct"]
    print(
        "95% paired month-bootstrap CI: "
        f"group F1 [{group_ci[0]:+.4f}, {group_ci[1]:+.4f}], "
        f"line accuracy [{line_ci[0]:+.4f}, {line_ci[1]:+.4f}]"
    )
    print(f"Month wins/ties/losses: {wins}/{ties}/{losses}")
    print(f"Exact paired month sign-test: two-sided p={sign_p:.3g} (ties excluded)")


if __name__ == "__main__":
    main()
