#!/usr/bin/env python3
"""Run Llama 3.1 8B in Ollama on the 500-line Phase 0 harness sample.

The fixed sample is the first hard-regime month at each dataset size:
``m50_00``, ``m150_00``, and ``m300_00`` (500 bank lines total). Results are
resumable per bank line.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from recon.data import Month, load_months
from recon.score import score_month, summarize
from recon.slm import Ollama, assembled_month_scores, evaluate_rows, run_shortlist


SAMPLE_MONTH_IDS = ("m50_00", "m150_00", "m300_00")
DEFAULT_MODEL = "llama3.1:8b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ollama Llama 3.1 8B through the shortlist harness on 500 bank lines."
    )
    parser.add_argument("--input", type=Path, default=Path("data/phase0_hard.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=4_096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lines/ollama_llama31_8b_harness_500.jsonl"),
        help="Resumable per-bank-line output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/ollama_llama31_8b_harness_500.json"),
    )
    parser.add_argument("--arity-hint", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected dataset sample without contacting Ollama.",
    )
    args = parser.parse_args()
    if args.num_ctx < 1:
        parser.error("--num-ctx must be at least 1")
    return args


def select_sample(path: Path) -> list[Month]:
    months = load_months(path)
    by_id = {month.id: month for month in months}
    missing = [month_id for month_id in SAMPLE_MONTH_IDS if month_id not in by_id]
    if missing:
        raise SystemExit(f"Input is missing sample months: {', '.join(missing)}")
    sample = [by_id[month_id] for month_id in SAMPLE_MONTH_IDS]
    sizes = [len(month.bank) for month in sample]
    if sizes != [50, 150, 300]:
        raise SystemExit(f"Expected sample sizes [50, 150, 300], found {sizes}")
    return sample


def completed_line_count(path: Path, selected: set[tuple[str, str]]) -> int:
    if not path.exists():
        return 0
    completed: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row["month"], row["bank"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise SystemExit(f"{path}:{line_no}: invalid result row: {exc}") from exc
            if key in selected:
                completed.add(key)
    return len(completed)


def check_ollama(host: str, model: str) -> None:
    request = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {host}. If it is not installed, run:\n\n"
            "  curl -fsSL https://ollama.com/install.sh | sh\n\n"
            "Then start Ollama (or run `ollama serve`), pull the model with "
            f"`ollama pull {model}`, and rerun this script.\nDetails: {exc}"
        ) from exc

    installed = {
        value
        for item in payload.get("models", [])
        for value in (item.get("name"), item.get("model"))
        if value
    }
    aliases = {model, model.split(":", 1)[0] + ":latest"}
    if not (installed & aliases):
        raise SystemExit(
            f"Ollama model {model!r} is not installed. Run:\n\n  ollama pull {model}\n"
        )


def build_summary(
    months: list[Month], rows: list[dict[str, Any]], model: str, arity_hint: bool
) -> dict[str, Any]:
    groups = assembled_month_scores(months, rows)
    assembled = summarize(
        [{"size": month.size, **score_month(month, groups[month.id])} for month in months]
    )
    return {
        "model": model,
        "backend": "ollama",
        "regime": "hard",
        "arity_hint": arity_hint,
        "month_ids": list(SAMPLE_MONTH_IDS),
        "months": len(months),
        "bank_lines": sum(len(month.bank) for month in months),
        "harness_line_metrics": evaluate_rows(months, rows),
        "assembled_no_escalation": assembled,
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    months = select_sample(args.input)
    selected = {(month.id, bank.id) for month in months for bank in month.bank}
    completed = completed_line_count(args.output, selected)
    print(
        f"Sample: {', '.join(SAMPLE_MONTH_IDS)}; {len(selected)} bank lines; "
        f"already complete: {completed}; remaining: {len(selected) - completed}"
    )
    if args.dry_run:
        return

    check_ollama(args.host, args.model)
    backend = Ollama(args.model, host=args.host.rstrip("/"), num_ctx=args.num_ctx)
    rows = run_shortlist(months, backend, args.output, args.arity_hint)
    summary = build_summary(months, rows, args.model, args.arity_hint)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
