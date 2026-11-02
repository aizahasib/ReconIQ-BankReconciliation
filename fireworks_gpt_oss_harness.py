#!/usr/bin/env python3
"""Run gpt-oss on the Phase 0 shortlist harness through Fireworks Serverless.

This uses the same input months as ``fireworks_gpt_oss.py``, but the unit of
work is one bank line plus the harness-generated ledger shortlist. Results are
resumable per bank line and can be compared with completed whole-month results
on the exact same month IDs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fireworks_gpt_oss import (
    DEFAULT_MODEL,
    ResponseValidationError,
    api_call,
    merge_usage,
    parse_content,
)
from recon.data import Month, load_months
from recon.score import score_month, summarize
from recon.shortlist import SYSTEM_PROMPT, build_shortlist, parse_answer, render_prompt
from recon.slm import assembled_month_scores, evaluate_rows


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "ledger": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        }
    },
    "required": ["ledger"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fireworks gpt-oss inside the Phase 0 shortlist harness."
    )
    parser.add_argument("--input", type=Path, default=Path("data/phase0_hard.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lines/fireworks_gpt_oss_120b_harness.jsonl"),
        help="Resumable per-bank-line JSONL output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/fireworks_gpt_oss_120b_harness.json"),
    )
    parser.add_argument(
        "--whole-month-output",
        type=Path,
        default=Path("results/lines/fireworks_gpt_oss_120b_whole_month.jsonl"),
        help="Whole-month results used for same-month comparison when available.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--month-id", action="append", help="Only run this month ID (repeatable).")
    parser.add_argument("--limit-lines", type=int, help="Run at most this many unfinished bank lines.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API requests (default: 8).")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-tokens", type=int, default=4_096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5, help="Retries for transient HTTP errors.")
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=2,
        help="Additional model calls after an unparseable/out-of-shortlist answer.",
    )
    parser.add_argument("--arity-hint", action="store_true", help="Include the harness arity hint.")
    parser.add_argument("--no-structured-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("limit_lines", "workers", "max_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("retries", "validation_retries"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row["month"], row["bank"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise SystemExit(f"{path}:{line_no}: invalid result row: {exc}") from exc
            if key in rows:
                raise SystemExit(f"{path}:{line_no}: duplicate result for {key[0]} {key[1]}")
            rows[key] = row
    return rows


def request_body(system: str, user: str, args: argparse.Namespace, correction: str | None) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if correction:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your previous answer was invalid: {correction}. Return a fresh JSON answer "
                    "using only candidate IDs shown above."
                ),
            }
        )
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    if not args.no_structured_output:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ledger_selection", "schema": ANSWER_SCHEMA},
        }
    return body


def infer_line(
    month: Month, bank_index: int, args: argparse.Namespace, api_key: str
) -> dict[str, Any]:
    bank = month.bank[bank_index]
    shortlist = build_shortlist(month, bank)
    prompt = render_prompt(shortlist, args.arity_hint)
    started = time.monotonic()
    total_usage: dict[str, Any] = {}
    last_error: str | None = None
    raw = ""
    response: dict[str, Any] = {}
    for attempt in range(args.validation_retries + 1):
        response = api_call(
            request_body(SYSTEM_PROMPT, prompt, args, last_error),
            api_key,
            args.timeout,
            args.retries,
        )
        merge_usage(total_usage, response.get("usage", {}))
        try:
            answer = parse_content(response)
            raw = json.dumps(answer, ensure_ascii=False)
        except ResponseValidationError as exc:
            last_error = str(exc)
        else:
            prediction = parse_answer(shortlist, raw)
            if prediction is not None:
                return {
                    "month": month.id,
                    "size": month.size,
                    "bank": bank.id,
                    "pred": prediction,
                    "raw": raw,
                    "gold": shortlist.gold,
                    "gold_in_shortlist": shortlist.gold_in_shortlist,
                    "n_cands": len(shortlist.candidates),
                    "n_options": shortlist.n_options,
                    "sec": time.monotonic() - started,
                    "usage": total_usage,
                    "validation_attempts": attempt + 1,
                    "model": response.get("model", args.model),
                }
            last_error = "answer contained an invalid or out-of-shortlist candidate ID"
        if attempt < args.validation_retries:
            print(
                f"  {month.id} {bank.id}: invalid answer; validation retry "
                f"{attempt + 1}/{args.validation_retries}",
                file=sys.stderr,
                flush=True,
            )
    raise ResponseValidationError(f"{month.id} {bank.id}: {last_error}")


def usage_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for row in rows:
        merge_usage(total, row.get("usage", {}))
    return total


def load_whole_month_groups(path: Path) -> dict[str, list[dict[str, list[str]]]]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["month"]] = row["groups"]
    return result


def make_summary(
    months: list[Month], rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    by_month: dict[str, set[str]] = {}
    for row in rows:
        by_month.setdefault(row["month"], set()).add(row["bank"])
    complete = [m for m in months if by_month.get(m.id) == {b.id for b in m.bank}]
    complete_ids = {m.id for m in complete}
    complete_rows = [row for row in rows if row["month"] in complete_ids]

    line_metrics = evaluate_rows(complete, complete_rows) if complete else {}
    line_metrics.pop("usd_per_line", None)
    assembled = assembled_month_scores(complete, complete_rows) if complete else {}
    assembled_summary = (
        summarize([{"size": m.size, **score_month(m, assembled[m.id])} for m in complete])
        if complete
        else {}
    )
    summary: dict[str, Any] = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "input": str(args.input),
        "selected_months": len(months),
        "complete_months": len(complete),
        "complete_lines": len(complete_rows),
        "usage": usage_totals(complete_rows),
        "harness_line_metrics": line_metrics,
        "assembled_no_escalation": assembled_summary,
    }

    whole = load_whole_month_groups(args.whole_month_output)
    equivalent = [m for m in complete if m.id in whole]
    if equivalent:
        harness_scores = [
            {"size": m.size, **score_month(m, assembled[m.id])} for m in equivalent
        ]
        whole_scores = [
            {"size": m.size, **score_month(m, whole[m.id])} for m in equivalent
        ]
        summary["equivalent_comparison"] = {
            "months": len(equivalent),
            "month_ids": [m.id for m in equivalent],
            "harness": summarize(harness_scores),
            "whole_month": summarize(whole_scores),
        }
    return summary


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    months = load_months(args.input)
    if args.month_id:
        requested = set(args.month_id)
        known = {m.id for m in months}
        missing = requested - known
        if missing:
            raise SystemExit(f"Unknown --month-id values: {', '.join(sorted(missing))}")
        months = [m for m in months if m.id in requested]

    existing = load_existing(args.output)
    tasks = [
        (month, index)
        for month in months
        for index, bank in enumerate(month.bank)
        if (month.id, bank.id) not in existing
    ]
    if args.limit_lines is not None:
        tasks = tasks[: args.limit_lines]
    selected_keys = {(m.id, b.id) for m in months for b in m.bank}
    completed = len(selected_keys & set(existing))

    print(
        f"Selected {len(months)} months / {len(selected_keys)} bank lines; "
        f"already complete {completed}; pending this run {len(tasks)}"
    )
    if args.dry_run:
        for month, index in tasks[:5]:
            shortlist = build_shortlist(month, month.bank[index])
            print(
                f"  {month.id} {month.bank[index].id}: {len(shortlist.candidates)} candidates, "
                f"{shortlist.n_options} options"
            )
        return

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY in the environment; the script never stores it.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    newly_completed = 0
    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures: dict[Future[dict[str, Any]], tuple[Month, int]] = {
            pool.submit(infer_line, month, index, args, api_key): (month, index)
            for month, index in tasks
        }
        for future in as_completed(futures):
            month, index = futures[future]
            bank = month.bank[index]
            try:
                row = future.result()
            except ResponseValidationError as exc:
                failed.append(f"{month.id}/{bank.id}")
                print(f"  SKIPPED {exc}", file=sys.stderr, flush=True)
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            existing[(month.id, bank.id)] = row
            newly_completed += 1
            if newly_completed % 25 == 0 or newly_completed == len(tasks):
                correct = sum(
                    item["pred"] == item["gold"]
                    for key, item in existing.items()
                    if key in selected_keys
                )
                denominator = len(selected_keys & set(existing))
                print(
                    f"  completed {newly_completed}/{len(tasks)} new lines; "
                    f"running raw accuracy {correct / denominator:.3f}",
                    flush=True,
                )

    rows = [row for key, row in existing.items() if key in selected_keys]
    summary = make_summary(months, rows, args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")
    if failed:
        print(
            "Skipped lines (left incomplete for the next run): " + ", ".join(failed),
            file=sys.stderr,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
