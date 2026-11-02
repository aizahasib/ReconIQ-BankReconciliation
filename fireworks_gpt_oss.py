#!/usr/bin/env python3
"""Run whole-month reconciliation with gpt-oss on Fireworks Serverless.

This is intentionally independent of the Phase 0 shortlist harness. It sends
each complete month (bank lines + ledger entries) to Fireworks in one request,
writes one resumable JSONL result per month, and optionally scores predictions
against the dataset's gold groups after inference.

No third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-120b"

SYSTEM_PROMPT = """You are an expert bank-reconciliation analyst.

Reconcile the complete bank statement against the complete general ledger for
one month. A valid match group has one or more bank line IDs and one or more
ledger entry IDs. Handle:
- one bank line to one ledger entry;
- one bank line to several ledger entries whose signed amounts net exactly;
- several partial bank payments to one ledger entry;
- processor payouts where a gross sale and negative fee net to the deposit;
- genuinely unmatched items on either side.

Use signed amounts, dates, descriptions, vendors, memos, check numbers, invoice
numbers, and accounts together. Equal amounts can be decoys, so text identity
matters. Do not force a match. Each ID may appear in at most one group. Omit
unmatched IDs entirely. Return only the requested JSON object."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bank": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "ledger": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["bank", "ledger"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


class FireworksError(RuntimeError):
    """An API or response-validation error."""


class ResponseValidationError(FireworksError):
    """The API succeeded, but the model's answer is not a valid reconciliation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fireworks gpt-oss on complete reconciliation months (no shortlist harness)."
    )
    parser.add_argument("--input", type=Path, default=Path("data/phase0_hard.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lines/fireworks_gpt_oss_120b_whole_month.jsonl"),
        help="Resumable JSONL output; completed month IDs are skipped.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--month-id", action="append", help="Only run this month ID (repeatable).")
    parser.add_argument("--limit", type=int, help="Run at most this many not-yet-completed months.")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=3,
        help="Additional model calls after an invalid answer (default: 3).",
    )
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Prompt for JSON without response_format (use only if the selected model rejects JSON schema).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and show request sizes without using an API key or calling Fireworks.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.validation_retries < 0:
        parser.error("--validation-retries cannot be negative")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    months: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                month = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            for key in ("id", "bank", "ledger"):
                if key not in month:
                    raise SystemExit(f"{path}:{line_no}: missing required field {key!r}")
            if month["id"] in seen:
                raise SystemExit(f"{path}:{line_no}: duplicate month ID {month['id']!r}")
            seen.add(month["id"])
            months.append(month)
    return months


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done.add(row["month"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise SystemExit(f"{path}:{line_no}: invalid resumable result row: {exc}") from exc
    return done


def format_amount(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) / 100:,.2f}"


def render_month(month: dict[str, Any]) -> str:
    bank = [
        {
            "id": item["id"],
            "date": item["date"],
            "amount": format_amount(item["amount"]),
            "description": item["desc"],
        }
        for item in month["bank"]
    ]
    ledger = [
        {
            "id": item["id"],
            "date": item["date"],
            "amount": format_amount(item["amount"]),
            "vendor_or_customer": item["vendor"],
            "account": item["account"],
            "memo": item["memo"],
        }
        for item in month["ledger"]
    ]
    payload = {"month_id": month["id"], "bank_lines": bank, "ledger_entries": ledger}
    return (
        "Reconcile this month. Amount strings are signed dollars. Return "
        '{"groups":[{"bank":["B..."],"ledger":["L..."]}, ...]}.\n\n'
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def request_body(month: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_month(month)},
        ],
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    if not args.no_structured_output:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reconciliation_groups", "schema": OUTPUT_SCHEMA},
        }
    return body


def api_call(body: dict[str, Any], api_key: str, timeout: float, retries: int) -> dict[str, Any]:
    encoded = json.dumps(body).encode("utf-8")
    transient_statuses = {408, 429, 500, 502, 503, 504, 520}
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=encoded,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code not in transient_statuses or attempt == retries:
                raise FireworksError(f"Fireworks HTTP {exc.code}: {detail[:1000]}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 30)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise FireworksError(f"Fireworks request failed after {attempt + 1} attempts: {exc}") from exc
            delay = min(2**attempt, 30)
        delay += random.random()
        print(f"  transient error; retrying in {delay:.1f}s ({attempt + 1}/{retries})", file=sys.stderr)
        time.sleep(delay)
    raise AssertionError("unreachable")


def parse_content(response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ResponseValidationError(
            f"Unexpected Fireworks response: {json.dumps(response)[:1000]}"
        ) from exc
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ResponseValidationError(f"Expected string JSON content, got {type(content).__name__}")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"Model returned invalid JSON: {content[:1000]}") from exc


def validate_groups(month: dict[str, Any], answer: dict[str, Any]) -> list[dict[str, list[str]]]:
    if not isinstance(answer, dict) or not isinstance(answer.get("groups"), list):
        raise ResponseValidationError("Model JSON must contain a 'groups' array")
    bank_ids = {item["id"] for item in month["bank"]}
    ledger_ids = {item["id"] for item in month["ledger"]}
    used_bank: set[str] = set()
    used_ledger: set[str] = set()
    groups: list[dict[str, list[str]]] = []
    for index, group in enumerate(answer["groups"]):
        if not isinstance(group, dict):
            raise ResponseValidationError(f"Group {index} is not an object")
        banks, ledgers = group.get("bank"), group.get("ledger")
        if not isinstance(banks, list) or not banks or not all(isinstance(x, str) for x in banks):
            raise ResponseValidationError(f"Group {index} has an invalid bank list")
        if not isinstance(ledgers, list) or not ledgers or not all(isinstance(x, str) for x in ledgers):
            raise ResponseValidationError(f"Group {index} has an invalid ledger list")
        if len(banks) != len(set(banks)) or len(ledgers) != len(set(ledgers)):
            raise ResponseValidationError(f"Group {index} contains a duplicate ID")
        unknown = (set(banks) - bank_ids) | (set(ledgers) - ledger_ids)
        if unknown:
            raise ResponseValidationError(f"Group {index} contains unknown IDs: {sorted(unknown)}")
        overlap = (set(banks) & used_bank) | (set(ledgers) & used_ledger)
        if overlap:
            raise ResponseValidationError(f"IDs appear in multiple groups: {sorted(overlap)}")
        used_bank.update(banks)
        used_ledger.update(ledgers)
        groups.append({"bank": banks, "ledger": ledgers})
    return groups


def merge_usage(total: dict[str, Any], current: dict[str, Any]) -> None:
    """Recursively sum token counts across validation attempts."""
    for key, value in current.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            nested = total.setdefault(key, {})
            if isinstance(nested, dict):
                merge_usage(nested, value)


def infer_valid_groups(
    month: dict[str, Any], args: argparse.Namespace, api_key: str
) -> tuple[list[dict[str, list[str]]], dict[str, Any], int, dict[str, Any]]:
    """Call the model until its answer passes cross-group ID validation."""
    total_usage: dict[str, Any] = {}
    last_error: ResponseValidationError | None = None
    for attempt in range(args.validation_retries + 1):
        body = request_body(month, args)
        if last_error is not None:
            body["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Your previous answer failed validation: "
                        f"{last_error}. Produce a fresh complete answer. In particular, every bank "
                        "and ledger ID may appear in at most one group."
                    ),
                }
            )
        response = api_call(body, api_key, args.timeout, args.retries)
        merge_usage(total_usage, response.get("usage", {}))
        try:
            groups = validate_groups(month, parse_content(response))
            return groups, response, attempt + 1, total_usage
        except ResponseValidationError as exc:
            last_error = exc
            if attempt < args.validation_retries:
                print(
                    f"  invalid model answer: {exc}; retrying validation "
                    f"({attempt + 1}/{args.validation_retries})",
                    file=sys.stderr,
                    flush=True,
                )
    assert last_error is not None
    raise last_error


def group_key(group: dict[str, Any]) -> frozenset[tuple[str, str]]:
    return frozenset(
        [("B", item) for item in group["bank"]] + [("L", item) for item in group["ledger"]]
    )


def score(month: dict[str, Any], predicted: list[dict[str, list[str]]]) -> dict[str, int]:
    gold_groups = month.get("gold")
    if not isinstance(gold_groups, list):
        return {}
    gold = {group_key(group) for group in gold_groups}
    pred = {group_key(group) for group in predicted}
    gold_of = {bank: group_key(group) for group in gold_groups for bank in group["bank"]}
    pred_of = {bank: group_key(group) for group in predicted for bank in group["bank"]}
    return {
        "tp": len(gold & pred),
        "n_pred": len(pred),
        "n_gold": len(gold),
        "line_ok": sum(gold_of.get(item["id"]) == pred_of.get(item["id"]) for item in month["bank"]),
        "n_lines": len(month["bank"]),
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def print_summary(output: Path) -> None:
    totals: defaultdict[str, float] = defaultdict(float)
    months = 0
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("score"):
                continue
            months += 1
            for key, value in row["score"].items():
                totals[key] += value
    if not months:
        return
    precision = ratio(totals["tp"], totals["n_pred"])
    recall = ratio(totals["tp"], totals["n_gold"])
    summary = {
        "months": months,
        "group_precision": precision,
        "group_recall": recall,
        "group_f1": ratio(2 * precision * recall, precision + recall),
        "bank_line_accuracy": ratio(totals["line_ok"], totals["n_lines"]),
    }
    print("Summary: " + json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    months = load_jsonl(args.input)
    if args.month_id:
        requested = set(args.month_id)
        known = {month["id"] for month in months}
        missing = requested - known
        if missing:
            raise SystemExit(f"Unknown --month-id values: {', '.join(sorted(missing))}")
        months = [month for month in months if month["id"] in requested]

    done = completed_ids(args.output)
    pending = [month for month in months if month["id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    if args.dry_run:
        print(f"Input months: {len(months)}; already complete: {len(done & {m['id'] for m in months})}")
        print(f"Would request: {len(pending)}")
        for month in pending:
            body = request_body(month, args)
            print(
                f"  {month['id']}: {len(month['bank'])} bank, {len(month['ledger'])} ledger, "
                f"{len(json.dumps(body).encode('utf-8')):,} request bytes"
            )
        return

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY in the environment; the script never stores it.")
    if not pending:
        print("Nothing to do; all selected months are already in the output file.")
        if args.output.exists():
            print_summary(args.output)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with args.output.open("a", encoding="utf-8") as handle:
        for position, month in enumerate(pending, 1):
            print(f"[{position}/{len(pending)}] {month['id']} ({len(month['bank'])} bank lines)", flush=True)
            started = time.monotonic()
            try:
                groups, response, validation_attempts, usage = infer_valid_groups(month, args, api_key)
            except ResponseValidationError as exc:
                failed.append(month["id"])
                print(
                    f"  SKIPPED after {args.validation_retries + 1} invalid answers: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            row = {
                "month": month["id"],
                "size": month.get("size", len(month["bank"])),
                "model": response.get("model", args.model),
                "reasoning_effort": args.reasoning_effort,
                "groups": groups,
                "usage": usage,
                "validation_attempts": validation_attempts,
                "seconds": time.monotonic() - started,
                "score": score(month, groups),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if row["score"]:
                s = row["score"]
                print(
                    f"  groups {s['tp']}/{s['n_gold']} exact; "
                    f"bank lines {s['line_ok']}/{s['n_lines']}; {row['seconds']:.1f}s",
                    flush=True,
                )
    print(f"Wrote {args.output}")
    if failed:
        print(
            "Skipped invalid months (left incomplete for the next run): " + ", ".join(failed),
            file=sys.stderr,
        )
    print_summary(args.output)


if __name__ == "__main__":
    main()
