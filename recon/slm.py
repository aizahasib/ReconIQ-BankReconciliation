"""Local model backends and the shortlist runner behind R0.4-R0.6."""
from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from .data import Month
from .shortlist import CHECKS, SYSTEM_PROMPT, assemble_groups, build_shortlist, parse_answer, render_prompt, run_checks


class Ollama:
    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 4096):
        self.model, self.host, self.num_ctx = model, host, num_ctx
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 200},
        }
        if self.model.startswith("qwen3"):
            body["think"] = False  # zero-shot, no reasoning trace; matches the SFT target format
        req = urllib.request.Request(f"{self.host}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]["content"]


class MLX:
    """MLX-LM backend, used for R0.6 so base and LoRA runs share one engine and chat template."""

    def __init__(self, model: str, adapter: str | None = None):
        from mlx_lm import load

        self.model, self.tok = load(model, adapter_path=adapter)
        self.name = f"mlx:{model}" + (f"+{Path(adapter).name}" if adapter else "")

    def complete(self, system: str, user: str) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = apply_template(self.tok, msgs, add_generation_prompt=True)
        return generate(self.model, self.tok, prompt, max_tokens=200, sampler=make_sampler(temp=0.0))


def apply_template(tok, msgs: list[dict], add_generation_prompt: bool) -> str:
    try:  # Qwen3 templates take enable_thinking; others ignore unknown kwargs or raise TypeError
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def run_shortlist(months: list[Month], backend, out_path: Path, arity_hint: bool = False) -> list[dict]:
    """Run the model on every bank line of every month; one JSON row per bank line, resumable."""
    done = {}
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done[(r["month"], r["bank"])] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("a") as f:
        for m in months:
            for i, b in enumerate(m.bank):
                if (m.id, b.id) in done:
                    rows.append(done[(m.id, b.id)])
                    continue
                sl = build_shortlist(m, b)
                t0 = time.time()
                text = backend.complete(SYSTEM_PROMPT, render_prompt(sl, arity_hint))
                pred = parse_answer(sl, text)
                r = {"month": m.id, "size": m.size, "bank": b.id, "pred": pred, "raw": text,
                     "gold": sl.gold, "gold_in_shortlist": sl.gold_in_shortlist, "n_cands": len(sl.candidates),
                     "n_options": sl.n_options, "sec": time.time() - t0, "usd": getattr(backend, "last_usd", 0.0)}
                f.write(json.dumps(r) + "\n")
                f.flush()
                rows.append(r)
                if (i + 1) % 25 == 0:
                    acc = sum(x["pred"] == x["gold"] for x in rows) / len(rows)
                    print(f"  {backend.name} {m.id} {i + 1}/{len(m.bank)} running acc {acc:.3f}", flush=True)
    return rows


def evaluate_rows(months: list[Month], rows: list[dict]) -> dict:
    """R0.4 raw accuracy and R0.5 check admission / false-accept rates from per-line rows."""
    by_month = defaultdict(dict)
    for r in rows:
        by_month[r["month"]][r["bank"]] = r
    mid = {m.id: m for m in months}

    tot = defaultdict(float)
    per_check_pass = defaultdict(float)
    per_check_wrong = defaultdict(float)
    for m_id, lines in by_month.items():
        m = mid[m_id]
        proposals = {b: r["pred"] for b, r in lines.items()}
        checks = run_checks(m, proposals)
        for b, r in lines.items():
            ok = r["pred"] == r["gold"]
            tot["n"] += 1
            tot["correct"] += ok
            tot["parse_fail"] += r["pred"] is None
            tot["gold_in_shortlist"] += r["gold_in_shortlist"]
            tot["ambiguous"] += r["n_options"] >= 2
            tot["ambiguous_correct"] += ok and r["n_options"] >= 2
            tot["sec"] += r.get("sec", 0)
            tot["usd"] += r.get("usd", 0)
            c = checks.get(b)
            if c is None:
                continue
            tot["nonempty"] += 1
            for k in CHECKS:
                if c[k]:
                    per_check_pass[k] += 1
                    per_check_wrong[k] += not ok
            if all(c[k] for k in CHECKS):
                tot["admitted"] += 1
                tot["admitted_wrong"] += not ok
    n = tot["n"] or 1
    return {
        "lines": int(tot["n"]),
        "raw_accuracy": tot["correct"] / n,
        "ambiguous_line_accuracy": tot["ambiguous_correct"] / (tot["ambiguous"] or 1),
        "parse_fail_rate": tot["parse_fail"] / n,
        "gold_in_shortlist": tot["gold_in_shortlist"] / n,
        "admission_rate": tot["admitted"] / n,
        "far_all_checks": tot["admitted_wrong"] / (tot["admitted"] or 1),
        "far_per_check": {k: per_check_wrong[k] / (per_check_pass[k] or 1) for k in CHECKS},
        "sec_per_line": tot["sec"] / n,
        "usd_per_line": tot["usd"] / n,
    }


def assembled_month_scores(months: list[Month], rows: list[dict]):
    """Month-level groups assembled from per-line proposals, for scoring with score_month."""
    by_month = defaultdict(dict)
    for r in rows:
        by_month[r["month"]][r["bank"]] = r["pred"]
    return {m.id: assemble_groups(by_month[m.id]) for m in months if m.id in by_month}
