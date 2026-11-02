"""Phase 0 (gate G0): the local runs behind the go/no-go decision.

The frontier baselines (R0.2 and the frontier-in-harness run) are not included; their results
are kept in results/ for reference.

    python phase0.py r01                 # mint 60 months + gold self-check (default and hardened regimes)
    python phase0.py r03                 # solver-only baseline S
    python phase0.py r04 --model phi4:14b --model qwen3:8b
    python phase0.py r04 --model phi4:14b --all-months   # Phase 0b: all 60 months -> results/r04_all.json
    python phase0.py r05                 # check FAR on every R0.4 run so far
    python phase0.py r06 --mlx-model mlx-community/Qwen3-8B-4bit

Every run writes results/<run>.json; per-line model outputs go to results/lines/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recon.data import load_months, save_months
from recon.gen import DEFAULT_PARAMS, HARD_PARAMS, amount_unresolvable, mint, self_check
from recon.score import score_month, summarize
from recon.shortlist import build_shortlist
from recon.solver import solve

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RES = ROOT / "results"
SIZES = [50, 150, 300]
BASE_SEED = 1
REGIMES = {"default": DEFAULT_PARAMS, "hard": HARD_PARAMS}
# The shortlist runs use one month per size from the active regime: 500 bank lines.
SLM_MONTHS = ["m50_00", "m150_00", "m300_00"]


def months(regime: str):
    return load_months(DATA / f"phase0_{regime}.jsonl")


def write(name: str, obj: dict) -> None:
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj, indent=2))


def r01(a) -> None:
    out = {}
    for regime, params in REGIMES.items():
        ms = mint(SIZES, 20, BASE_SEED, params)
        save_months(ms, DATA / f"phase0_{regime}.jsonl")
        fails = {m.id: self_check(m) for m in ms}
        unres = sum(amount_unresolvable(m)[0] for m in ms)
        n = sum(len(m.bank) for m in ms)
        ambiguous = sum(build_shortlist(m, b).n_options >= 2 for m in ms for b in m.bank)
        types: dict = {}
        for m in ms:
            for g in m.gold:
                types[g["type"]] = types.get(g["type"], 0) + len(g["bank"])
        out[regime] = {
            "months": len(ms), "bank_lines": n,
            "self_check_pass": sum(not v for v in fails.values()),
            "self_check_failures": {k: v[:3] for k, v in fails.items() if v},
            "amount_unresolvable_frac": unres / n,
            "needs_text_frac": ambiguous / n,
            "bank_lines_by_group_type": types,
            "params": params,
        }
        out[regime]["go"] = out[regime]["self_check_pass"] == len(ms) and out[regime]["amount_unresolvable_frac"] >= 0.30
    write("r01", out)


def r03(a) -> None:
    out = {}
    for regime in REGIMES:
        rows = []
        for m in months(regime):
            t0 = time.time()
            groups = solve(m)
            rows.append({"size": m.size, "sec": time.time() - t0, "usd": 0.0, **score_month(m, groups)})
        s = summarize(rows)
        out[regime] = {"summary": s, "go": s["all"]["group_f1"] <= 0.80}
    write("r03", out)


def r04(a) -> None:
    from recon.slm import Ollama, evaluate_rows, run_shortlist

    if a.all_months:  # Phase 0b: every month of the regime, written to its own result file
        r04_all(a)
        return
    ms = [m for m in months(a.regime) if m.id in SLM_MONTHS]
    out = json.loads((RES / "r04.json").read_text()) if (RES / "r04.json").exists() else {}
    for model in a.model:
        backend = Ollama(model)
        tag = f"{a.regime}_{model.replace(':', '_')}" + ("_arity" if a.arity_hint else "")
        rows = run_shortlist(ms, backend, RES / "lines" / f"r04_{tag}.jsonl", a.arity_hint)
        out[tag] = {"model": model, "regime": a.regime, "arity_hint": a.arity_hint, "months": SLM_MONTHS,
                    **evaluate_rows(ms, rows)}
        out[tag]["go"] = out[tag]["raw_accuracy"] >= 0.50
        print(tag, json.dumps(out[tag], indent=1))
    write("r04", out)


def r04_all(a) -> None:
    """Shortlist + local model on all 60 months; seeds the line log from the 3-month R0.4 run."""
    import shutil

    from recon.slm import Ollama, assembled_month_scores, evaluate_rows, run_shortlist

    ms = months(a.regime)
    out = json.loads((RES / "r04_all.json").read_text()) if (RES / "r04_all.json").exists() else {}
    for model in a.model:
        tag = f"{a.regime}_{model.replace(':', '_')}"
        log = RES / "lines" / f"r04all_{tag}.jsonl"
        seed = RES / "lines" / f"r04_{tag}.jsonl"
        if not log.exists() and seed.exists():
            shutil.copy(seed, log)  # the SLM months are a subset; don't redo them
        rows = run_shortlist(ms, Ollama(model), log)
        groups = assembled_month_scores(ms, rows)
        assembled = summarize([{"size": m.size, **score_month(m, groups[m.id])} for m in ms])
        out[tag] = {"model": model, "regime": a.regime, "months": len(ms), **evaluate_rows(ms, rows),
                    "assembled_no_escalation": assembled}
        print(tag, json.dumps({k: v["group_f1"] for k, v in assembled.items()}), flush=True)
    write("r04_all", out)


def r05(a) -> None:
    from recon.slm import assembled_month_scores, evaluate_rows

    out = {}
    for path in sorted((RES / "lines").glob("r0[46]_*.jsonl")):
        regime = path.stem.split("_")[1]
        ms = [m for m in months(regime) if m.id in SLM_MONTHS]
        rows = [json.loads(l) for l in path.open()]
        ev = evaluate_rows(ms, rows)
        groups = assembled_month_scores(ms, rows)
        assembled = summarize([{"size": m.size, **score_month(m, groups[m.id])} for m in ms if m.id in groups])
        out[path.stem] = {
            "admission_rate": ev["admission_rate"], "far_all_checks": ev["far_all_checks"],
            "far_per_check": ev["far_per_check"],
            "assembled_group_f1_no_escalation": {k: v["group_f1"] for k, v in assembled.items()},
            "go": ev["far_all_checks"] <= 0.10 and ev["admission_rate"] >= 0.40,
        }
    write("r05", out)


def r06(a) -> None:
    from recon import sft
    from recon.slm import MLX, evaluate_rows, run_shortlist

    data_dir = DATA / f"sft_{a.regime}_{a.n_train}"
    adapter = ROOT / "adapters" / f"{a.mlx_model.split('/')[-1]}_{a.regime}_{a.n_train}"
    info = {}
    if not (data_dir / "train.jsonl").exists():
        info = sft.make_data(data_dir, a.n_train, REGIMES[a.regime])
    info["round_trip_failures"] = sft.round_trip(data_dir)
    assert info["round_trip_failures"] == 0
    if not (adapter / "adapters.safetensors").exists():
        t0 = time.time()
        sft.train(a.mlx_model, data_dir, adapter, a.n_train)
        info["train_sec"] = time.time() - t0

    ms = [m for m in months(a.regime) if m.id in SLM_MONTHS]
    out = {"data": info}
    short = a.mlx_model.split("/")[-1]
    for tag, adp in ((f"{a.regime}_{short}_base", None), (f"{a.regime}_{short}_lora{a.n_train}", adapter)):
        backend = MLX(a.mlx_model, str(adp) if adp else None)
        rows = run_shortlist(ms, backend, RES / "lines" / f"r06_{tag}.jsonl")
        out[tag] = evaluate_rows(ms, rows)
        del backend
    base, tuned = out[f"{a.regime}_{short}_base"], out[f"{a.regime}_{short}_lora{a.n_train}"]
    out["delta_raw_accuracy_pp"] = 100 * (tuned["raw_accuracy"] - base["raw_accuracy"])
    out["delta_far_pp"] = 100 * (tuned["far_all_checks"] - base["far_all_checks"])
    out["go"] = out["delta_raw_accuracy_pp"] >= 10 and out["delta_far_pp"] <= 2
    write(f"r06_{a.n_train}", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", choices=["r01", "r03", "r04", "r05", "r06"])
    p.add_argument("--regime", default="hard", choices=list(REGIMES))
    p.add_argument("--model", action="append", default=None, help="ollama model for r04 (repeatable)")
    p.add_argument("--arity-hint", action="store_true")
    p.add_argument("--all-months", action="store_true", help="r04: every month of the regime, not just SLM_MONTHS")
    p.add_argument("--mlx-model", default="mlx-community/Qwen3-8B-4bit")
    p.add_argument("--n-train", type=int, default=2000)
    a = p.parse_args()
    a.model = a.model or ["phi4:14b", "qwen3:8b"]
    globals()[a.run](a)


if __name__ == "__main__":
    main()
