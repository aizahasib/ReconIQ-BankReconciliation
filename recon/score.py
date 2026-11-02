"""Scoring shared by every system: group F1, line accuracy, unmatched P/R."""
from __future__ import annotations

from collections import defaultdict

from .data import Month


def _key(g: dict) -> frozenset:
    return frozenset([("B", b) for b in g["bank"]] + [("L", l) for l in g["ledger"]])


def score_month(m: Month, pred_groups: list[dict]) -> dict:
    """pred_groups: [{"bank": [ids], "ledger": [ids]}]; anything not in a group is predicted unmatched."""
    pred_groups = [g for g in pred_groups if g.get("bank") and g.get("ledger")]
    gold = {_key(g) for g in m.gold}
    pred = {_key(g) for g in pred_groups}
    tp = len(gold & pred)

    gold_of = {b: _key(g) for g in m.gold for b in g["bank"]}
    pred_of = {b: _key(g) for g in pred_groups for b in g["bank"]}
    line_ok = sum(gold_of.get(b.id) == pred_of.get(b.id) for b in m.bank)

    def unmatched(side: str, ids: list[str], groups) -> set[str]:
        used = {x for g in groups for x in g[side]}
        return set(ids) - used

    out = {"tp": tp, "n_pred": len(pred), "n_gold": len(gold), "line_ok": line_ok, "n_lines": len(m.bank)}
    for side, ids in (("bank", [b.id for b in m.bank]), ("ledger", [l.id for l in m.ledger])):
        gu, pu = unmatched(side, ids, m.gold), unmatched(side, ids, pred_groups)
        out[f"unm_{side}_tp"] = len(gu & pu)
        out[f"unm_{side}_pred"] = len(pu)
        out[f"unm_{side}_gold"] = len(gu)
    return out


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize(rows: list[dict]) -> dict:
    """Micro-average per-month score rows (each carries 'size') overall and by month size."""
    by: dict = defaultdict(list)
    for r in rows:
        by[r["size"]].append(r)
        by["all"].append(r)
    out = {}
    for k, rs in by.items():
        s = defaultdict(float)
        for r in rs:
            for f, v in r.items():
                if isinstance(v, (int, float)):
                    s[f] += v
        p, rc = _ratio(s["tp"], s["n_pred"]), _ratio(s["tp"], s["n_gold"])
        out[str(k)] = {
            "months": len(rs),
            "group_precision": p,
            "group_recall": rc,
            "group_f1": _ratio(2 * p * rc, p + rc),
            "line_accuracy": _ratio(s["line_ok"], s["n_lines"]),
            "unmatched_bank_p": _ratio(s["unm_bank_tp"], s["unm_bank_pred"]),
            "unmatched_bank_r": _ratio(s["unm_bank_tp"], s["unm_bank_gold"]),
            "unmatched_ledger_p": _ratio(s["unm_ledger_tp"], s["unm_ledger_pred"]),
            "unmatched_ledger_r": _ratio(s["unm_ledger_tp"], s["unm_ledger_gold"]),
            **({"usd_per_month": s["usd"] / len(rs)} if "usd" in s else {}),
            **({"sec_per_month": s["sec"] / len(rs)} if "sec" in s else {}),
        }
    return dict(sorted(out.items(), key=lambda kv: (kv[0] == "all", int(kv[0]) if kv[0].isdigit() else 0)))
