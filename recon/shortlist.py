"""R0.4 shortlist format and R0.5 checks.

The unit of work is one bank line: the filter hands the model at most 10 ledger
candidates plus the exact-sum sets it found among them, and the model names the
ledger entries this bank line settles (or none).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .data import BankLine, LedgerEntry, Month, day, fmt_amount
from .gen import DATE_WINDOW
from .solver import PARTIAL_WINDOW, in_window, subset_sums

MAX_CANDIDATES = 10


@dataclass
class Shortlist:
    bank: BankLine
    candidates: list[LedgerEntry]  # shown as C1..Cn in this order
    sets: list[tuple[str, ...]]  # exact-sum sets, ledger ids, all inside candidates
    partials: dict[str, list[str]] = field(default_factory=dict)  # ledger id -> sibling bank ids
    gold: list[str] = field(default_factory=list)  # gold ledger ids for this bank line's group

    @property
    def n_options(self) -> int:
        return len(self.sets) + len(self.partials)

    def cid(self, lid: str) -> str:
        return f"C{[c.id for c in self.candidates].index(lid) + 1}"

    def lid(self, cid: str) -> str | None:
        m = re.fullmatch(r"C(\d+)", cid.strip())
        if not m or not (1 <= int(m.group(1)) <= len(self.candidates)):
            return None
        return self.candidates[int(m.group(1)) - 1].id

    @property
    def gold_in_shortlist(self) -> bool:
        ids = {c.id for c in self.candidates}
        return all(l in ids for l in self.gold)


def build_shortlist(m: Month, b: BankLine, cap: int = MAX_CANDIDATES) -> Shortlist:
    lb = m.ledger_by_id()
    bd = day(b.date)
    pool = [l for l in m.ledger if in_window(b.date, l.date)]
    sets = subset_sums(b.amount, [(l.id, l.amount) for l in pool], 4, cap=60)

    def spread(s) -> int:
        return max(abs(bd - day(lb[x].date)) for x in s)

    sets.sort(key=lambda s: (len(s), spread(s), s))

    partials: dict[str, list[str]] = {}
    for l in m.ledger:
        if (l.amount > 0) != (b.amount > 0) or abs(l.amount) <= abs(b.amount):
            continue
        if not in_window(b.date, l.date, PARTIAL_WINDOW):
            continue
        others = [(x.id, x.amount) for x in m.bank if x.id != b.id and (x.amount > 0) == (l.amount > 0)
                  and in_window(x.date, l.date, PARTIAL_WINDOW)]
        sib = subset_sums(l.amount - b.amount, others, 2, cap=5)
        if sib:
            partials[l.id] = list(sib[0])

    ordered: list[str] = []
    for s in sets:
        for x in s:
            if x not in ordered:
                ordered.append(x)
    for x in partials:
        if x not in ordered:
            ordered.append(x)
    ordered = ordered[:cap]
    keep = set(ordered)
    sets = [s for s in sets if set(s) <= keep]
    partials = {k: v for k, v in partials.items() if k in keep}
    cands = sorted((lb[x] for x in ordered), key=lambda l: (l.date, l.id))

    gold = next((g["ledger"] for g in m.gold if b.id in g["bank"]), [])
    return Shortlist(b, cands, sets, partials, sorted(gold))


SYSTEM_PROMPT = (
    "You reconcile one bank statement line against a business's ledger. "
    "A bank line can settle one ledger entry, several ledger entries (one deposit or payment covering several invoices, "
    "or a card-processor payout where the gross sale and a negative fee entry net to the bank amount), "
    "or be one of several partial payments toward a single larger ledger entry. Some bank lines have no ledger match. "
    "Several candidates may share the same amount; use the vendor, customer, memo, check number and description text to decide. "
    'Answer with JSON only: {"ledger": ["C1", ...]} listing the candidate ids this bank line settles, or {"ledger": []} if none.'
)


def _arity(sl: Shortlist) -> str:
    sizes = sorted({len(s) for s in sl.sets})
    parts = []
    if 1 in sizes:
        parts.append("one-to-one")
    if any(k > 1 for k in sizes):
        parts.append("one-to-many")
    if sl.partials:
        parts.append("partial payment")
    return ", ".join(parts) or "none found"


def render_prompt(sl: Shortlist, arity_hint: bool = False) -> str:
    b = sl.bank
    lines = [f"Bank line: {b.date} | {fmt_amount(b.amount)} | {b.desc}", "",
             "Ledger candidates (id | date | amount | vendor/customer | account | memo):"]
    if not sl.candidates:
        lines.append("(none)")
    for i, c in enumerate(sl.candidates, 1):
        lines.append(f"C{i} | {c.date} | {fmt_amount(c.amount)} | {c.vendor} | {c.account} | {c.memo}")
    lines.append("")
    if sl.sets:
        lines.append("Candidate sets whose amounts sum exactly to the bank amount: "
                     + "; ".join("{" + ", ".join(sl.cid(x) for x in s) + "}" for s in sl.sets))
    else:
        lines.append("No candidate set sums exactly to the bank amount.")
    if sl.partials:
        lines.append("Possible partial-payment targets (this bank line plus other bank lines sum to the entry): "
                     + "; ".join(sl.cid(k) for k in sl.partials))
    if arity_hint:
        lines.append(f"Filter's arity hint: {_arity(sl)}")
    lines.append("")
    lines.append('Which candidates does this bank line settle? Reply with JSON only: {"ledger": [...]}')
    return "\n".join(lines)


def target_json(sl: Shortlist) -> str:
    return json.dumps({"ledger": [sl.cid(x) for x in sl.gold if x in {c.id for c in sl.candidates}]})


def parse_answer(sl: Shortlist, text: str) -> list[str] | None:
    """Map the model's C-ids back to ledger ids; None means unparseable or an id outside the shortlist."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ids = obj.get("ledger") if isinstance(obj, dict) else None
    if not isinstance(ids, list):
        return None
    out = []
    for c in ids:
        lid = sl.lid(str(c))
        if lid is None:
            return None
        out.append(lid)
    return sorted(set(out))


# ---- R0.5 checks ---------------------------------------------------------------
CHECKS = ("sum", "date", "unique")


def run_checks(m: Month, proposals: dict[str, list[str] | None]) -> dict[str, dict]:
    """Per bank line: pass/fail per check plus diagnostics, for non-empty proposals.

    Bank lines proposing the identical ledger set form one cluster (that is how a
    many-to-one group is assembled); the sum check is applied to the cluster.
    """
    bb, lb = m.bank_by_id(), m.ledger_by_id()
    clusters: dict[tuple, list[str]] = defaultdict(list)
    for bid, p in proposals.items():
        if p:
            clusters[tuple(p)].append(bid)
    claimed: dict[str, set] = defaultdict(set)
    for key in clusters:
        for l in key:
            claimed[l].add(key)

    out = {}
    for key, bids in clusters.items():
        sb = sum(bb[x].amount for x in bids)
        sl = sum(lb[l].amount for l in key)
        window = PARTIAL_WINDOW if len(bids) > 1 else DATE_WINDOW
        for bid in bids:
            bad_dates = [l for l in key if not in_window(bb[bid].date, lb[l].date, window)]
            shared = sorted({l for l in key if len(claimed[l]) > 1})
            out[bid] = {
                "sum": sb == sl,
                "date": not bad_dates,
                "unique": not shared,
                "diag": "; ".join(filter(None, [
                    "" if sb == sl else f"bank total {fmt_amount(sb)} != ledger total {fmt_amount(sl)}",
                    "" if not bad_dates else f"outside date window: {bad_dates}",
                    "" if not shared else f"ledger entries also claimed by another group: {shared}",
                ])),
                "cluster": sorted(bids),
            }
    return out


def assemble_groups(proposals: dict[str, list[str] | None]) -> list[dict]:
    clusters: dict[tuple, list[str]] = defaultdict(list)
    for bid, p in proposals.items():
        if p:
            clusters[tuple(p)].append(bid)
    return [{"bank": sorted(b), "ledger": list(k)} for k, b in clusters.items()]
