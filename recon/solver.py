"""Amount-and-date filter plus bounded exact-sum search. Used by R0.3 (Baseline S) and the shortlist builder.

No text is read anywhere in this module.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .data import Month, day
from .gen import DATE_WINDOW

PARTIAL_WINDOW = (-2, 20)  # partial payments can trail the invoice by a few weeks


def in_window(bank_date: str, ledger_date: str, window=DATE_WINDOW) -> bool:
    return window[0] <= day(bank_date) - day(ledger_date) <= window[1]


def subset_sums(target: int, items: list[tuple[str, int]], max_k: int = 4, cap: int = 200) -> list[tuple[str, ...]]:
    """All subsets of size 1..max_k of `items` (id, amount) summing exactly to target, smallest first."""
    found: list[tuple[str, ...]] = []
    n = len(items)
    ids = [i for i, _ in items]
    amts = [a for _, a in items]
    by_amt = defaultdict(list)
    for i, a in enumerate(amts):
        by_amt[a].append(i)

    for i in by_amt.get(target, []):
        found.append((ids[i],))
    if max_k >= 2:
        for i in range(n):
            for j in by_amt.get(target - amts[i], []):
                if j > i:
                    found.append((ids[i], ids[j]))
    if max_k >= 3 and len(found) < cap:
        for i in range(n):
            rest = target - amts[i]
            for j in range(i + 1, n):
                for k in by_amt.get(rest - amts[j], []):
                    if k > j:
                        found.append((ids[i], ids[j], ids[k]))
            if len(found) >= cap:
                break
    if max_k >= 4 and len(found) < cap:
        pairs = defaultdict(list)
        for i, j in combinations(range(n), 2):
            pairs[amts[i] + amts[j]].append((i, j))
        seen = set()
        for s, plist in pairs.items():
            other = pairs.get(target - s)
            if not other:
                continue
            for (i, j) in plist:
                for (k, l) in other:
                    if j < k:  # i<j<k<l gives each 4-set exactly once
                        key = (ids[i], ids[j], ids[k], ids[l])
                        if key not in seen:
                            seen.add(key)
                            found.append(key)
                if len(found) >= cap:
                    break
            if len(found) >= cap:
                break
    return found[:cap]


def solve(m: Month, max_k: int = 4) -> list[dict]:
    """Baseline S: greedy exact matching, then subset sums, then partial-payment sets."""
    lb = m.ledger_by_id()
    used_b, used_l = set(), set()
    groups = []

    def spread(bid: str, lids) -> int:
        bd = day(m.bank_by_id()[bid].date)
        return max(abs(bd - day(lb[l].date)) for l in lids)

    # 1. one-to-one exact amounts, most constrained bank line first, closest date wins.
    cands = {}
    for b in m.bank:
        cands[b.id] = [l for l in m.ledger if l.amount == b.amount and in_window(b.date, l.date)]
    for b in sorted(m.bank, key=lambda b: (len(cands[b.id]) or 99, b.id)):
        free = [l for l in cands[b.id] if l.id not in used_l]
        if not free:
            continue
        l = min(free, key=lambda l: (abs(day(b.date) - day(l.date)), l.id))
        groups.append({"bank": [b.id], "ledger": [l.id]})
        used_b.add(b.id)
        used_l.add(l.id)

    # 2. one bank line to 2..max_k ledger entries (covers one-to-many and fee-netted).
    for b in m.bank:
        if b.id in used_b:
            continue
        pool = [(l.id, l.amount) for l in m.ledger if l.id not in used_l and in_window(b.date, l.date)]
        subsets = [s for s in subset_sums(b.amount, pool, max_k) if len(s) >= 2]
        if not subsets:
            continue
        best = min(subsets, key=lambda s: (len(s), spread(b.id, s), s))
        groups.append({"bank": [b.id], "ledger": list(best)})
        used_b.add(b.id)
        used_l.update(best)

    # 3. 2..3 bank lines to one ledger entry (partial payments).
    for l in m.ledger:
        if l.id in used_l:
            continue
        pool = [(b.id, b.amount) for b in m.bank if b.id not in used_b
                and in_window(b.date, l.date, PARTIAL_WINDOW) and abs(b.amount) < abs(l.amount)
                and (b.amount > 0) == (l.amount > 0)]
        subsets = [s for s in subset_sums(l.amount, pool, 3) if len(s) >= 2]
        if not subsets:
            continue
        best = min(subsets, key=lambda s: (len(s), s))
        groups.append({"bank": list(best), "ledger": [l.id]})
        used_l.add(l.id)
        used_b.update(best)
    return groups
