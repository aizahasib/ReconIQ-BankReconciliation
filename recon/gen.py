"""R0.1 minimal generator.

The ledger is minted first; bank lines are derived from it with controlled
perturbations, and the true matching is recorded as it is made, so gold is
exact by construction.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

from .data import BankLine, LedgerEntry, Month, day

# Default perturbation rates for Phase 0. Phase 1 parameterizes these fully.
DEFAULT_PARAMS = {
    "w_one_one": 0.56,
    "w_decoy_pair": 0.10,  # two 1:1 groups with identical amounts; only text separates them
    "w_one_many": 0.11,  # one bank line settles 2-4 ledger entries
    "w_many_one": 0.08,  # 2-3 partial bank payments settle one ledger entry
    "w_fee": 0.10,  # bank = gross ledger entry + negative fee ledger entry
    "unmatched_rate": 0.05,  # each side
    "max_lag": 5,
    "decoy_same_day": False,  # decoy siblings share a ledger date, so date cannot break the tie
    "decoy_max_k": 2,  # decoy sets of 2..k groups with one amount
}

# R0.3 branch: amounts were too informative under DEFAULT_PARAMS, so raise
# equal-amount decoys until at least 40% of bank lines need text to resolve.
HARD_PARAMS = {
    **DEFAULT_PARAMS,
    "w_one_one": 0.40,
    "w_decoy_pair": 0.28,
    "decoy_same_day": True,
    "decoy_max_k": 3,
}

# (name, bank token, expense account, low $, high $, fixed amount $ or None, method)
VENDORS = [
    ("Acme Office Supply", "ACMEOFFICE", "Office Supplies", 40, 900, None, "ach"),
    ("Brightwater Utilities", "BRIGHTWATER UTIL", "Utilities", 120, 1400, None, "ach"),
    ("Cobalt Cloud Hosting", "COBALTCLOUD", "Software", 80, 2400, None, "card"),
    ("Delta Freight Lines", "DELTA FREIGHT", "Freight", 150, 4800, None, "ach"),
    ("Evergreen Landscaping", "EVERGREEN LAND", "Repairs", 90, 1500, None, "check"),
    ("Fulton Property Mgmt", "FULTON PROP", "Rent", 0, 0, 4250.00, "ach"),
    ("Granite Insurance Co", "GRANITE INS", "Insurance", 0, 0, 612.40, "ach"),
    ("Harbor Telecom", "HARBOR TEL", "Telephone", 60, 420, None, "ach"),
    ("Ironclad Security", "IRONCLAD SEC", "Security", 75, 650, None, "card"),
    ("Juniper Legal LLP", "JUNIPER LEGAL", "Legal Fees", 400, 6500, None, "check"),
    ("Keystone Paper Co", "KEYSTONE PAPER", "Supplies", 35, 780, None, "card"),
    ("Lumen Electric", "LUMEN ELEC", "Utilities", 90, 1100, None, "ach"),
    ("Meridian Staffing", "MERIDIAN STAFF", "Contract Labor", 800, 9000, None, "ach"),
    ("Northwind Logistics", "NORTHWIND LOG", "Freight", 120, 3900, None, "ach"),
    ("Orchard Catering", "ORCHARD CATER", "Meals", 60, 1200, None, "card"),
    ("Pinnacle Software", "PINNACLE SW", "Software", 0, 0, 299.00, "card"),
    ("Quarry Stone Supply", "QUARRY STONE", "Materials", 200, 5200, None, "check"),
    ("Redwood Accounting", "REDWOOD ACCT", "Professional Fees", 250, 2200, None, "check"),
    ("Summit Fuel", "SUMMIT FUEL", "Fuel", 40, 600, None, "card"),
    ("Tidewater Marine", "TIDEWATER MAR", "Equipment", 300, 7800, None, "ach"),
    ("Umbra Design Studio", "UMBRA DESIGN", "Marketing", 150, 3000, None, "ach"),
    ("Vantage Ads", "VANTAGE ADS", "Advertising", 100, 2500, None, "card"),
    ("Westfield Parts", "WESTFIELD PRT", "Parts", 30, 1800, None, "card"),
    ("Yellowstone Tools", "YELLOWSTONE TL", "Tools", 25, 950, None, "card"),
    ("Zenith Payroll Svcs", "ZENITH PAYROLL", "Payroll Fees", 0, 0, 185.00, "ach"),
    ("Alder Janitorial", "ALDER JANITOR", "Cleaning", 120, 900, None, "check"),
    ("Beacon Printing", "BEACON PRINT", "Printing", 45, 1300, None, "card"),
    ("Crescent Water", "CRESCENT WATER", "Utilities", 25, 160, None, "card"),
]

CUSTOMERS = [
    "Brightline LLC", "Cedar & Co", "Dunmore Foods", "Elmwood Clinic", "Foxglove Retail",
    "Garnet Builders", "Hollis Dental", "Indigo Events", "Jasper Motors", "Kilnworks Pottery",
    "Larkspur Hotels", "Monarch Realty", "Nimbus Analytics", "Oakridge Schools", "Pepper & Pine",
    "Quill Publishing", "Rowan Farms", "Sable Fitness", "Thornbury Law", "Upland Brewing",
    "Verity Labs", "Willow Dental", "Xenon Robotics", "Yarrow Bakery", "Zephyr Travel",
]

PROCESSORS = [("Stripe", "STRIPE TRANSFER"), ("Square", "SQUARE INC DEPOSIT")]
CITIES = ["SEATTLE WA", "PORTLAND OR", "BOISE ID", "TACOMA WA", "SPOKANE WA"]


@dataclass
class _Draft:
    """A group before ids are assigned. Each item is (date ordinal, amount, text fields)."""
    type: str
    bank: list[tuple[int, int, str]]
    ledger: list[tuple[int, int, str, str, str]]  # date, amount, memo, vendor, account


class MonthBuilder:
    def __init__(self, size: int, seed: int, params: dict | None = None):
        self.size = size
        self.seed = seed
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.r = random.Random(seed)
        year = 2026
        mon = self.r.randint(1, 12)
        self.start = dt.date(year, mon, 1).toordinal()
        nxt = dt.date(year + (mon == 12), mon % 12 + 1, 1).toordinal()
        self.end = nxt - 1  # last day of month
        self.check_no = self.r.randint(1000, 8000)
        self.inv_no = self.r.randint(1000, 8000)
        self.used_amounts: set[int] = set()

    # ---- small helpers -------------------------------------------------------
    def _iso(self, d: int) -> str:
        return dt.date.fromordinal(d).isoformat()

    def _digits(self, n: int) -> str:
        return "".join(self.r.choice("0123456789") for _ in range(n))

    def _amount(self, lo: float, hi: float) -> int:
        # Mostly cents-level amounts, some round ones (round amounts collide more).
        if self.r.random() < 0.2:
            return int(round(self.r.uniform(lo, hi) / 50) * 50 * 100) or 5000
        return int(round(self.r.uniform(lo, hi) * 100))

    def _fresh_amount(self, lo: float, hi: float) -> int:
        for _ in range(50):
            a = self._amount(lo, hi)
            if a not in self.used_amounts:
                self.used_amounts.add(a)
                return a
        return a

    def _ledger_date(self) -> int:
        # Leave room for lag so the bank line lands inside the month.
        return self.r.randint(self.start, self.end - self.p["max_lag"])

    def _lag(self, method: str) -> int:
        if method == "check":
            return self.r.randint(2, self.p["max_lag"])
        return self.r.choice([0, 1, 1, 1, 2, 2, 3, 4, 5][: self.p["max_lag"] + 4])

    def _next_inv(self) -> str:
        self.inv_no += self.r.randint(1, 3)
        return f"INV-{self.inv_no}"

    # ---- text ---------------------------------------------------------------
    def _vendor_bank_desc(self, v, d: int, method: str, check: int | None) -> str:
        token = v[1]
        if method == "check":
            return f"CHECK {check}"
        if method == "card":
            mmdd = dt.date.fromordinal(d).strftime("%m%d")
            return self.r.choice([
                f"POS PURCHASE {token} {self.r.choice(CITIES)}",
                f"CHECKCARD {mmdd} {token} {self._digits(4)}",
                f"DEBIT CARD {token}*{self._digits(6)}",
            ])
        return self.r.choice([
            f"ACH DEBIT {token} {self._digits(6)}",
            f"{token} PAYMENT PPD ID:{self._digits(9)}",
            f"ONLINE PMT {token} CONF {self._digits(7)}",
        ])

    def _customer_bank_desc(self, cust: str, named: bool = True) -> str:
        up = cust.upper().replace("&", "AND")
        if not named:
            return self.r.choice([f"MOBILE DEPOSIT REF {self._digits(8)}", f"BRANCH DEPOSIT {self._digits(6)}"])
        return self.r.choice([
            f"DEPOSIT {up}",
            f"ACH CREDIT {up[:16]} PAYMENT",
            f"WIRE IN {up[:18]} REF {self._digits(6)}",
            f"REMOTE DEP {up[:14]}",
        ])

    # ---- group makers -------------------------------------------------------
    def _vendor_payment(self, amount: int | None = None, v=None, ldate: int | None = None) -> _Draft:
        v = v or self.r.choice(VENDORS)
        method = v[6]
        a = amount if amount is not None else (int(v[5] * 100) if v[5] else self._fresh_amount(v[3], v[4]))
        ld = ldate if ldate is not None else self._ledger_date()
        bd = ld + self._lag(method)
        if method == "check":
            self.check_no += 1
            memo = f"Check #{self.check_no} - {self._next_inv()}"
            desc = self._vendor_bank_desc(v, bd, method, self.check_no)
        else:
            memo = f"Bill pmt {self._next_inv()} {v[0]}"
            desc = self._vendor_bank_desc(v, bd, method, None)
        return _Draft("one_one", [(bd, -a, desc)], [(ld, -a, memo, v[0], v[2])])

    def _customer_receipt(self, amount: int | None = None, cust: str | None = None,
                          ldate: int | None = None, named: bool | None = None) -> _Draft:
        cust = cust or self.r.choice(CUSTOMERS)
        a = amount if amount is not None else self._fresh_amount(200, 8000)
        ld = ldate if ldate is not None else self._ledger_date()
        bd = ld + self.r.choice([0, 0, 1, 1, 2, 3])
        named = (self.r.random() > 0.15) if named is None else named
        memo = f"Pmt recd {self._next_inv()} {cust}"
        return _Draft("one_one", [(bd, a, self._customer_bank_desc(cust, named))],
                      [(ld, a, memo, cust, "Accounts Receivable")])

    def one_one(self) -> _Draft:
        return self._vendor_payment() if self.r.random() < 0.6 else self._customer_receipt()

    def decoy_pair(self, k: int = 2) -> list[_Draft]:
        """k groups with the same amount and close dates; only memo/vendor text tells them apart."""
        ld = self._ledger_date()

        def sib_date() -> int:
            if self.p["decoy_same_day"]:
                return ld
            return min(ld + self.r.randint(0, 3), self.end - self.p["max_lag"])

        if self.r.random() < 0.5:
            vs = self.r.sample([v for v in VENDORS if v[6] != "check"], k)
            a = self._fresh_amount(max(vs[0][3], 50), max(vs[0][4], 900))
            out = [self._vendor_payment(a, v, ld if i == 0 else sib_date()) for i, v in enumerate(vs)]
        else:
            cs = self.r.sample(CUSTOMERS, k)
            a = self._fresh_amount(300, 6000)
            out = [self._customer_receipt(a, c, ld if i == 0 else sib_date(), True) for i, c in enumerate(cs)]
        for d in out:
            d.type = "decoy"
        return out

    def one_many(self) -> _Draft:
        k = self.r.choice([2, 2, 3, 3, 4])
        ld = self._ledger_date()
        if self.r.random() < 0.6:
            cust = self.r.choice(CUSTOMERS)
            parts = [self._fresh_amount(150, 3500) for _ in range(k)]
            ledger = [(ld - self.r.randint(0, 1), a, f"Pmt recd {self._next_inv()} {cust}", cust, "Accounts Receivable")
                      for a in parts]
            bank = [(ld + self.r.choice([0, 1, 2]), sum(parts), self._customer_bank_desc(cust, True))]
        else:
            v = self.r.choice([v for v in VENDORS if v[5] is None and v[6] == "ach"])
            parts = [self._fresh_amount(v[3], v[4]) for _ in range(k)]
            ledger = [(ld, -a, f"Bill pmt {self._next_inv()} {v[0]}", v[0], v[2]) for a in parts]
            bank = [(ld + self._lag("ach"), -sum(parts), self._vendor_bank_desc(v, ld, "ach", None))]
        return _Draft("one_many", bank, ledger)

    def many_one(self) -> _Draft:
        k = self.r.choice([2, 2, 3])
        cust = self.r.choice(CUSTOMERS)
        total = self._fresh_amount(1200, 9000)
        # Split total into k positive partial payments.
        cuts = sorted(self.r.sample(range(1, total), k - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        ld = self.r.randint(self.start, self.end - 12)
        inv = self._next_inv()
        bank = []
        d = ld
        for a in parts:
            d = min(d + self.r.randint(0, 5), self.end)
            bank.append((d, a, self._customer_bank_desc(cust, self.r.random() > 0.2)))
        ledger = [(ld, total, f"Pmt recd {inv} {cust} (partial pmts)", cust, "Accounts Receivable")]
        return _Draft("many_one", bank, ledger)

    def fee_netted(self) -> _Draft:
        proc, token = self.r.choice(PROCESSORS)
        gross = self._fresh_amount(300, 12000)
        fee = int(round(gross * self.r.uniform(0.026, 0.031))) + 30
        ld = self._ledger_date()
        bd = ld + self.r.choice([1, 2, 2, 3])
        batch = self._digits(4)
        bank = [(bd, gross - fee, f"{token} {self._digits(10)}")]
        ledger = [
            (ld, gross, f"{proc} sales batch {batch}", proc, "Sales"),
            (ld, -fee, f"{proc} processing fees batch {batch}", proc, "Merchant Fees"),
        ]
        return _Draft("fee", bank, ledger)

    # ---- unmatched ----------------------------------------------------------
    def bank_only(self) -> tuple[int, int, str]:
        kind = self.r.random()
        if kind < 0.3:
            return (self.r.randint(self.start, self.end), -self.r.choice([1200, 1500, 2500, 3500]), "MONTHLY SERVICE FEE")
        if kind < 0.45:
            return (self.end, self.r.randint(5, 900), "INTEREST PAYMENT")
        if kind < 0.7:  # prior-month item clearing at the start of this month (timing difference)
            return (self.start + self.r.randint(0, 3), -self._fresh_amount(80, 2500),
                    f"CHECK {self.check_no - self.r.randint(20, 60)}")
        v = self.r.choice([v for v in VENDORS if v[6] == "card"])
        d = self.r.randint(self.start, self.end)
        return (d, -self._fresh_amount(15, 300), self._vendor_bank_desc(v, d, "card", None))

    def ledger_only(self) -> tuple[int, int, str, str, str]:
        if self.r.random() < 0.6:  # deposit in transit / outstanding check at month end (timing difference)
            ld = self.end - self.r.randint(0, 2)
            if self.r.random() < 0.5:
                cust = self.r.choice(CUSTOMERS)
                return (ld, self._fresh_amount(200, 6000), f"Pmt recd {self._next_inv()} {cust}", cust, "Accounts Receivable")
            v = self.r.choice(VENDORS)
            self.check_no += 1
            return (ld, -self._fresh_amount(100, 3000), f"Check #{self.check_no} - {self._next_inv()}", v[0], v[2])
        v = self.r.choice(VENDORS)  # outstanding check written mid-month that has not cleared
        self.check_no += 1
        return (self._ledger_date(), -self._fresh_amount(100, 3000),
                f"Check #{self.check_no} - {self._next_inv()}", v[0], v[2])

    # ---- assembly -----------------------------------------------------------
    def build(self, month_id: str) -> Month:
        p = self.p
        n_unm_bank = max(1, round(self.size * p["unmatched_rate"]))
        target = self.size - n_unm_bank
        drafts: list[_Draft] = []
        nb = 0
        kinds = ["one_one", "decoy_pair", "one_many", "many_one", "fee"]
        weights = [p["w_one_one"], p["w_decoy_pair"], p["w_one_many"], p["w_many_one"], p["w_fee"]]
        while nb < target:
            room = target - nb
            kind = self.r.choices(kinds, weights)[0]
            if kind == "decoy_pair" and room >= 2:
                new = self.decoy_pair(self.r.randint(2, min(p["decoy_max_k"], room)))
            elif kind == "many_one" and room >= 3:
                new = [self.many_one()]
            elif kind == "one_many":
                new = [self.one_many()]
            elif kind == "fee":
                new = [self.fee_netted()]
            else:
                new = [self.one_one()]
            drafts.extend(new)
            nb += sum(len(d.bank) for d in new)

        n_ledger_matched = sum(len(d.ledger) for d in drafts)
        n_unm_ledger = max(1, round(n_ledger_matched * p["unmatched_rate"] / (1 - p["unmatched_rate"])))
        bank_only = [self.bank_only() for _ in range(n_unm_bank)]
        ledger_only = [self.ledger_only() for _ in range(n_unm_ledger)]

        # Assign ids in date order (random tie-break), bank and ledger separately.
        bank_items = [(b, gi) for gi, d in enumerate(drafts) for b in d.bank] + [(b, None) for b in bank_only]
        ledger_items = [(l, gi) for gi, d in enumerate(drafts) for l in d.ledger] + [(l, None) for l in ledger_only]
        self.r.shuffle(bank_items)
        self.r.shuffle(ledger_items)
        bank_items.sort(key=lambda x: x[0][0])
        ledger_items.sort(key=lambda x: x[0][0])

        groups = [{"type": d.type, "bank": [], "ledger": []} for d in drafts]
        bank, ledger = [], []
        for i, ((d, a, desc), gi) in enumerate(bank_items, 1):
            bid = f"B{i:03d}"
            bank.append(BankLine(bid, self._iso(d), a, desc))
            if gi is not None:
                groups[gi]["bank"].append(bid)
        for i, ((d, a, memo, vendor, acct), gi) in enumerate(ledger_items, 1):
            lid = f"L{i:03d}"
            ledger.append(LedgerEntry(lid, self._iso(d), a, memo, vendor, acct))
            if gi is not None:
                groups[gi]["ledger"].append(lid)
        return Month(month_id, self.size, self.seed, bank, ledger, groups, dict(p))


def mint(sizes: list[int], per_size: int, base_seed: int, params: dict | None = None) -> list[Month]:
    months = []
    for size in sizes:
        for k in range(per_size):
            seed = base_seed + size * 1000 + k
            months.append(MonthBuilder(size, seed, params).build(f"m{size}_{k:02d}"))
    return months


# ---- R0.1 self-check ----------------------------------------------------------
DATE_WINDOW = (-2, 7)  # bank date minus ledger date, in days, used by the filter and checks


def self_check(m: Month) -> list[str]:
    """Return a list of problems; empty means the month's gold is internally consistent."""
    errs = []
    bb, lb = m.bank_by_id(), m.ledger_by_id()
    if len(m.bank) != m.size:
        errs.append(f"bank count {len(m.bank)} != size {m.size}")
    seen_b, seen_l = set(), set()
    for g in m.gold:
        sb = sum(bb[b].amount for b in g["bank"])
        sl = sum(lb[l].amount for l in g["ledger"])
        if sb != sl:
            errs.append(f"group {g} sums bank {sb} != ledger {sl}")
        if not g["bank"] or not g["ledger"]:
            errs.append(f"empty side in {g}")
        for b in g["bank"]:
            if b in seen_b:
                errs.append(f"bank {b} in two groups")
            seen_b.add(b)
            for l in g["ledger"]:
                lag = day(bb[b].date) - day(lb[l].date)
                if g["type"] != "many_one" and not (DATE_WINDOW[0] <= lag <= DATE_WINDOW[1]):
                    errs.append(f"{b}/{l} lag {lag} outside window")
        for l in g["ledger"]:
            if l in seen_l:
                errs.append(f"ledger {l} in two groups")
            seen_l.add(l)
    first = min(day(b.date) for b in m.bank)
    last = max(day(b.date) for b in m.bank)
    if last - first > 31:
        errs.append("bank dates span more than a month")
    return errs


def amount_unresolvable(m: Month) -> tuple[int, int]:
    """Count bank lines an exact one-to-one amount match alone cannot resolve.

    A bank line is amount-resolvable only if its gold group is 1:1 and its amount
    is unique among in-window ledger entries and among bank lines near that entry;
    or if it is gold-unmatched and no in-window ledger entry has its amount.
    """
    lb = m.ledger_by_id()
    group_of = {b: g for g in m.gold for b in g["bank"]}
    unres = 0
    for b in m.bank:
        bd = day(b.date)
        same_amt_ledger = [l for l in m.ledger if l.amount == b.amount
                           and DATE_WINDOW[0] <= bd - day(l.date) <= DATE_WINDOW[1]]
        g = group_of.get(b.id)
        if g is None:
            resolvable = not same_amt_ledger
        elif len(g["bank"]) == 1 and len(g["ledger"]) == 1:
            l = lb[g["ledger"][0]]
            same_amt_bank = [x for x in m.bank if x.amount == b.amount
                             and DATE_WINDOW[0] <= day(x.date) - day(l.date) <= DATE_WINDOW[1]]
            resolvable = len(same_amt_ledger) == 1 and len(same_amt_bank) == 1
        else:
            resolvable = False
        unres += not resolvable
    return unres, len(m.bank)
