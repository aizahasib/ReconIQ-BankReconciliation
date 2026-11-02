"""Core records for one reconciliation month, plus JSON I/O."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BankLine:
    id: str
    date: str  # ISO date
    amount: int  # cents, signed: deposits positive, withdrawals negative
    desc: str


@dataclass
class LedgerEntry:
    id: str
    date: str
    amount: int
    memo: str
    vendor: str
    account: str


@dataclass
class Month:
    id: str
    size: int
    seed: int
    bank: list[BankLine]
    ledger: list[LedgerEntry]
    # Each gold group: {"type": str, "bank": [ids], "ledger": [ids]}.
    gold: list[dict]
    params: dict = field(default_factory=dict)

    def bank_by_id(self) -> dict[str, BankLine]:
        return {b.id: b for b in self.bank}

    def ledger_by_id(self) -> dict[str, LedgerEntry]:
        return {l.id: l for l in self.ledger}

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Month":
        return Month(
            id=d["id"], size=d["size"], seed=d["seed"],
            bank=[BankLine(**b) for b in d["bank"]],
            ledger=[LedgerEntry(**l) for l in d["ledger"]],
            gold=d["gold"], params=d.get("params", {}),
        )


def day(iso: str) -> int:
    return dt.date.fromisoformat(iso).toordinal()


def fmt_amount(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) / 100:,.2f}"


def save_months(months: list[Month], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for m in months:
            f.write(json.dumps(m.to_json()) + "\n")


def load_months(path: Path) -> list[Month]:
    with path.open() as f:
        return [Month.from_json(json.loads(line)) for line in f if line.strip()]
