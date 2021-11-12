#!/usr/bin/python3

from typing import Tuple, Any, Dict, Sequence, List, TextIO, Optional, Mapping, Union

from dataclasses import dataclass, field
from dateutil import relativedelta
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, date

import json
import sys
import subprocess
import logging
import jinja2
import argparse
import re

OneDay = timedelta(days=1) - timedelta(seconds=1)
Cents = Decimal("0.01")


def get_generated_template():
    with open("generated.template.ledger") as f:
        return jinja2.Template(f.read())


def flatten(a):
    return [leaf for sl in a for leaf in sl]


def quantize(d):
    return d.quantize(Cents, ROUND_HALF_UP)


def datetime_today_that_is_sane() -> datetime:
    return datetime.combine(date.today(), datetime.min.time())


@dataclass
class Names:
    available: str = "allocations:checking:available"
    refunded: str = "allocations:checking:refunded"
    emergency: str = "allocations:checking:savings:emergency"
    taxes: str = "allocations:checking:savings:main"
    reserved: str = "assets:checking:reserved"


class Rule:
    def apply(self, date: datetime, balances: "Balances") -> List["Transaction"]:
        raise NotImplementedError


@dataclass
class MoveSpec:
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class Configuration:
    ledger_file: str
    names: Names
    rules: List[Rule] = field(default_factory=list)


@dataclass
class Balance:
    date: datetime
    account: str
    value: Decimal


@dataclass
class Balances:
    balances: List[Balance]


@dataclass
class Ledger:
    path: str

    def balances(self, expression: List[str]) -> Balances:
        command = [
            "ledger",
            "-f",
            self.path,
            "-F",
            "%D|%A|%(display_total)\n",
            "balance",
            "--no-total",
            "--flat",
        ] + expression
        sp = subprocess.run(command, stdout=subprocess.PIPE)

        log.info(" ".join(command).replace("\n", "NL"))

        balances: List[Balance] = []

        for line in sp.stdout.strip().decode("utf-8").split("\n"):
            fields = line.split("|")
            if len(fields) != 3:
                continue

            date_string, account, value_string = fields
            date = datetime.strptime(date_string, "%Y/%m/%d")
            value = Decimal(value_string.replace("$", ""))

            balances.append(Balance(date, account, value))

        return Balances(balances)


@dataclass
class Posting:
    account: str
    value: Decimal
    note: Optional[str] = None

    def ledger_value(self) -> str:
        return f"${self.value:.2f}"


@dataclass
class Transaction:
    date: datetime
    payee: str
    cleared: bool
    postings: List[Posting] = field(default_factory=list)

    def ledger_date(self) -> str:
        actual = self.date.time()
        if actual == datetime.min.time():
            return self.date.strftime("%Y/%m/%d")
        return self.date.strftime("%Y/%m/%d %H:%M:%S")

    def append(self, p: Posting):
        self.postings.append(p)


@dataclass
class MaximumRule(Rule):
    path: str
    moves: List[MoveSpec]
    maximum: Decimal
    compiled: Optional[re.Pattern] = None

    def apply_matching(self, date: datetime, balances: Balances) -> List[Transaction]:
        raise NotImplementedError

    def apply(self, date: datetime, balances: Balances) -> List[Transaction]:
        if self.compiled is None:
            self.compiled = re.compile(self.path)

        matching: List[Balance] = []

        for balance in balances.balances:
            if self.compiled.match(balance.account):
                log.info(f"match {balance.account} ({self.path})")
                matching.append(balance)

        return self.apply_matching(date, Balances(matching))

    def move_value(
        self, date: datetime, value: Decimal, note: str
    ) -> List[Transaction]:
        tx = Transaction(date, note, False)
        for move in self.moves:
            for from_path, to_path in move.paths.items():
                tx.append(Posting("[" + from_path + "]", -value, ""))
                tx.append(Posting("[" + to_path + "]", value, ""))
        return [tx]


@dataclass
class MaximumBalanceRule(MaximumRule):
    def apply_matching(self, date: datetime, balances: Balances) -> List[Transaction]:
        return flatten(
            [
                self.move_value(date, self.maximum, "maximum reached")
                for b in balances.balances
                if b.value > self.maximum
            ]
        )


@dataclass
class ExcessBalanceRule(MaximumRule):
    def apply_matching(self, date: datetime, balances: Balances) -> List[Transaction]:
        return flatten(
            [
                self.move_value(date, b.value - self.maximum, "moving excess")
                for b in balances.balances
                if b.value > self.maximum
            ]
        )


@dataclass
class RelativeBalance:
    balance: Balance
    percentage: Decimal


@dataclass
class DistributedBalancedRule(Rule):
    path: str
    compiled: Optional[re.Pattern] = None

    def apply(self, date: datetime, balances: Balances) -> List[Transaction]:
        if self.compiled is None:
            self.compiled = re.compile(self.path)

        narrowed = [b for b in balances.balances if self.compiled.match(b.account)]
        total = sum([b.value for b in narrowed])

        relative = [
            RelativeBalance(b, b.value / total * Decimal(100)) for b in narrowed
        ]
        relative.sort(key=lambda r: -r.percentage)

        for r in relative:
            log.info(
                f"{date.date()} {r.balance.account:50} {r.balance.value:10} {r.percentage:10.2f}%"
            )

        return []


@dataclass
class OverdraftProtection(Rule):
    path: str
    overdraft: str
    compiled: Optional[re.Pattern] = None

    def apply(self, date: datetime, balances: Balances) -> List[Transaction]:
        if self.compiled is None:
            self.compiled = re.compile(self.path)

        danger = [
            b
            for b in balances.balances
            if self.compiled.match(b.account) and b.value < 0
        ]

        return [self._cover(date, balance) for balance in danger]

    def _cover(self, date: datetime, balance: Balance) -> Transaction:
        excess = -balance.value
        log.info(f"covering {balance}")
        tx = Transaction(date, "overdraft protection", True)
        tx.append(Posting("[" + self.overdraft + "]", -excess, ""))
        tx.append(Posting("[" + balance.account + "]", excess, ""))
        return tx


@dataclass
class Finances:
    cfg: Configuration
    today: datetime

    def rebalance(self, file: TextIO) -> List[Transaction]:
        l = Ledger(self.cfg.ledger_file)
        balances = l.balances(["--exchange", "$"])

        txs: List[Transaction] = []

        for rule in self.cfg.rules:
            log.info(f"applying {rule}")
            txs += rule.apply(self.today, balances)

        return txs


def parse_names(**kwargs) -> Names:
    return Names(**kwargs)


def parse_move_spec(**kwargs) -> MoveSpec:
    return MoveSpec(kwargs)


def parse_rule(
    maximum: Optional[Decimal] = None,
    excess: Optional[Decimal] = None,
    overdraft: Optional[str] = None,
    moves: Optional[List[Mapping[str, str]]] = None,
    **kwargs,
) -> Rule:
    if overdraft:
        return OverdraftProtection(overdraft=overdraft, **kwargs)
    if excess:
        assert moves
        return ExcessBalanceRule(
            maximum=excess, moves=[parse_move_spec(**move) for move in moves], **kwargs
        )
    if maximum:
        assert moves
        return MaximumBalanceRule(
            maximum=maximum, moves=[parse_move_spec(**move) for move in moves], **kwargs
        )
    return DistributedBalancedRule(**kwargs)


def parse_configuration(
    today: Optional[datetime] = None,
    ledger_file: Optional[str] = None,
    names: Mapping[str, Any] = None,
    rules: Optional[List[Mapping[str, Any]]] = None,
    **kwargs,
) -> Configuration:
    assert today
    assert ledger_file
    assert names
    assert rules
    return Configuration(
        ledger_file=ledger_file,
        names=parse_names(**names),
        rules=[parse_rule(**rule) for rule in rules],
    )


def try_truncate_file(fn: str):
    try:
        with open(fn, "w+") as f:
            f.seek(0)
            f.truncate()
    except:
        pass


def rebalance(config_path: str, file_name: str, today: datetime) -> None:
    with open(config_path, "r") as file:
        raw = json.loads(file.read())
        configuration = parse_configuration(today=today, **raw)

    try_truncate_file(file_name)

    f = Finances(configuration, today)
    txs = f.rebalance(file)

    t = get_generated_template()

    # Write cleared transactions to the generated file, these are to be included
    # automatically and are for virtual adjustments.
    with open(file_name, "w") as file:
        rendered = t.render(txs=[tx for tx in txs if tx.cleared])
        file.write(rendered)
        file.write("\n\n")

    # Anything uncleared needs human intervention to perform because they
    # require physical money to move around.
    manual = [tx for tx in txs if not tx.cleared]
    if manual:
        sys.stdout.write("\n\n")
        sys.stdout.write("; copy and paste the following when you schedule them")
        sys.stdout.write("\n\n")
        sys.stdout.write(t.render(txs=manual).strip())
        sys.stdout.write("\n\n")


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log_file = logging.FileHandler("rebal.log")
    log_file.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)7s] %(message)s",
        handlers=[console, log_file],
    )

    log = logging.getLogger("rebal")

    parser = argparse.ArgumentParser(description="ledger allocations tool")
    parser.add_argument("-c", "--config-file", action="store", default="rebal.json")
    parser.add_argument("-l", "--ledger-file", action="store", default="rebal.g.ledger")
    parser.add_argument("-t", "--today", action="store", default=None)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--no-debug", action="store_true", default=False)
    args = parser.parse_args()

    today = datetime_today_that_is_sane()

    if args.debug:
        console.setLevel(logging.DEBUG)

    if args.today:
        today = datetime.strptime(args.today, "%Y/%m/%d")
        log.warning(f"today overriden to {today}")

    rebalance(args.config_file, args.ledger_file, today)
