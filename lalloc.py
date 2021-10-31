#!/usr/bin/python3

from typing import Tuple, Any, Dict, List, TextIO, Optional, Mapping, Union

from dataclasses import dataclass, field
from dateutil import relativedelta

import json
import subprocess
import logging
import datetime
import jinja2
import argparse
import decimal
import re

OneDay = datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
Cents = decimal.Decimal("0.01")


def get_income_template(name: str):
    with open(name + ".template.ledger") as f:
        return jinja2.Template(f.read())


def get_generated_template():
    with open("generated.template.ledger") as f:
        return jinja2.Template(f.read())


def flatten(a):
    return [leaf for sl in a for leaf in sl]


def quantize(d):
    return d.quantize(Cents, decimal.ROUND_HALF_UP)


def datetime_today_that_is_sane() -> datetime.datetime:
    return datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time()
    ) - relativedelta.relativedelta(days=0)


def fail():
    raise Exception("fail")


@dataclass
class Posting:
    account: str
    value: decimal.Decimal
    note: Optional[str] = None

    def ledger_value(self) -> str:
        return f"${self.value:.2f}"


@dataclass
class Transaction:
    date: datetime.datetime
    payee: str
    cleared: bool
    postings: List[Posting] = field(default_factory=list)

    def append(self, p: Posting):
        self.postings.append(p)

    def ledger_date(self) -> str:
        actual = self.date.time()
        if actual == datetime.datetime.min.time():
            return self.date.strftime("%Y/%m/%d")
        return self.date.strftime("%Y/%m/%d %H:%M:%S")


class Handler:
    def expand(self, tx: Transaction, posting: Posting):
        return []


@dataclass
class HandlerPath:
    path: str
    handler: Handler
    compiled: Optional[re.Pattern] = None

    def matches(self, path: str) -> bool:
        if not self.compiled:
            self.compiled = re.compile(self.path)
        return self.compiled.match(path) is not None


@dataclass
class Transactions:
    txs: List[Transaction]

    def txns(self):
        return self.txs

    def apply_handlers(self, paths: List[HandlerPath]) -> List[Any]:
        cache: Dict[str, Handler] = {}
        returning: List[Any] = []

        for tx in self.txs:
            for p in tx.postings:
                account = p.account
                if account not in cache:
                    cache[account] = NoopHandler()
                    for hp in paths:
                        if hp.matches(account):
                            log.info(
                                f"{tx.date.date()} handler {account} ({hp.path}): {hp.handler}"
                            )
                            cache[account] = hp.handler
                            break

                for p in cache[account].expand(tx, p):
                    returning.append(p)

        return returning


@dataclass
class Ledger:
    path: str

    def register(self, expression: List[str]) -> Transactions:
        command = [
            "ledger",
            "-f",
            self.path,
            "-S",
            "date",
            "-F",
            "1|%D|%A|%t|%X|%P\n%/0|%D|%A|%t|%X|%P\n",
            "register",
        ] + expression
        sp = subprocess.run(command, stdout=subprocess.PIPE)

        log.info(" ".join(command).replace("\n", "NL"))

        txs: List[Transaction] = []
        tx: Union[Transaction, None] = None

        for line in sp.stdout.strip().decode("utf-8").split("\n"):
            fields = line.split("|")
            if len(fields) != 6:
                continue

            first, date_string, account, value_string, cleared, payee = fields
            date = datetime.datetime.strptime(date_string, "%Y/%m/%d")
            value = decimal.Decimal(value_string.replace("$", ""))

            if int(first) == 1:
                tx = Transaction(date, payee, len(cleared) > 0)
                txs.append(tx)
            assert tx
            tx.append(Posting(account, value, None))

        return Transactions(txs)


@dataclass
class IncomeDefinition:
    name: str
    epoch: datetime.datetime
    factor: decimal.Decimal


class Move:
    def txns(self):
        return []


@dataclass
class SimpleMove(Move):
    date: datetime.datetime
    value: decimal.Decimal
    from_path: str
    to_path: str
    payee: str

    def txns(self):
        tx = Transaction(self.date, self.payee, True)
        tx.append(Posting("[" + self.from_path + "]", -self.value, ""))
        tx.append(Posting("[" + self.to_path + "]", self.value, ""))
        return [tx]


@dataclass
class DatedMoney:
    note: str
    path: str
    date: datetime.datetime
    total: decimal.Decimal
    taken: decimal.Decimal = decimal.Decimal(0)
    where: Dict[str, decimal.Decimal] = field(default_factory=dict)

    def left(self) -> decimal.Decimal:
        return self.total - self.taken

    def take(
        self, money: "DatedMoney", partial=False
    ) -> Tuple["DatedMoney", List[Move]]:
        assert money.total >= 0
        assert quantize(money.total) == money.total
        taking = money.total
        if partial:
            left = self.left()
            if taking > left:
                taking = left
        else:
            assert self.taken + taking <= self.total
        log.debug(
            f"{money.date.date()} {money.path:50} {money.total:10} take total={self.total:8} taken={self.taken:8} taking={taking:8} after={self.total - self.taken - taking:8} {self.path}"
        )
        self.taken += taking
        self.where[money.path] = taking
        leftover = DatedMoney(
            note=money.note,
            path=money.path,
            date=money.date,
            total=money.total - taking,
            where=money.where,
        )
        return (
            leftover,
            [
                SimpleMove(
                    money.date, taking, self.path, money.path, f"payback '{money.note}'"
                )
            ],
        )

    def move(
        self, path: str, note: str, date: Optional[datetime.datetime] = None
    ) -> Tuple[Optional["DatedMoney"], List[Move]]:
        left = self.left()
        if left == 0:
            return (None, [])
        self.taken = self.total
        effective_date = date if date else self.date
        moved = DatedMoney(note=note, path=path, date=effective_date, total=left)
        return (moved, [SimpleMove(effective_date, left, self.path, path, note)])


def move_all_dated_money(
    dms: List[DatedMoney],
    note: str,
    path: str,
    date: Optional[datetime.datetime] = None,
) -> Tuple[List[DatedMoney], List[Move]]:
    tuples = [dm.move(path, note, date) for dm in dms]
    new_money = [dm for dm, m in tuples if dm]
    moves = flatten([m for dm, m in tuples])
    return (new_money, moves)


@dataclass
class MoneyPool:
    money: List[DatedMoney] = field(default_factory=list)

    def add(
        self, note: str, date: datetime.datetime, total: decimal.Decimal, path: str
    ):
        self.money.append(DatedMoney(note=note, date=date, total=total, path=path))

    def include(self, more: List[DatedMoney]):
        self.money += more
        self.money.sort(key=lambda p: (p.date, p.total))

    def take(self, taking: DatedMoney) -> List[Move]:
        available = [
            dm for dm in self.money if dm.left() > 0 and dm.date <= taking.date
        ]
        moves: List[Move] = []

        total_available = sum([dm.left() for dm in available])
        if total_available < taking.total:
            log.info(
                f"{taking.date.date()} {taking.path:50} {taking.total:10} insufficient available {total_available}"
            )
            return []

        for dm in available:
            taking, step = dm.take(taking, partial=True)
            if step:
                moves += step
            if taking.total == 0:
                break

        return moves


@dataclass
class Period(DatedMoney):
    income: IncomeDefinition = field(default_factory=fail)

    def after(self, spec: str) -> bool:
        testing = datetime.datetime.strptime(spec, "%m/%d/%Y")
        return self.date >= testing

    def before(self, spec: str) -> bool:
        testing = datetime.datetime.strptime(spec, "%m/%d/%Y")
        return self.date < testing

    def yearly(self, value: decimal.Decimal) -> str:
        v = quantize(
            value / decimal.Decimal(decimal.Decimal(12.0) * self.income.factor)
        )
        remaining, moves = self.take(
            DatedMoney(
                note="yearly expense", date=self.date, total=v, path=self.income.name
            )
        )
        assert remaining.total == 0
        return f"{v:.2f}"

    def monthly(self, value: decimal.Decimal) -> str:
        v = quantize(value / self.income.factor)
        remaining, moves = self.take(
            DatedMoney(
                note="monthly expense", date=self.date, total=v, path=self.income.name
            )
        )
        assert remaining.total == 0
        return f"{v:.2f}"

    def done(self) -> str:
        l = self.left()
        if l < 0.0:
            raise Exception(f"overallocated ({l})")
        return f"{l:.2f}"

    def display(self) -> str:
        return f"{self.left():.2f}"

    def __str__(self) -> str:
        return self.date.strftime("%Y/%m/%d")

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Expense:
    date: datetime.datetime
    value: decimal.Decimal
    note: str
    path: str


@dataclass
class Payment:
    date: datetime.datetime
    value: decimal.Decimal
    note: str
    path: str


@dataclass
class Income:
    date: datetime.datetime
    value: decimal.Decimal
    income: str
    note: str


@dataclass
class Spending:
    payments: List[Payment]
    paid: List[Payment] = field(default_factory=list)

    def pay_from(self, date: datetime.datetime, money: MoneyPool) -> List[Move]:
        log.info(f"{date.date()} {'':50} {'':10} pay-from")

        paying: List[Payment] = []
        for payment in self.payments:
            if payment.date < date:
                paying.append(payment)

        moves: List[Move] = []
        for p in paying:
            step = money.take(
                DatedMoney(date=date, total=p.value, note=p.note, path=p.path)
            )
            if step:
                moves += step
                self.payments.remove(p)
                self.paid.append(p)
            else:
                break
        return moves

    def _sort(self):
        self.expenses.sort(key=lambda p: (p.date, p.value))


@dataclass
class Schedule(Handler):
    maximum: Optional[decimal.Decimal] = None

    def expand(self, tx: Transaction, posting: Posting):
        if posting.value > 0:
            return []

        expense = Expense(
            date=tx.date,
            value=posting.value.copy_abs(),
            path=posting.account,
            note=tx.payee,
        )

        if self.maximum:
            payments: List[Payment] = []
            remaining = expense.value
            date = expense.date
            while remaining > 0:
                taking = self.maximum if remaining > self.maximum else remaining
                log.debug(f"schedule: date={date} {taking} note={expense.note}")
                payments.append(
                    Payment(
                        date=date, value=taking, path=expense.path, note=expense.note
                    )
                )
                remaining -= taking
                date += relativedelta.relativedelta(weeks=2)
            return payments

        log.debug(
            f"{expense.date.date()} schedule: {expense.value:10} note='{expense.note}'"
        )
        return [
            Payment(
                date=expense.date,
                value=expense.value,
                path=expense.path,
                note=expense.note,
            )
        ]


@dataclass
class NoopHandler(Handler):
    def expand(self, tx: Transaction, posting: Posting):
        return []


@dataclass
class IncomeHandler(Handler):
    income: IncomeDefinition
    path: str

    def expand(self, tx: Transaction, posting: Posting):
        log.info(f"{tx.date.date()} income: {posting} {posting.value}")
        return [
            Period(
                note=tx.payee,
                path=self.path,
                date=tx.date,
                total=posting.value.copy_abs(),
                income=self.income,
            )
        ]


@dataclass
class DatedMoneyHandler(Handler):
    positive_only = True

    def expand(self, tx: Transaction, posting: Posting):
        if self.positive_only:
            if posting.value < 0:
                return []
        log.info(f"{tx.date.date()} dated-money: {posting} {posting.value}")
        return [
            DatedMoney(
                note=tx.payee,
                path=posting.account,
                date=tx.date,
                total=posting.value.copy_abs(),
            )
        ]


@dataclass
class Names:
    available: str = "allocations:checking:available"
    refunded: str = "allocations:checking:refunded"
    emergency: str = "allocations:checking:savings:emergency"
    reserved: str = "assets:checking:reserved"


@dataclass
class Configuration:
    ledger_file: str
    names: Names
    incomes: List[HandlerPath]
    spending: List[HandlerPath]
    emergency: List[HandlerPath]
    refund: List[HandlerPath]
    income_pattern = "^income:"
    allocation_pattern = "^allocations:"


@dataclass
class Finances:
    cfg: Configuration
    today: datetime.datetime

    def allocate(self, file: TextIO):
        l = Ledger(self.cfg.ledger_file)

        names = self.cfg.names

        default_args = ["-S", "date", "--current"]
        exclude_allocations = ["and", "not", "tag(allocation)"]
        emergency_transactions = l.register(default_args + [names.emergency])
        income_transactions = l.register(
            default_args + [self.cfg.income_pattern] + exclude_allocations
        )
        allocation_transactions = l.register(
            default_args + [self.cfg.allocation_pattern] + exclude_allocations
        )

        income_periods = income_transactions.apply_handlers(self.cfg.incomes)
        emergency_money = emergency_transactions.apply_handlers(self.cfg.emergency)
        refund_money = allocation_transactions.apply_handlers(self.cfg.refund)
        spending = Spending(allocation_transactions.apply_handlers(self.cfg.spending))

        ytd_spending = sum([e.value for e in spending.payments])

        # Allocate static income by rendering income template for each income
        # transaction, this will generate transactions directly into file and
        # return the left over income we can apply to spending.
        available = MoneyPool()
        for period in income_periods:
            date = period.date
            template = get_income_template(period.income.name)
            rendered = template.render(period=period)

            log.info(f"{period} {period.note:50} {period.left():10} allocated")

            file.write(rendered)
            file.write("\n\n")

            available.add(names.reserved, date, period.left(), names.reserved)

        # This is how much money is left over to cover other expenses.
        income_after_static = sum([dm.left() for dm in available.money])

        moves: List[Move] = []

        # This moves refunded money in allocations to a single refund account
        # and then makes that money available for payback.
        refunded_money_moved, move_refunded = move_all_dated_money(
            refund_money, "refunded", names.refunded
        )
        moves += move_refunded
        available.include(refunded_money_moved)

        # Payback for spending for each pay period.
        for period in income_periods:
            moves += spending.pay_from(period.date, available)

        # How much is still unpaid for, this is spending accumulating until the
        # next pay period.
        unpaid_spending = sum([e.value for e in spending.payments])
        log.info(f"{self.today.date()} unpaid-spending: {unpaid_spending}")
        for payment in spending.payments:
            log.info(f"{self.today.date()} {payment}")
        moves += spending.pay_from(self.today, available)

        income_after_payback = sum([dm.left() for dm in available.money])
        for money in available.money:
            left = money.left()
            if left > 0:
                log.info(f"{money.date.date()} {money.path:50} {left:10}")

        # Move any left over income to the available account.
        _, moving = move_all_dated_money(
            available.money, "reserving left over income", names.available, self.today
        )
        moves += moving

        log.info(f"{self.today.date()} income-after-static: {income_after_static}")
        log.info(f"{self.today.date()} income-after-payback: {income_after_payback}")
        log.info(f"{self.today.date()} ytd-spending: {ytd_spending}")

        return flatten([m.txns() for m in moves])


def try_truncate_file(fn: str):
    try:
        with open(fn, "w+") as f:
            f.seek(0)
            f.truncate()
    except:
        pass


def parse_names(**kwargs) -> Names:
    return Names(**kwargs)


def parse_income(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    assert handler
    return HandlerPath(
        path,
        IncomeHandler(
            IncomeDefinition(
                handler["income"]["name"],
                datetime.datetime.fromtimestamp(int(handler["income"]["epoch"])),
                decimal.Decimal(handler["income"]["factor"]),
            ),
            handler["path"],
        ),
    )


def parse_spending_handler(maximum: Optional[str] = None) -> Handler:
    if maximum:
        return Schedule(maximum=decimal.Decimal(maximum))
    return Schedule()


def parse_spending(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    if handler:
        return HandlerPath(path, parse_spending_handler(**handler))
    return HandlerPath(path, Schedule())


def parse_petty(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    return HandlerPath(path, DatedMoneyHandler())


def parse_emergency(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    return HandlerPath(path, DatedMoneyHandler())


def parse_refund(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    return HandlerPath(path, DatedMoneyHandler())


def parse_configuration(
    ledger_file: Optional[str] = None,
    names: Mapping[str, Any] = None,
    income: List[Mapping[str, Any]] = None,
    spending: List[Mapping[str, Any]] = None,
    refund: List[Mapping[str, Any]] = None,
    emergency: List[Mapping[str, Any]] = None,
    **kwargs,
) -> Configuration:
    assert ledger_file
    assert income
    assert spending
    assert refund
    assert emergency
    assert names
    return Configuration(
        ledger_file=ledger_file,
        names=parse_names(**names),
        incomes=[parse_income(**obj) for obj in income],
        spending=[parse_spending(**obj) for obj in spending],
        emergency=[parse_emergency(**obj) for obj in emergency],
        refund=[parse_refund(**obj) for obj in refund],
    )


def allocate(config_path: str, file_name: str, today: datetime.datetime) -> None:
    with open(config_path, "r") as file:
        raw = json.loads(file.read())
        configuration = parse_configuration(**raw)

    f = Finances(configuration, today)

    try_truncate_file(file_name)

    with open(file_name, "w") as file:
        txs = f.allocate(file)
        t = get_generated_template()
        rendered = t.render(txs=txs)
        file.write(rendered)
        file.write("\n\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)5s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    log = logging.getLogger("lalloc")

    parser = argparse.ArgumentParser(description="ledger allocations tool")
    parser.add_argument("-c", "--config-file", action="store", default="lalloc.json")
    parser.add_argument(
        "-l", "--ledger-file", action="store", default="lalloc.g.ledger"
    )
    parser.add_argument("-a", "--allocate", action="store_true", default=False)
    parser.add_argument("-t", "--today", action="store", default=None)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--no-debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.allocate:
        today = datetime_today_that_is_sane()
        if args.today:
            today = datetime.datetime.strptime(args.today, "%Y/%m/%d")
            log.warning(f"today overriden to {today}")

        allocate(args.config_file, args.ledger_file, today)
