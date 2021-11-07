#!/usr/bin/python3

from typing import Tuple, Any, Dict, Sequence, List, TextIO, Optional, Mapping, Union

from dataclasses import dataclass, field
from dateutil import relativedelta
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, date

import json
import subprocess
import logging
import jinja2
import argparse
import re

OneDay = timedelta(days=1) - timedelta(seconds=1)
Cents = Decimal("0.01")


def get_income_template(name: str):
    with open(name + ".template.ledger") as f:
        return jinja2.Template(f.read())


def get_generated_template():
    with open("generated.template.ledger") as f:
        return jinja2.Template(f.read())


def flatten(a):
    return [leaf for sl in a for leaf in sl]


def quantize(d):
    return d.quantize(Cents, ROUND_HALF_UP)


def datetime_today_that_is_sane() -> datetime:
    return datetime.combine(date.today(), datetime.min.time())


def fail():
    raise Exception("fail")


@dataclass
class Names:
    available: str = "allocations:checking:available"
    refunded: str = "allocations:checking:refunded"
    emergency: str = "allocations:checking:savings:emergency"
    taxes: str = "allocations:checking:savings:main"
    reserved: str = "assets:checking:reserved"


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

    def append(self, p: Posting):
        self.postings.append(p)

    def ledger_date(self) -> str:
        actual = self.date.time()
        if actual == datetime.min.time():
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

    def before(self, date: datetime) -> "Transactions":
        return Transactions([tx for tx in self.txs if tx.date <= date])

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
            date = datetime.strptime(date_string, "%Y/%m/%d")
            value = Decimal(value_string.replace("$", ""))

            if int(first) == 1:
                tx = Transaction(date, payee, len(cleared) > 0)
                txs.append(tx)
            assert tx
            tx.append(Posting(account, value, None))

        return Transactions(txs)


@dataclass
class IncomeDefinition:
    name: str
    epoch: datetime
    factor: Decimal


class Move:
    def txns(self):
        return []

    def compensate(self, names: Names):
        return []


@dataclass
class SimpleMove(Move):
    date: datetime
    value: Decimal
    from_path: str
    to_path: str
    payee: str

    def txns(self):
        tx = Transaction(self.date, self.payee, True)
        tx.append(Posting("[" + self.from_path + "]", -self.value, ""))
        tx.append(Posting("[" + self.to_path + "]", self.value, ""))
        return [tx]

    def compensate(self, names: Names):
        if self.to_path == names.emergency:  # HACK
            log.info(f"{self.date.date()} {self.to_path:50} {self.value:10} returning")
            return {self.to_path: self.value}
        return {}


@dataclass
class Taken:
    total: Decimal
    after: "DatedMoney"
    moves: Sequence[Move]
    payments: Sequence["Payment"] = field(default_factory=list)


@dataclass
class DatedMoney:
    date: datetime
    total: Decimal
    path: str
    note: str
    taken: Decimal = Decimal(0)
    where: Dict[str, Decimal] = field(default_factory=dict)

    def left(self) -> Decimal:
        return self.total - self.taken

    def redate(self, date: datetime) -> "DatedMoney":
        return DatedMoney(date, self.left(), self.path, self.note)

    def take(self, money: "DatedMoney", verb="payback", partial=False) -> Taken:
        assert money.total >= 0
        assert quantize(money.total) == money.total
        verb = verb if verb else "payback"
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
        after = DatedMoney(
            date=money.date,
            total=money.total - taking,
            path=money.path,
            note=money.note,
            where=money.where,
        )
        moves = [
            SimpleMove(
                money.date,
                taking,
                self.path,
                money.path,
                f"{verb} '{money.date.date()} {money.note}'",
            )
        ]
        return Taken(taking, after, moves)

    def move(
        self, path: str, note: str, date: Optional[datetime] = None
    ) -> Tuple[Optional["DatedMoney"], List[Move]]:
        left = self.left()
        if left == 0:
            return (None, [])
        self.taken = self.total
        effective_date = date if date else self.date
        moved = DatedMoney(date=effective_date, total=left, path=path, note=note)
        return (moved, [SimpleMove(effective_date, left, self.path, path, note)])


@dataclass
class RequirePayback(DatedMoney):
    def take(
        self, money: DatedMoney, verb: Optional[str] = None, partial: bool = False
    ) -> Taken:
        taken = super().take(money, verb=verb, partial=partial)
        log.info(f"{money.date.date()} {self.path:50} {taken.total:10} require-payback")
        payments = [
            Payment(
                date=money.date,
                total=taken.total,
                path=self.path,
                note=f"borrowing for '{money.date.date()} {money.note}'",
            )
        ]
        return Taken(taken.total, taken.after, taken.moves, payments)


def move_all_dated_money(
    dms: List[DatedMoney],
    note: str,
    path: str,
    date: Optional[datetime] = None,
) -> Tuple[List[DatedMoney], List[Move]]:
    tuples = [dm.move(path, note, date) for dm in dms]
    new_money = [dm for dm, m in tuples if dm]
    moves = flatten([m for dm, m in tuples])
    return (new_money, moves)


def redate_all_dated_money(dms: List[DatedMoney], date: datetime) -> List[DatedMoney]:
    return [dm.redate(date) for dm in dms]


class InsufficientFundsError(Exception):
    payment: DatedMoney


@dataclass
class MoneyPool:
    names: Names
    money: List[DatedMoney] = field(default_factory=list)

    def add(self, note: str, date: datetime, total: Decimal, path: str):
        self.money.append(DatedMoney(date=date, total=total, path=path, note=note))

    def include(self, more: List[DatedMoney]):
        self.money += more

    def left(self) -> Decimal:
        return Decimal(sum([dm.left() for dm in self.money]))

    def only_income(self, date: datetime, names: Names) -> "MoneyPool":
        return MoneyPool(
            self.names,
            [dm for dm in self.money if dm.date >= date and dm.path != names.emergency],
        )

    def only_emergency(self, date: datetime, names: Names) -> "MoneyPool":
        return MoneyPool(
            self.names,
            [dm for dm in self.money if dm.date >= date and dm.path == names.emergency],
        )

    def _can_use(self, taking: DatedMoney, available: DatedMoney) -> bool:
        if taking.path == available.path:
            return False
        return available.left() > 0 and taking.date >= available.date

    def _total_available(self, date: datetime) -> Decimal:
        available = [dm for dm in self.money if dm.date >= date]
        return Decimal(sum([dm.left() for dm in available]))

    def can_take(self, taking: DatedMoney) -> bool:
        available = [dm for dm in self.money if self._can_use(taking, dm)]
        total_available = sum([dm.left() for dm in available])
        return total_available >= taking.total

    def take(self, taking: DatedMoney) -> Taken:
        if not self.can_take(taking):
            for dm in self.money:
                using = self._can_use(taking, dm)
                log.warning(
                    f"{dm.date.date()} {dm.path:50} {dm.left():10} / {dm.total:10} taking={taking.path} using={using}"
                )
            total_available = self._total_available(taking.date)
            log.warning(
                f"{taking.date.date()} {taking.path:50} {taking.total:10} insufficient available {total_available}"
            )
            raise InsufficientFundsError(taking)

        available = [dm for dm in self.money if self._can_use(taking, dm)]
        moves: List[Move] = []
        payments: List[Payment] = []

        remaining = taking
        for available_money in available:
            verb = (
                "borrowing"
                if available_money.path == self.names.emergency
                else "payback"
            )
            taken = available_money.take(remaining, verb=verb, partial=True)
            remaining = taken.after
            if taken.moves:
                moves += taken.moves
                payments += taken.payments
            if remaining.total == 0:
                break

        assert remaining.total == 0

        return Taken(
            taking.total,
            DatedMoney(taking.date, Decimal(), "error", "error"),
            moves,
            payments,
        )

    def log(self):
        for money in self.money:
            left = money.left()
            if left > 0:
                log.info(
                    f"{money.date.date()} {money.path:50} {left:10} / {money.total:10}"
                )
            else:
                log.debug(
                    f"{money.date.date()} {money.path:50} {left:10} / {money.total:10}"
                )


@dataclass
class Tax:
    date: datetime
    rate: Decimal
    note: Optional[str] = None
    path: Optional[str] = None
    compiled: Optional[re.Pattern] = None

    def matches(self, p: DatedMoney) -> bool:
        if not self.compiled:
            self.compiled = re.compile(self.path if self.path else self.note)
        if self.note:
            return self.compiled.match(p.note) is not None
        return self.compiled.match(p.path) is not None


@dataclass
class TaxSystem:
    names: Names
    taxes: List[Tax] = field(default_factory=list)

    def add(
        self,
        date: datetime,
        rate: Decimal,
        note: Optional[str] = None,
        path: Optional[str] = None,
    ):
        log.info(f"{date.date()} {path or note:50} {'':10} taxing {rate:.2}")
        self.taxes.append(Tax(date, rate, note=note, path=path))

    def tax(self, payment: "DatedMoney", taken: Taken) -> List["Payment"]:
        for tax in self.taxes:
            if payment.date >= tax.date:
                if tax.rate > 0 and tax.matches(payment):
                    taxed = quantize(tax.rate * taken.total)
                    log.info(
                        f"{payment.date.date()} {payment.path:50} {payment.total:10} taxing {taxed:10} {self.names.taxes}"
                    )
                    return [
                        Payment(
                            date=payment.date,
                            total=taxed,
                            path=self.names.taxes,
                            note=f"{tax.rate:.2} taxes on {taken.total} from {payment.date.date()} {payment.note}",
                        )
                    ]
        return []


@dataclass
class Period(DatedMoney):
    tax_system: TaxSystem = field(default_factory=fail)
    income: IncomeDefinition = field(default_factory=fail)

    def after(self, spec: str) -> bool:
        testing = datetime.strptime(spec, "%m/%d/%Y")
        return self.date >= testing

    def before(self, spec: str) -> bool:
        testing = datetime.strptime(spec, "%m/%d/%Y")
        return self.date < testing

    def yearly(self, value: Decimal) -> str:
        v = quantize(value / Decimal(Decimal(12.0) * self.income.factor))
        taken = self.take(
            DatedMoney(
                date=self.date, total=v, path=self.income.name, note="yearly expense"
            )
        )
        assert taken.after.total == 0
        return f"{v:.2f}"

    def monthly(self, value: Decimal) -> str:
        v = quantize(value / self.income.factor)
        taken = self.take(
            DatedMoney(
                date=self.date, total=v, path=self.income.name, note="monthly expense"
            )
        )
        assert taken.after.total == 0
        return f"{v:.2f}"

    def done(self) -> str:
        l = self.left()
        if l < 0.0:
            raise Exception(f"overallocated ({l})")
        return f"{l:.2f}"

    def display(self) -> str:
        return f"{self.left():.2f}"

    def tax(
        self,
        rate: Optional[float] = None,
        path: Optional[str] = None,
        note: Optional[str] = None,
    ):
        assert path or note
        self.tax_system.add(
            self.date, Decimal(rate if rate else 0), path=path, note=note
        )
        return f"; tax {path} {rate}"

    def __str__(self) -> str:
        return self.date.strftime("%Y/%m/%d")

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Expense:
    date: datetime
    value: Decimal
    note: str
    path: str


@dataclass
class Payment(DatedMoney):
    def redate(self, date: Optional[datetime]) -> "Payment":
        if date and date != self.date:
            log.debug(
                f"{self.date.date()} {self.path:50} {self.total:10} payment redate {date.date()}"
            )
            return Payment(date, self.left(), self.path, self.note)
        return self


@dataclass
class Paid:
    moves: List[Move]
    available: List[DatedMoney]

    def combine(self, other: "Paid") -> "Paid":
        return Paid(self.moves + other.moves, self.available + other.available)


@dataclass
class Spending:
    names: Names
    today: datetime
    payments: List[Payment]
    tax_system: TaxSystem
    paid: List[Payment] = field(default_factory=list)
    verbose: bool = False

    def dates(self) -> List[datetime]:
        unique: Dict[datetime, bool] = {}
        for p in self.payments:
            unique[p.date] = True
        return list(unique.keys())

    def _make_payments(
        self,
        date: datetime,
        money: MoneyPool,
        upcoming_payments: Optional[datetime],
        payments: List[Payment],
        partial: bool = False,
    ) -> Paid:
        moves: List[Move] = []
        available: Dict[str, Decimal] = {}
        for p in payments:
            # In partial moded we're allowed to exit early, without handling
            # every payment we were given so bail if this one can't be handled.
            if partial:
                if not money.can_take(p):
                    break

            # This is here because spending money keeps the original date, so
            # bump to the income date so that the income can be used.
            redated = p.redate(date)

            # Pay for the thing.
            taken = money.take(redated)
            if len(taken.moves) > 0:
                assert taken.total > 0

                self.payments.remove(p)
                self.paid.append(p)
                need_sort = False

                for tax in self.tax_system.tax(redated, taken):
                    self.payments.append(tax)
                    need_sort = True

                for p in taken.payments:
                    self.payments.append(p.redate(upcoming_payments))
                    need_sort = True

                if need_sort:
                    self._sort()

                for m in taken.moves:
                    for key, value in m.compensate(self.names).items():
                        available[key] = available.setdefault(key, Decimal(0)) + value

                moves += taken.moves
            else:
                break

        # Be sure and maintain the RequirePayback behavior of money that's returned.
        return Paid(
            moves,
            [
                RequirePayback(date, total, path, "returned: ignored note")
                for path, total in available.items()
            ],
        )

    def _get_payments(self, date: datetime) -> Tuple[List[Payment], List[Payment]]:
        emergency: List[Payment] = []
        expenses: List[Payment] = []
        for payment in self.payments:
            if payment.date <= date and payment.date <= self.today:
                if payment.path == self.names.emergency:
                    emergency.append(payment)
                else:
                    expenses.append(payment)
        return (emergency, expenses)

    def pay_from(
        self,
        date: datetime,
        money: MoneyPool,
        upcoming_payments: Optional[datetime],
        paranoid: bool = False,
    ) -> Paid:
        assert upcoming_payments != date

        log.info(
            f"{date.date()} ------------------------------------------------------------------------------------------"
        )

        emergency, expenses = self._get_payments(date)
        total_expenses = sum([p.left() for p in expenses])
        total_emergency = sum([p.left() for p in emergency])
        logged_upcoming = upcoming_payments.date() if upcoming_payments else ""
        log.info(
            f"{date.date()} {'':50} {'':10} pay-from expenses={total_expenses:10} emergency={total_emergency:10} {len(expenses):2}/{len(emergency):2} upcoming={logged_upcoming}"
        )

        if self.verbose:
            for dm in money.money:
                log.debug(
                    f"{dm.date.date()} {dm.path:50} {dm.left():10} / {dm.total:10}"
                )
            log.debug(f"{date.date()}")

        if paranoid:
            # Pay off whatever we can from, partially so no errors if this fails.
            paid = self._make_payments(
                date, money, upcoming_payments, emergency + expenses, partial=True
            )

            # Now, get whatever can still be paid, this is a little more
            # expensive performance wise though way easier to maintain and work
            # with because there's only the one function to maintain.
            emergency, expenses = self._get_payments(date)

            # If we have a pay date coming up, we can be more flexible with when
            # emergency gets paid back.
            if upcoming_payments:
                # Move emergency payments to the upcoming pay date, we can never
                # have outstanding payments in the past because they'll never
                # get paid because we require payment dates to be after income
                # dates, so if an emergency payment can't be made we gotta redate.
                for p in emergency:
                    self.payments.remove(p)
                    self.payments.append(p.redate(upcoming_payments))

                # Now we require all expenses to be paid, this should fall back on
                # emergency to cover everything.
                return paid.combine(
                    self._make_payments(date, money, upcoming_payments, expenses)
                )

        # No unpaid payments allowed.
        return self._make_payments(date, money, upcoming_payments, emergency + expenses)

    def _sort(self):
        self.payments.sort(key=lambda p: (p.date, p.total))


@dataclass
class Schedule(Handler):
    today: datetime
    maximum: Optional[Decimal] = None

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
                        date=date, total=taking, path=expense.path, note=expense.note
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
                total=expense.value,
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
    tax_system: TaxSystem

    def expand(self, tx: Transaction, posting: Posting):
        log.info(f"{tx.date.date()} income: {posting} {posting.value}")
        return [
            Period(
                note=tx.payee,
                path=self.path,
                date=tx.date,
                total=posting.value.copy_abs(),
                income=self.income,
                tax_system=self.tax_system,
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
                date=tx.date,
                total=posting.value.copy_abs(),
                path=posting.account,
                note=tx.payee,
            )
        ]


@dataclass
class Configuration:
    ledger_file: str
    names: Names
    incomes: List[HandlerPath]
    spending: List[HandlerPath]
    emergency: List[HandlerPath]
    refund: List[HandlerPath]
    tax_system: TaxSystem
    income_pattern = "^income:"
    allocation_pattern = "^allocations:"


@dataclass
class Finances:
    cfg: Configuration
    today: datetime

    def allocate(self, file: TextIO, paranoid: bool):
        l = Ledger(self.cfg.ledger_file)

        names = self.cfg.names

        default_args = ["-S", "date", "--current"]
        exclude_allocations = ["and", "not", "tag(allocation)"]
        emergency_transactions = l.register(default_args + [names.emergency]).before(
            self.today
        )
        income_transactions = l.register(
            default_args + [self.cfg.income_pattern] + exclude_allocations
        ).before(self.today)
        allocation_transactions = l.register(
            default_args + [self.cfg.allocation_pattern] + exclude_allocations
        ).before(self.today)

        income_periods = income_transactions.apply_handlers(self.cfg.incomes)
        emergency_money = emergency_transactions.apply_handlers(self.cfg.emergency)
        refund_money = allocation_transactions.apply_handlers(self.cfg.refund)
        spending = Spending(
            names,
            self.today,
            [
                p
                for p in allocation_transactions.apply_handlers(self.cfg.spending)
                if p.date <= self.today
            ],
            self.cfg.tax_system,
        )

        ytd_spending = sum([e.total for e in spending.payments])

        # Allocate static income by rendering income template for each income
        # transaction, this will generate transactions directly into file and
        # return the left over income we can apply to spending.
        available = MoneyPool(names)
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

        # A move is a simplified model that produces Transactions.
        moves: List[Move] = []

        # This moves refunded money in allocations to a single refund account
        # and then makes that money available for payback.
        refunded_money_moved, move_refunded = move_all_dated_money(
            refund_money, "refunded", names.refunded
        )
        moves += move_refunded
        available.include(refunded_money_moved)

        # Require pay back of anything pulled from emergency.
        available.include(
            [
                RequirePayback(dm.date, dm.total, dm.path, dm.note)
                for dm in emergency_money
            ]
        )

        # If we're paranoid we'll be reconciling on income and every spend.
        income_dates = [period.date for period in income_periods]
        reconcile_dates = income_dates + spending.dates() if paranoid else income_dates
        reconcile_dates.sort()

        try:
            # Payback for spending for each pay period.
            for date in reconcile_dates:

                def get_payback_date():
                    # It's this method that determines the date that any
                    # payments generated during this pay back period should be
                    # paid back.

                    # Ideally, and in many situations, this will be the date of
                    # the income period following the reconciling date.

                    # If there are no income periods left, because we're on the
                    # final one, we use today's date.  The one downside to this
                    # is when today's date is also one of the `reconcile_dates`,
                    # which will break because then we'll end up processing the
                    # same date twice, and with no chance of additional income,
                    # and so this will fail if any emergency funds require
                    # payback.

                    # So we just add a day, for that scenario. This seems to be
                    # ok because this leaves the deductions from emergency, as
                    # they should be and ensures the emergency paybacks are
                    # never attempted given they'll fail due to insufficient
                    # funds.

                    # I'm also tired and this works.
                    upcoming_pay_dates = [f for f in income_dates if f > date]
                    future = self.today
                    if self.today in reconcile_dates:
                        future += timedelta(days=1)
                    return (
                        upcoming_pay_dates[0] if len(upcoming_pay_dates) > 0 else future
                    )

                # Pay for spending up until this date.
                paid = spending.pay_from(
                    date, available, get_payback_date(), paranoid=paranoid
                )
                moves += paid.moves

                # Make any money that was paid back to emergency (or anything
                # else down the road) available.
                available.include(paid.available)

            # How much is still unpaid for, this is spending accumulating until the
            # next pay period.
            unpaid_spending = sum([e.total for e in spending.payments])
            unpaid_spending_non_emergency = sum(
                [e.total for e in spending.payments if e.path != names.emergency]
            )

            for payment in spending.payments:
                log.info(f"{self.today.date()} unpaid {payment}")
            log.info(f"{self.today.date()} unpaid: {unpaid_spending}")

            # Only bother paying for spending if there's expenses to pay for, we
            # can leave emergency hanging.
            if unpaid_spending_non_emergency > 0:
                log.info(
                    f"{self.today.date()} unpaid-non-emergency: {unpaid_spending_non_emergency}"
                )
                paid = spending.pay_from(self.today, available, None, paranoid=paranoid)
                moves += paid.moves

            available.log()

            # Move any left over income to the available account.
            income_after_payback = sum([dm.left() for dm in available.money])
            left_over = [dm for dm in available.money if dm.path != names.emergency]
            _, moving = move_all_dated_money(
                left_over, "reserving left over income", names.available, self.today
            )
            moves += moving

            log.info(f"{self.today.date()} income-after-static: {income_after_static}")
            log.info(
                f"{self.today.date()} income-after-payback: {income_after_payback}"
            )
            log.info(f"{self.today.date()} ytd-spending: {ytd_spending}")
        except InsufficientFundsError:
            log.error("InsufficientFundsError", exc_info=True)

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


def parse_income(
    path: str,
    handler: Optional[Dict[str, Any]] = None,
    tax_system: Optional[TaxSystem] = None,
    **kwargs,
):
    assert handler
    assert tax_system
    return HandlerPath(
        path,
        IncomeHandler(
            IncomeDefinition(
                handler["income"]["name"],
                datetime.fromtimestamp(int(handler["income"]["epoch"])),
                Decimal(handler["income"]["factor"]),
            ),
            handler["path"],
            tax_system,
        ),
    )


def parse_spending_handler(
    today: Optional[datetime] = None, maximum: Optional[str] = None
) -> Handler:
    assert today
    if maximum:
        return Schedule(today=today, maximum=Decimal(maximum))
    return Schedule(today=today)


def parse_spending(
    path: str,
    handler: Optional[Dict[str, Any]] = None,
    today: Optional[datetime] = None,
    **kwargs,
):
    assert today
    if handler:
        return HandlerPath(path, parse_spending_handler(today=today, **handler))
    return HandlerPath(path, Schedule(today=today))


def parse_petty(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    return HandlerPath(path, DatedMoneyHandler())


def parse_emergency(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    return HandlerPath(path, DatedMoneyHandler())


def parse_refund(path: str, handler: Optional[Dict[str, Any]] = None, **kwargs):
    return HandlerPath(path, DatedMoneyHandler())


def parse_configuration(
    today: Optional[datetime] = None,
    ledger_file: Optional[str] = None,
    names: Mapping[str, Any] = None,
    income: List[Mapping[str, Any]] = None,
    spending: List[Mapping[str, Any]] = None,
    refund: List[Mapping[str, Any]] = None,
    emergency: List[Mapping[str, Any]] = None,
    **kwargs,
) -> Configuration:
    assert today
    assert ledger_file
    assert income
    assert spending
    assert refund
    assert emergency
    assert names
    names_parsed = parse_names(**names)
    tax_system = TaxSystem(names=names_parsed)
    return Configuration(
        ledger_file=ledger_file,
        names=names_parsed,
        incomes=[parse_income(tax_system=tax_system, **obj) for obj in income],
        spending=[parse_spending(today=today, **obj) for obj in spending],
        emergency=[parse_emergency(**obj) for obj in emergency],
        refund=[parse_refund(**obj) for obj in refund],
        tax_system=tax_system,
    )


def allocate(config_path: str, file_name: str, today: datetime, paranoid: bool) -> None:
    with open(config_path, "r") as file:
        raw = json.loads(file.read())
        configuration = parse_configuration(today=today, **raw)

    f = Finances(configuration, today)

    try_truncate_file(file_name)

    with open(file_name, "w") as file:
        txs = f.allocate(file, paranoid)
        t = get_generated_template()
        rendered = t.render(txs=txs)
        file.write(rendered)
        file.write("\n\n")


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log_file = logging.FileHandler("lalloc.log")
    log_file.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)7s] %(message)s",
        handlers=[console, log_file],
    )

    log = logging.getLogger("lalloc")

    parser = argparse.ArgumentParser(description="ledger allocations tool")
    parser.add_argument("-c", "--config-file", action="store", default="lalloc.json")
    parser.add_argument(
        "-l", "--ledger-file", action="store", default="lalloc.g.ledger"
    )
    parser.add_argument("-t", "--today", action="store", default=None)
    parser.add_argument("-p", "--paranoid", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--no-debug", action="store_true", default=False)
    args = parser.parse_args()

    today = datetime_today_that_is_sane()

    if args.debug:
        console.setLevel(logging.DEBUG)

    if args.today:
        today = datetime.strptime(args.today, "%Y/%m/%d")
        log.warning(f"today overriden to {today}")

    allocate(args.config_file, args.ledger_file, today, args.paranoid)
