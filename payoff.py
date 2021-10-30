#!/usr/bin/python3

import argparse
import logging
import datetime

from dateutil import relativedelta
from dataclasses import dataclass, field


@dataclass
class PayoffCalculator:
    total: float
    name: str
    # expense: str
    # allocation: str
    date: datetime.date
    payments: int

    def calculate(self):
        indention = "    "
        source = "[allocations:checking:shaved]"
        destination = "[allocations:checking:savings:main:payoff]"
        remaining = int(self.total * 100)
        payment = int(remaining / self.payments)

        payment_date = self.date
        for n in range(self.payments):
            p = remaining if n == self.payments - 1 else payment
            print(f"{payment_date.strftime('%Y/%m/%d')} * {self.name}")
            print(f"{indention}{source:50}-${p/100:.2f}")
            print(f"{indention}{destination:50} ${p/100:.2f}")
            print()
            remaining -= p
            payment_date += relativedelta.relativedelta(months=1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)5s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    log = logging.getLogger("money")

    parser = argparse.ArgumentParser(description="payoff tool")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-n", "--name", action="store", default="some random expense")
    parser.add_argument(
        "-e", "--expense", action="store", default="some random expense"
    )
    parser.add_argument(
        "-a", "--allocation", action="store", default="some random expense"
    )
    parser.add_argument("-t", "--total", action="store", default=1000, type=float)
    parser.add_argument("-p", "--payments", action="store", default=12, type=int)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    date = datetime.date.today()
    calculator = PayoffCalculator(
        name=args.name, total=args.total, payments=args.payments, date=date
    )
    calculator.calculate()
