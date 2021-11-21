#!/usr/bin/python3

import argparse
import logging
import datetime

from dateutil import relativedelta
from dataclasses import dataclass, field


@dataclass
class PayoffCalculator:
    total: float
    date: datetime.date
    payments: int
    source: str
    destination: str
    payee: str

    def calculate(self):
        indention = "    "
        remaining = int(self.total * 100)
        payment = int(remaining / self.payments)

        log.info(
            f"generating {self.total} payoff {self.source} -> {self.destination} for '{self.payee}'"
        )

        payment_date = self.date
        for n in range(self.payments):
            p = remaining if n == self.payments - 1 else payment
            print(f"{payment_date.strftime('%Y/%m/%d')} * {self.payee}")
            print(f"{indention}{self.source:50}-${p/100:.2f}")
            print(f"{indention}{self.destination:50} ${p/100:.2f}")
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

    log = logging.getLogger("payoff")

    parser = argparse.ArgumentParser(description="payoff tool")
    parser.add_argument("-D", "--debug", action="store_true", default=False)
    parser.add_argument("-p", "--payee", action="store", default="some random expense")
    parser.add_argument(
        "-s", "--source", action="store", default="[allocations:checking:auto:payoff]"
    )
    parser.add_argument(
        "-d",
        "--destination",
        action="store",
        default="[allocations:checking:savings:main:payoff]",
    )
    parser.add_argument("-t", "--total", action="store", default=1000, type=float)
    parser.add_argument("-n", "--payments", action="store", default=12, type=int)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    date = datetime.date.today()
    calculator = PayoffCalculator(
        total=args.total,
        payments=args.payments,
        date=date,
        source=args.source,
        destination=args.destination,
        payee=args.payee,
    )
    calculator.calculate()
