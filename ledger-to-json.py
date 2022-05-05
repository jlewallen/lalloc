#!/usr/bin/python3

from typing import Dict, Sequence, List, Optional, Union, Any

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

import json, sys
import subprocess
import logging
import argparse


@dataclass
class Posting:
    account: str
    value: Decimal
    note: Optional[str] = None

    def serialize(self) -> Dict[str, Any]:
        return dict(account=self.account, value=float(self.value), note=self.note)


@dataclass
class Transaction:
    date: datetime
    payee: str
    cleared: bool
    postings: List[Posting] = field(default_factory=list)

    def append(self, p: Posting):
        self.postings.append(p)

    def serialize(self) -> Dict[str, Any]:
        return dict(
            date=self.date.isoformat(),
            payee=self.payee,
            cleared=self.cleared,
            postings=[p.serialize() for p in self.postings],
        )


@dataclass
class Transactions:
    txs: List[Transaction]

    def serialize(self) -> List[Dict[str, Any]]:
        return [tx.serialize() for tx in self.txs]


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
            "1|%D|%A|%t|%X|%P|%N\n%/0|%D|%A|%t|%X|%P|%N\n",
            "register",
        ] + expression
        sp = subprocess.run(command, stdout=subprocess.PIPE)

        log.info(" ".join(command).replace("\n", "NL"))

        txs: List[Transaction] = []
        tx: Union[Transaction, None] = None

        for line in sp.stdout.strip().decode("utf-8").split("\n"):
            fields = line.split("|")
            if len(fields) != 7:
                continue

            first, date_string, account, value_string, cleared, payee, note = fields
            date = datetime.strptime(date_string, "%Y/%m/%d")
            if "$" not in value_string:
                continue
            value = Decimal(value_string.replace("$", ""))

            if int(first) == 1:
                tx = Transaction(date, payee, len(cleared) > 0)
                txs.append(tx)
            assert tx
            tx.append(Posting(account, value, note.strip()))

        return Transactions(txs)


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)7s] %(message)s",
        handlers=[console],
    )

    log = logging.getLogger("ledger-to-json")

    parser = argparse.ArgumentParser(description="ledger allocations tool")
    parser.add_argument("-l", "--ledger-file", action="store", required=True)
    parser.add_argument("-o", "--output-file", action="store")
    args = parser.parse_args()

    ledger = Ledger(args.ledger_file)
    all_transactions = ledger.register([])
    serialized = all_transactions.serialize()

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(serialized, f)
    else:
        json.dump(serialized, sys.stdout)
