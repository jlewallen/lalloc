#!/usr/bin/python3

from typing import Dict, Sequence, List, Optional, Union, Any, Tuple

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

import logging, argparse, subprocess
import json, sys, re
import hashlib, base64


txs_by_mid: Dict[str, "Transaction"] = {}


def flatten(a):
    return [leaf for sl in a for leaf in sl]


def calculate_transaction_hash(
    date: datetime, payee: str, values: List[Union[Decimal, int]]
) -> str:
    magnitude = sum([v for v in values if v > 0])
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{date}".encode())
    h.update(f"{payee}".encode())
    h.update(f"{magnitude}".encode())
    return base64.b32encode(h.digest()).decode("utf-8").replace("=", "")


@dataclass
class Posting:
    account: str
    value: Decimal
    note: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    lines: Optional[Tuple[int, int]] = None

    def ledger_value(self) -> str:
        return f"${self.value:.2f}"

    def ledger_tags(self) -> str:
        return " ".join([f"{t}:" for t in self.tags])

    def serialize(self) -> Dict[str, Any]:
        return dict(account=self.account, value=float(self.value), note=self.note)


@dataclass
class Transaction:
    date: datetime
    payee: str
    cleared: bool
    postings: List[Posting] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    mid: Optional[str] = None

    def append(self, p: Posting):
        self.postings.append(p)

    def ledger_date(self) -> str:
        actual = self.date.time()
        if actual == datetime.min.time():
            return self.date.strftime("%Y/%m/%d")
        return self.date.strftime("%Y/%m/%d %H:%M:%S")

    @property
    def unique_refs(self) -> Sequence[str]:
        unique: List[str] = []
        for ref in self.refs:
            if ref not in unique:
                unique.append(ref)
        return unique

    def total_value(self) -> Decimal:
        return Decimal(sum([p.value for p in self.postings]))

    def magnitude(self) -> Decimal:
        m = self.total_value()
        if abs(m) > 0.0001:
            return m
        return Decimal(sum([abs(p.value) for p in self.postings]))

    def has_references(self) -> bool:
        return len(re.findall(r"#(\S+)#", self.payee)) > 0

    def referenced_mids(self) -> List[str]:
        return flatten([s.split(",") for s in re.findall(r"#(\S+)#", self.payee)])

    def date_part(self) -> str:
        return self.date.strftime("%Y%m%d")

    def payee_part(self) -> str:
        simpler = re.sub("\(.+\)", "", self.payee).strip()
        simpler = re.sub("#\S+#", "", simpler).strip()
        return (
            simpler.replace("'", "")
            .replace(",", "")
            .replace(" ", "_")
            .replace("-", "")
            .replace("/", "_")
            .replace(":", "_")
            .replace(".", "_")
        )

    def has_account(self, account: str) -> bool:
        return len([p for p in self.postings if p.account == account]) > 0

    def has_account_matching(self, pattern: str) -> bool:
        return len([p for p in self.postings if re.fullmatch(pattern, p.account)]) > 0

    def balance(self, account: str) -> Decimal:
        return Decimal(sum([p.value for p in self.postings if p.account == account]))

    def with_postings_matching(self, pattern: str) -> "Transaction":
        return Transaction(
            self.date,
            self.payee,
            self.cleared,
            [p for p in self.postings if re.fullmatch(pattern, p.account)],
            mid=self.mid,
        )

    def with_postings_for(self, account: str) -> "Transaction":
        return Transaction(
            self.date,
            self.payee,
            self.cleared,
            [p for p in self.postings if p.account == account],
            mid=self.mid,
        )

    def serialize(self) -> Dict[str, Any]:
        return dict(
            date=self.date.isoformat(),
            payee=self.payee,
            cleared=self.cleared,
            postings=[p.serialize() for p in self.postings],
            mid=self.mid,
        )


@dataclass
class Transactions:
    txs: List[Transaction]

    def txns(self):
        return self.txs

    def accounts(self) -> Sequence[str]:
        return [
            account
            for account in {
                posting.account: True for tx in self.txs for posting in tx.postings
            }
        ]

    def before(self, date: datetime) -> "Transactions":
        return Transactions([tx for tx in self.txs if tx.date <= date])

    def after(self, date: datetime) -> "Transactions":
        return Transactions([tx for tx in self.txs if tx.date >= date])

    def with_postings_matching(self, pattern: str) -> "Transactions":
        return Transactions([tx for tx in self.txs if tx.has_account_matching(pattern)])

    def only_postings_for(self, account: str) -> "Transactions":
        return Transactions(
            [
                tx.with_postings_for(account)
                for tx in self.txs
                if tx.has_account(account)
            ]
        )

    def only_postings_matching(self, pattern: str) -> "Transactions":
        return Transactions(
            [
                tx.with_postings_matching(pattern)
                for tx in self.txs
                if tx.has_account_matching(pattern)
            ]
        )

    def account(self, account: str) -> "Transactions":
        return Transactions([tx for tx in self.txs if tx.has_account(account)])

    def balance(self, account: Optional[str] = None) -> Decimal:
        assert account
        return Decimal(
            sum([tx.balance(account) for tx in self.txs if tx.has_account(account)])
        )

    def exclude_with_references(self) -> "Transactions":
        return Transactions([tx for tx in self.txs if not tx.has_references()])

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
            "1|%S|%b|%e|%D|%A|%t|%X|%P|%N\n%/0|%S|%b|%e|%D|%A|%t|%X|%P|%N\n",
            "register",
        ] + expression
        sp = subprocess.run(command, stdout=subprocess.PIPE)

        log = logging.getLogger("ledger")
        log.info(" ".join(command).replace("\n", "NL"))

        txs: List[Transaction] = []
        tx: Union[Transaction, None] = None

        for line in sp.stdout.strip().decode("utf-8").split("\n"):
            fields = line.split("|")

            if len(fields) != 10:
                continue  # This ignores notes information!

            (
                first,
                file,
                start_line,
                end_line,
                date_string,
                account,
                value_string,
                cleared,
                payee,
                note,
            ) = fields

            date = datetime.strptime(date_string, "%Y/%m/%d")
            if "$" not in value_string:
                continue
            value = Decimal(value_string.replace("$", ""))

            if int(first) == 1:
                tx = Transaction(date, payee, len(cleared) > 0)
                txs.append(tx)

            assert tx
            tx.append(
                Posting(
                    account, value, note.strip(), lines=(int(start_line), int(end_line))
                )
            )

        by_mid: Dict[str, Transaction] = {}
        for tx in txs:
            # Maybe include file eventually?
            mid = calculate_transaction_hash(
                tx.date, tx.payee, flatten([p.lines for p in tx.postings])
            )

            if mid in by_mid:
                logging.warning(f"{mid} {tx}")
                logging.warning(f"{mid} {by_mid[mid]}")

            tx.mid = mid
            by_mid[mid] = tx

        return Transactions(txs)


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)7s] %(message)s",
        handlers=[console],
    )

    log = logging.getLogger("ledger")

    parser = argparse.ArgumentParser(description="ledger python wrapper")
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
