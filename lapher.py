#!/usr/bin/python3

from typing import Tuple, Any, Dict, Sequence, List, TextIO, Optional, Mapping, Union

from dataclasses import dataclass, field
from dateutil import relativedelta
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, date

import argparse, logging, json, re


@dataclass
class Posting:
    path: str
    value: Decimal
    note: str


@dataclass
class Transaction:
    date: datetime
    payee: str
    cleared: bool
    postings: List[Posting]
    mid: str

    def date_part(self) -> str:
        return self.date.strftime("%Y%m%d")

    def payee_part(self) -> str:
        simpler = re.sub("\(.+\)", "", self.payee).strip()
        simpler = re.sub("#", "_", simpler).strip()
        return (
            simpler.replace("'", "")
            .replace(",", "")
            .replace(" ", "_")
            .replace("-", "")
            .replace(".", "_")
        )


@dataclass
class Node:
    id: str

    def __str__(self):
        return f"Node<>"

    def graphviz(self, f: TextIO) -> List["Node"]:
        return []


@dataclass
class AccountNode(Node):
    path: str

    def graphviz(self, f: TextIO) -> List[Node]:
        return []


@dataclass
class OpeningNode(Node):
    pass


@dataclass
class PostingNode(Node):
    account: AccountNode
    value: Decimal

    def graphviz(self, f: TextIO) -> List[Node]:
        return []


@dataclass
class TransactionNode(Node):
    tx: Transaction
    postings: List[PostingNode] = field(default_factory=list)
    sibling: Optional["TransactionNode"] = None

    def graphviz(self, f: TextIO) -> List[Node]:
        if self.sibling:
            f.write(f"  {self.id} -- {self.sibling.id}\n")
            return [self.sibling]
        return []


@dataclass
class WithdrawlNode(Node):
    pass


@dataclass
class SpendNode(Node):
    pass


def create_initial_transaction_graph(file: str) -> TransactionNode:
    txs = load(file)

    head: Optional[TransactionNode] = None
    tail: Optional[TransactionNode] = None
    accounts: Dict[str, AccountNode] = {}

    def get_account(path: str) -> AccountNode:
        if path not in accounts:
            accounts[path] = AccountNode(path.replace(":", "_"), path)
        return accounts[path]

    for index, tx in enumerate(txs):
        tx_id = f"T{tx.date_part()}_{index}_{tx.payee_part()}"
        postings = [
            PostingNode(f"T{index}_{pindex}_{index}", get_account(p.path), p.value)
            for pindex, p in enumerate(tx.postings)
        ]
        node = TransactionNode(tx_id, tx, postings=postings)
        if head:
            tail.sibling = node
            tail = node
        else:
            head = node
            tail = node

    return head


def graph(json_file: str, dot_file: str):
    head = create_initial_transaction_graph(json_file)

    queue: List[Node] = [head]
    seen: List[Node] = []
    with open(dot_file, "w") as f:
        f.write("strict graph {\n")
        while len(queue) > 0:
            visiting = queue.pop()
            if visiting not in seen:
                seen.append(visiting)
                for n in visiting.graphviz(f):
                    queue.append(n)
        f.write("}\n")


def load_posting(
    account: Optional[str] = None,
    note: Optional[str] = None,
    value: Optional[str] = None,
) -> Posting:
    return Posting(path=account, note=note, value=Decimal(value))


def load_transaction(
    date: Optional[str] = None,
    payee: Optional[str] = None,
    cleared: Optional[bool] = None,
    mid: Optional[str] = None,
    postings: Optional[Dict[str, Any]] = None,
) -> Transaction:
    assert date
    assert payee
    assert mid
    assert postings
    return Transaction(
        date=datetime.strptime(date, "%Y-%m-%dT%H:%M:%S"),
        payee=payee,
        cleared=cleared,
        mid=mid,
        postings=[load_posting(**p) for p in postings],
    )


def load(file: str) -> List[Transaction]:
    with open(file) as f:
        return [load_transaction(**tx) for tx in json.load(f)]


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)7s] %(message)s",
        handlers=[console],
    )

    log = logging.getLogger("lapher")

    parser = argparse.ArgumentParser(description="ledger graph tool")
    parser.add_argument("-f", "--json-file", action="store", required=True)
    parser.add_argument("-o", "--dot-file", action="store", required=True)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        console.setLevel(logging.DEBUG)

    graph(args.json_file, args.dot_file)
