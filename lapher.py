#!/usr/bin/python3

from typing import Tuple, Any, Dict, Sequence, List, TextIO, Optional, Mapping, Union

from dataclasses import dataclass, field
from dateutil import relativedelta
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, date

import argparse, logging, json, re

import ledger


def flatten(a):
    return [leaf for sl in a for leaf in sl]


@dataclass
class Node:
    id: str

    def __str__(self):
        return f"Node<>"

    def graphviz(self, nodes: Mapping[str, "Node"], f: TextIO) -> List["Node"]:
        return []


@dataclass
class AccountNode(Node):
    path: str

    def graphviz(self, nodes: Mapping[str, Node], f: TextIO) -> List[Node]:
        return []


@dataclass
class OpeningNode(Node):
    pass


@dataclass
class PostingNode(Node):
    account: AccountNode
    value: Decimal

    def graphviz(self, nodes: Mapping[str, Node], f: TextIO) -> List[Node]:
        return []


@dataclass
class TransactionNode(Node):
    tx: ledger.Transaction
    postings: List[PostingNode] = field(default_factory=list)
    sibling: Optional["TransactionNode"] = None

    def graphviz(self, nodes: Mapping[str, Node], f: TextIO) -> List[Node]:
        for mid in self.tx.referenced_mids():
            if mid in nodes:
                f.write(f"  {self.id} -- {nodes[mid].id}\n")
        if self.sibling:
            f.write(f"  {self.id} -- {self.sibling.id} [color=blue]\n")
            return [self.sibling]
        return []


@dataclass
class WithdrawlNode(Node):
    pass


@dataclass
class SpendNode(Node):
    pass


@dataclass
class ExpenseNode(Node):
    tx: ledger.Transaction
    sibling: Optional[Node] = None

    def graphviz(self, nodes: Mapping[str, Node], f: TextIO) -> List[Node]:
        if self.sibling:
            f.write(f'  "{self.id}" -- "{self.sibling.id}" [color=blue]\n')
            return [self.sibling]
        return []


def create_expenses_graph(file: str) -> Tuple[Node, Dict[str, Node]]:
    txs = load(file).only_postings_matching("^expenses:.+$")

    nodes: Dict[str, Node] = {}
    head: Optional[ExpenseNode] = None
    tail: Optional[ExpenseNode] = None

    for tx in txs.txns():
        if head:
            assert tail
            tail.sibling = ExpenseNode(tx.mid, tx)
            tail = tail.sibling
        else:
            head = ExpenseNode(tx.mid, tx)
            tail = head

    assert head

    return head, nodes


def create_initial_transaction_graph(
    file: str,
) -> Tuple[Node, Dict[str, TransactionNode]]:
    txs = load(file)

    head: Optional[TransactionNode] = None
    tail: Optional[TransactionNode] = None
    accounts: Dict[str, AccountNode] = {}
    nodes: Dict[str, TransactionNode] = {}

    def get_account(path: str) -> AccountNode:
        if path not in accounts:
            accounts[path] = AccountNode(path.replace(":", "_"), path)
        return accounts[path]

    for index, tx in enumerate(txs.txns()):
        tx_id = f"T{tx.date_part()}_{index}_{tx.payee_part()}"

        postings = [
            PostingNode(f"T{index}_{pindex}_{index}", get_account(p.path), p.value)
            for pindex, p in enumerate(tx.postings)
        ]

        node = TransactionNode(tx_id, tx, postings=postings)

        nodes[tx.mid] = node

        if head:
            assert tail
            tail.sibling = node
            tail = node
        else:
            head = node
            tail = node

    assert head

    return head, nodes


def graph(json_file: str, dot_file: str):
    # head, nodes = create_initial_transaction_graph(json_file)
    head, nodes = create_expenses_graph(json_file)

    queue: List[Node] = [head]
    seen: List[Node] = []
    with open(dot_file, "w") as f:
        f.write("strict graph {\n")
        while len(queue) > 0:
            visiting = queue.pop()
            if visiting not in seen:
                seen.append(visiting)
                for n in visiting.graphviz(nodes, f):
                    queue.append(n)
        f.write("}\n")


def load_posting(
    account: Optional[str] = None,
    note: Optional[str] = None,
    value: Optional[str] = None,
) -> ledger.Posting:
    assert account
    assert value
    return ledger.Posting(account=account, note=note, value=Decimal(value))


def load_transaction(
    date: Optional[str] = None,
    payee: Optional[str] = None,
    cleared: Optional[bool] = None,
    mid: Optional[str] = None,
    postings: Optional[List[Dict[str, Any]]] = None,
) -> ledger.Transaction:
    assert date
    assert payee
    assert mid
    assert postings
    assert cleared is not None
    return ledger.Transaction(
        date=datetime.strptime(date, "%Y-%m-%dT%H:%M:%S"),
        payee=payee,
        cleared=cleared,
        mid=mid,
        postings=[load_posting(**p) for p in postings],
    )


def load(file: str) -> ledger.Transactions:
    with open(file) as f:
        return ledger.Transactions([load_transaction(**tx) for tx in json.load(f)])


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
