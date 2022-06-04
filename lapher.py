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


def is_virtual(path: str) -> bool:
    return (
        path.startswith("allocations:")
        or path.endswith(":reserved")
        or path.startswith("income:")  # May be controversial?
        or path.startswith("receivable:")  # May be controversial?
    )


def is_physical(path: str) -> bool:
    return not is_virtual(path)


def is_expense(path: str) -> bool:
    return path.startswith("expenses:")


def is_income(path: str) -> bool:
    return path.startswith("income:")


@dataclass
class Nodes:
    pass


@dataclass
class Node:
    id: str

    def __str__(self):
        return f"Node<>"

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence["Node"]:
        return []


@dataclass
class AccountNode(Node):
    path: str

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        return []


@dataclass
class PostingNode(Node):
    account: AccountNode
    value: Decimal

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        return []


@dataclass
class TransactionNode(Node):
    tx: ledger.Transaction
    postings: List[PostingNode] = field(default_factory=list)
    sibling: Optional["TransactionNode"] = None

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        pass


@dataclass
class DebugNode(Node):
    tx: ledger.Transaction

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        payee = self.tx.payee_part()
        f.write(f'  "{self.id}" [label="{payee}" color=blue]\n')
        return []


@dataclass
class SpentIncome:
    tx: ledger.Transaction
    total: Decimal
    taxes: bool


@dataclass
class EmergencyIncome:
    total: Decimal
    taxes: bool


@dataclass
class AllocatedIncome:
    account: str
    total: Decimal
    taxes: bool


@dataclass
class UnderstoodSpending:
    tx: ledger.Transaction
    income: List[SpentIncome]
    emergency: List[EmergencyIncome]
    allocated: List[AllocatedIncome]


@dataclass
class ReferencesNode(Node):
    original: ledger.Transaction
    txs: Sequence[ledger.Transaction]
    debugging: bool = False

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        return []

    def __str__(self):
        payees = " ".join([tx.payee_part() for tx in self.txs])
        return f"ReferencesNode<{len(self.txs)} {payees}>"

    def get_allocation_accounts(
        self, postings: List[ledger.Posting]
    ) -> List[AllocatedIncome]:
        # TODO Check polarity
        return [
            AllocatedIncome(p.account, p.value, False)
            for p in postings
            if p.account.startswith("allocations:")
            and "liabilities" not in p.account  # HACK To narrow search.
        ]

    def get_allocation_transaction(
        self, all_txs: ledger.Transactions
    ) -> List[AllocatedIncome]:
        return self.get_allocation_accounts(self.original.postings)

    def get_emergency_transaction(
        self, all_txs: ledger.Transactions
    ) -> List[EmergencyIncome]:
        spent: List[EmergencyIncome] = []

        for ref in self.txs:
            if ref.has_account_matching("^.+:emergency$"):
                paid = ref.total_matching("^.+:emergency$")
                if self.debugging:
                    log.info(f"  E {ref.date_part()} {ref.payee_part()} ${paid:.2f}")
                    for p in ref.postings:
                        log.info(f"{p.account} {p.ledger_value()}")
                taxes = "taxes_on" in ref.payee_part()
                spent.append(EmergencyIncome(paid, taxes))

        return spent

    def get_income_transactions(
        self, all_txs: ledger.Transactions
    ) -> List[SpentIncome]:

        spent: List[SpentIncome] = []
        for ref in self.txs:
            for other in ref.references(all_txs):
                if other != self.original:
                    if other.has_account_matching("^income:.+$"):
                        paid = ref.total_matching("^allocations:.+$")
                        if self.debugging:
                            log.info(f"  I '{ref.payee}'/'{other.payee}':")
                            log.info(
                                f"  I {other.date_part()} {other.payee_part()} ${paid:.2f}"
                            )
                            for p in ref.postings:
                                log.info(f"{p.account} {p.ledger_value()}")
                        taxes = "taxes_on" in ref.payee_part()
                        spent.append(SpentIncome(other, paid, taxes))
                    else:
                        log.info(f"  I '{other.payee}'")

        return spent

    def understand_spending(
        self, all_txs: ledger.Transactions
    ) -> Optional[UnderstoodSpending]:
        # Questions:
        # Where did the physical money come from?
        # Where did the logical money come from: "How was the expense paid off?"

        expensed = self.original.total_matching("^expenses:.+$")
        if expensed < 0:
            return None

        p = ",".join([p.short() for p in self.original.postings])
        log.info(
            f"{self.original.date.date()} {self.original.payee} ${expensed:.2f} {p}"
        )

        for ref in self.txs:
            p = ",".join([p.short() for p in ref.postings])
            log.info(f"  R {p} '{ref.payee_part()}'")

        income = self.get_income_transactions(all_txs)
        emergency = self.get_emergency_transaction(all_txs)
        allocated = self.get_allocation_transaction(all_txs)

        if self.debugging:
            for spent_emergency in emergency:
                log.info(
                    f"  EMR ${spent_emergency.total:.2f} {'taxes' if spent_emergency.taxes else ''}"
                )

            for spent_allocated in allocated:
                log.info(
                    f"  ALL ${spent_allocated.total:.2f} {spent_allocated.account} {'taxes' if spent_allocated.taxes else ''}"
                )

            for spent_income in income:
                p = ",".join([p.short() for p in spent_income.tx.postings])
                log.info(f"  INC {spent_income.tx.payee} {p} ${spent_income.total:.2f}")

        return UnderstoodSpending(
            self.original,
            [i for i in income if not i.taxes],
            [e for e in emergency if not e.taxes],
            [a for a in allocated if not a.taxes],
        )


@dataclass
class UnderstoodIncome:
    deposited: List[ledger.Posting]
    allocated: Mapping[str, Decimal]
    payback: Mapping[str, Decimal]


@dataclass
class IncomeNode(Node):
    tx: ledger.Transaction
    sibling: Optional["Node"] = None
    debugging: bool = True

    def understand(self, all_txs: ledger.Transactions) -> UnderstoodIncome:
        log.info(f"{self.tx.date.date()} '{self.tx.payee}'")

        assert self.tx.mid

        # Find physical accounts.
        physically_deposited = [p for p in self.tx.postings if is_physical(p.account)]
        if len(physically_deposited) != 1:
            log.info(f"Physical: {self.tx.postings}")
        assert len(physically_deposited) >= 1

        if self.debugging:
            for physical in physically_deposited:
                log.info(f"  PHY {physical.account} += {physical.ledger_value()}")

        allocated: Dict[str, List[Decimal]] = dict()
        payback: Dict[str, List[Decimal]] = dict()

        for reference in all_txs.find_references(self.tx.mid):
            if self.debugging:
                log.info(f"  REF '{reference.payee}'")

            if "preallocating" in reference.payee:
                for posting in reference.postings:
                    if posting.value > 0:
                        allocated.setdefault(posting.account, []).append(posting.value)
            elif "payback" in reference.payee:
                for posting in reference.postings:
                    if posting.value > 0:
                        payback.setdefault(posting.account, []).append(posting.value)
            else:
                for posting in reference.postings:
                    if posting.value > 0:
                        allocated.setdefault(posting.account, []).append(posting.value)

        return UnderstoodIncome(
            deposited=physically_deposited,
            allocated={
                path: Decimal(sum(values)) for path, values in allocated.items()
            },
            payback={path: Decimal(sum(values)) for path, values in payback.items()},
        )

    def graphviz(
        self, all_txs: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        visiting = [self.sibling] if self.sibling else []
        date_part = self.tx.date_part()
        payee_part = self.tx.payee_part()
        understood = self.understand(all_txs)

        f.write(
            f'  "{self.id}" [label="{date_part} {payee_part}" style=filled fillcolor=green color=white fontcolor=black]\n'
        )

        for posting in understood.deposited:
            f.write(f'  "{self.id}" -- "{posting.account}"\n')

        for path, allocated in understood.allocated.items():
            f.write(f'  "{self.id}" -- "{path}"\n')

        for path, allocated in understood.payback.items():
            f.write(f'  "{self.id}" -- "{path}"\n')

        if self.sibling:
            f.write(f'  "{self.id}" -- "{self.sibling.id}" [color=gray]\n')

        return visiting


@dataclass
class ExpenseNode(Node):
    tx: ledger.Transaction
    sibling: Optional["Node"] = None
    debug_refs: List[DebugNode] = field(default_factory=list)
    debugging: bool = False

    def graphviz(
        self, all: ledger.Transactions, nodes: Nodes, f: TextIO
    ) -> Sequence[Node]:
        date_part = self.tx.date_part()
        payee_part = self.tx.payee_part()

        refs = ReferencesNode(self.id, self.tx, [d.tx for d in self.debug_refs])

        # TODO Change color if no understood spending
        understood = refs.understand_spending(all)

        color = (
            "red"
            if understood
            and (understood.income or understood.emergency or understood.allocated)
            else "blue"
        )

        f.write(
            f'  "{self.id}" [label="{date_part} {payee_part}" style=filled fillcolor={color} color=white fontcolor=white]\n'
        )

        visiting: List[Node] = []

        if understood:
            # TODO Group by
            by_income_tx: Dict[str, List[SpentIncome]] = {}
            for spent_income in understood.income:
                assert spent_income.tx.mid
                by_income_tx.setdefault(spent_income.tx.mid, []).append(spent_income)

            for mid, spent_incomes in by_income_tx.items():
                income_tx = spent_incomes[0].tx
                spent_income_total = sum([si.total for si in spent_incomes])

                color = "black"
                f.write(
                    f'  "{self.id}" -- "{income_tx.mid}" [label=" {spent_income_total:{".0" if abs(spent_income_total) > 1 else ".2"}f}" color={color}]\n'  # type:ignore
                )
                assert income_tx.mid
                visiting.append(IncomeNode(id=income_tx.mid, tx=income_tx))

            if not understood.income:
                if understood.emergency:
                    total_paid = sum([se.total for se in understood.emergency])
                    color = "pink"
                    f.write(
                        f'  "{self.id}" -- "Emergency" [label=" {total_paid:{".0" if abs(total_paid) > 1 else ".2"}f}" color={color}]\n'  # type:ignore
                    )
                elif understood.allocated:
                    for a in understood.allocated:
                        total_paid = a.total
                        color = "purple"
                        f.write(
                            f'  "{self.id}" -- "{a.account}" [label=" {total_paid:{".0" if abs(total_paid) > 1 else ".2"}f}" color={color}]\n'
                        )

        if self.debugging:
            for debug_ref in self.debug_refs:
                m = re.findall(
                    r"taxes_on_\d+(_\d+)?_from_\d+", debug_ref.tx.payee_part()
                )
                if m:
                    pass
                else:
                    f.write(
                        f'  "{self.id}" -- "{debug_ref.id}" [color=blue] /* {debug_ref.tx.payee_part()} */\n'
                    )
                    visiting.append(debug_ref)

        if self.debugging:
            for posting in [p for p in self.tx.postings if is_expense(p.account)]:
                f.write(f'  "{self.id}" -- "{posting.account}" [color=red]\n')

        if self.sibling:
            f.write(f'  "{self.id}" -- "{self.sibling.id}" [color=gray]\n')
            return visiting + [self.sibling]

        return visiting


def is_skipping(tx: ledger.Transaction) -> bool:
    return (
        re.fullmatch("^\S+ interest$", tx.payee) is not None or "dividend" in tx.payee
    )


def create_income_graph(
    file: str, start: Optional[datetime]
) -> Tuple[Node, ledger.Transactions, Nodes]:
    txs = load(file)

    all_income = txs.with_postings_matching("^income:.+$")
    if start:
        all_income = all_income.after(start)

    head: Optional[IncomeNode] = None
    tail: Optional[IncomeNode] = None

    for tx in [t for t in all_income.txns() if not is_skipping(t)]:
        log.info(f"{tx.payee} {tx.postings}")
        assert "lowes" not in tx.payee

        income_node = IncomeNode(tx.mid, tx)

        if head:
            assert tail
            tail.sibling = income_node
            tail = tail.sibling
        else:
            head = income_node
            tail = head

    assert head

    return head, txs, Nodes()


def create_expenses_graph(
    file: str, start: Optional[datetime]
) -> Tuple[Node, ledger.Transactions, Nodes]:
    txs = load(file)

    by_referenced: Dict[str, List[DebugNode]] = {}
    for tx in txs.txns():
        for ref in tx.referenced_mids():
            arr = by_referenced.setdefault(ref, [])
            arr.append(DebugNode(tx.mid, tx))

    all_expenses = txs.with_postings_matching("^expenses:.+$")
    if start:
        all_expenses = all_expenses.after(start)

    head: Optional[ExpenseNode] = None
    tail: Optional[ExpenseNode] = None

    for tx in all_expenses.txns():
        debug_refs = by_referenced[tx.mid] if tx.mid in by_referenced else []
        expense_node = ExpenseNode(tx.mid, tx, debug_refs=debug_refs)

        if head:
            assert tail
            tail.sibling = expense_node
            tail = tail.sibling
        else:
            head = expense_node
            tail = head

    assert head

    return head, txs, Nodes()


def create_initial_transaction_graph(
    file: str, start: Optional[datetime]
) -> Tuple[Node, ledger.Transactions, Nodes]:
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

    return head, txs, Nodes()


def graph(json_file: str, dot_file: str, start: Optional[datetime]):
    # head, txs, nodes = create_initial_transaction_graph(json_file)
    # head, txs, nodes = create_expenses_graph(json_file, start=start)
    head, txs, nodes = create_income_graph(json_file, start=start)

    queue: List[Node] = [head]
    seen: List[str] = []
    with open(dot_file, "w") as f:
        f.write("strict graph {\n")
        while len(queue) > 0:
            visiting = queue.pop()
            if visiting.id not in seen:
                seen.append(visiting.id)
                for n in visiting.graphviz(txs, nodes, f):
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
    assert cleared is not None
    return ledger.Transaction(
        date=datetime.strptime(date, "%Y-%m-%dT%H:%M:%S"),
        payee=payee,
        cleared=cleared,
        mid=mid,
        postings=[load_posting(**p) for p in postings] if postings else [],
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
    parser.add_argument("-s", "--start", action="store")
    args = parser.parse_args()

    if args.debug:
        console.setLevel(logging.DEBUG)

    start: Optional[datetime] = None
    if args.start:
        start = datetime.strptime(args.start, "%Y/%m/%d")
        log.warning(f"start overriden to {start}")

    graph(args.json_file, args.dot_file, start)
