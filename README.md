# lalloc

## What is lalloc?

Lalloc is a tool for use alongside [ledger-cli](https://www.ledger-cli.org/) to
help manage personal finances. It builds on top of a slightly opinionated
accounting structure to deliver some tools to help automate and streamline the
allocation of incoming money to expenses and savings goals.

## What is ledger-cli?

According to the [ledger-cli](https://www.ledger-cli.org/) site:

> "Ledger is a powerful, double-entry accounting system that is accessed from the UNIX command-line."

To elaborate, [ledger-cli](https://www.ledger-cli.org/) parses simple text
files containing transaction history and a sprinkling of other directives in
order to allow querying and inspection of accounting information. It's a very
nerdy way to manage personal finances, admittedly.

**Note**: A familiarity with double entry accounting systems is recommended before continuing.

## What does it do?

Long story short, `lalloc` manages what is effectively an elaborate envelope
based budgeting system. When income is deposited, that money is allocated to
expenses and savings goals based on user defined rules. These envelopes are
called allocations. The envelopes themselves are `ledger-cli` virtual accounts
that exist alongside the physical accounts.

When run, `lalloc` will begin by examining the ledger file for income. These
are identified by looking for withdrawals from accounts that match a pattern,
for example `income:.*`

```
2021/01/15 * income
    income:job                                                                       -$2000.00
    assets:checking                                                                   $2000.00
```

Once these are collected, `lalloc` will look for a template with a name
matching each income. In the above example `lalloc` will look for
`job.ledger.template`. This template generates allocations that get the ball
rolling. It's in this file that you can allocate fixed money to things that are
predictable and/or significant. Rent/mortgage, property taxes, once yearly
bills, and savings are all excellent candidates for allocating in this file.
This template is expected to yield valid `ledger-cli` text and has flow control
and some helpers for defining these kinds of allocations:

```
{{ period }} * job income allocation (fixed)
    ; allocation: job
    [allocations:checking:mortgage]              ${{ period.monthly(1200) }}  ; mortgage
    [allocations:checking:taxes]                 ${{ period.yearly(6000) }}   ; property taxes
    [allocations:savings]                        ${{ period.monthly(1000) }}  ; savings
    [assets:checking:reserved]                 ; ${{ period.display() }}

{% if period.after('2/5/2021') -%}

{{ period }} * job income allocation (fixed)
    ; allocation: job
    [allocations:checking:rec]                   ${{ period.monthly(200) }}   ; rec spending
    [assets:checking:reserved]                 ; ${{ period.display() }}

{% endif -%}
```

**Note**: It's possible to use just this template to manage your finances.

After that file is processed, `lalloc` is then able to calculate how much of
that income  is available to be allocated to other expenses "automatically".

A configuration file defines which allocations are treated "automatically" in
the following steps. These are expenses that depend more on personal behavior
and choice than obligation - dining, miscellanea for the home or drinks at the
bar are some examples.

```
2021/01/07 * lan noodle
    assets:checking                                                                   -$100.00
    expenses:dining                                                                    $100.00
```

To find this spending, `lalloc` will scan the allocations that have been
configured and look for spending events. For each paycheck, _whatever income is
available_ is divided up among any _previous_ spending.

**Note**: `lalloc` will only use previously unallocated income to
cover spending, never future income.

This means that if a purchase occurs on the first of the month and there's no
available income to cover that purchase, the funds to cover that expense have
to come from somewhere. For this purpose, an emergency or buffer fund is used.
This is an account that has savings that can be borrowed against to cover any
excess spending. The amount in this account should be roughly equal to the
amount of automatically covered spending you're likely to have in a pay period.
`lalloc` will then reimburse the emergency fund when more income is deposisted.

After all previous spending has been 'covered' the remaining income is moved to
an `available` account. The balance of this account indicates how much money is
available for spending on anything unanticipated. If all income to date has
been allocated and none is available, then the balance of the emergency account
indicates how serious the situation is, as that's where money to cover spending
will be allocated from until more income is available.

## Why?

There are several benefits from doing all of this work:

1) Money available to spend on automatically covered expenses is visible in the
`available` account. If you want to go out to dinner or splurge on a minor
purchase, the amount of money available to allocate to those kinds of purchases
is easily visible. In situations where no income is available for these kinds
of purchases, the emergency fund will show how far you've gone into the
"warning zone" So you can either adjust your future behavior or shuffle money
around to compensate.

```
            $2360.00  allocations:checking:available
            $1200.00  allocations:checking:mortgage
             $600.00  allocations:checking:rec
            $3460.00  allocations:checking:savings
             $500.00  allocations:checking:savings:emergency
              $60.00  allocations:checking:savings:main
           $-1600.00  allocations:checking:savings:slow
            $2000.00  allocations:checking:taxes
            $4000.00  allocations:savings
          $-13620.00  assets:checking:reserved
           $-5000.00  equity
             $600.00  expenses:dinner
            $1100.00  expenses:groceries
            $3600.00  expenses:mortgage
              $80.00  expenses:services
            $2000.00  expenses:whoops
          $-16000.00  income:job
```

By monitoring the balance of these two accounts you can tune behavior to the
current financial situation.

2) Because the accounts used to manage the allocations are all virtual and
exist alongside the physical accounts the algorithm can be adjusted and toyed
with very easily. For example, it's possible to make small changes to the job
template to try new life situations - increased rent, savings, a car payment,
etc... It will then adjust automatic spending and alert you if you don't have
the buffer/emergency on hand to handle the swings.

All of the ledger transactions that `lalloc` generates are written to a single
file that can be easily included in your main `ledger-cli` file and thrown away
if you decide the system's not for you.

## Tricks

In the configuration file it's possible to alter the semantics of an allocation
account, and this offers some nifty tricks:

### Self Tax

Sometimes you want to save extra money for some goal but you'd like that money
to come from your day to day income that would normally be used to cover
automatic expenses rather than from your existing (usually fixed) savings
allocation.

One way to do this is to tax some categories of discretionary spending. This
means that when an allocation is withdrawn from, an extra payment is scheduled
for the taxed amount to a savings account and covered just like any other
automatic expense. Over time, the extra savings grows and requires very little
behavioral change, as at the end of the day all that's required is to pay
attention to the `available` or `emergency` balances.  Ideas for this include a
fancy dinner tax or a tax on recreation.

```
2021/02/09 * fancy dinner
    assets:checking                                                                   -$150.00
    expenses:dinner                                                                    $150.00
```

`lalloc` seeing this with a tax of 10% in place for this pay period on
`expenses:dinner` will generate the following allocations:

```
21-Apr-15 payback 'fancy dinner'   [assets:checking:reserved]                         $-150.00
                                   [allocations:checking:available:services]           $150.00

21-Apr-15 payback 'taxes'          [assets:checking:reserved]                          $-15.00
                                   [allocations:checking:savings:main]                  $15.00

```

Taxes like this can come and go by customizing directives in the job templates.

### Required Payback

It's also possible to 'cover' allocations from existing savings, rather than
using unallocated income. By enabling this behavior you can require the payback
of a purchase to savings and also restrict the maximum payment to spread that
reimbursement out over several pay periods. This is very useful for large,
emergency purchases that you can't weather from your usual unallocated income,
or for personal splurges. It also allows you to spread that purchase out over
time easily and your other spending will continue to be covered automatically
and even adjusted retroactively if you've scheduled the allocation to begin in
the past.

```
2021/01/07 * whoa, unexpected purchase
    : allocation:
    assets:checking                                                                  -$2000.00
    expenses:whoops                                                                   $2000.00
    [allocations:checking:savings:slow]                                              -$2000.00
    [assets:checking:reserved]                                                        $2000.00
```

This purchase will override the default schedule and instead slowly allocate
money back to cover the purchase and rebuild savings.

```
21-Jan-07 whoa, unexpected purchase            [allocations:checking:savings:slow]   $-2000.00
21-Jan-15 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
21-Feb-01 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
21-Feb-15 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
21-Feb-28 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
21-Mar-15 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
21-Apr-01 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $30.00
21-Apr-01 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $20.00
21-Apr-01 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
21-Apr-15 payback 'whoa, unexpected purchase'  [allocations:checking:savings:slow]      $50.00
```

### Intuitive Saving

You can also move money from `available` to any other allocation and the system
will compensate. This is another way to quickly squirrel money away or test
interesting financial scenarios.

## Example

Please see the example folder!
