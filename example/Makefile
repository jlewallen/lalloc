scenarios: paranoid_payments today example.json example.png

lalloc.g.ledger: example.ledger rules.ledger job.template.ledger generated.template.ledger ../lalloc.py
	$(MAKE) scenarios

example.json: lalloc.g.ledger
	../ledger.py -l example.ledger -o example.json

example.png: example.json
	../lapher.py -f example.json -o example.dot
	dot -Tpng example.dot -o example.png

today:
	../lalloc.py --today "2021/4/23"

paranoid_payments:
	../lalloc.py --today "2021/1/05" --paranoid
	../lalloc.py --today "2021/1/09" --paranoid
	../lalloc.py --today "2021/1/14" --paranoid
	../lalloc.py --today "2021/1/15" --paranoid
	../lalloc.py --today "2021/2/14" --paranoid
	../lalloc.py --today "2021/2/16" --paranoid
	../lalloc.py --today "2021/3/01" --paranoid
	../lalloc.py --today "2021/3/15" --paranoid
