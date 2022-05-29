default: check examples

check:
	mypy payoff.py
	mypy lalloc.py
	mypy rebal.py
	mypy lapher.py

examples:
	cd example && make

.PHONY: examples
