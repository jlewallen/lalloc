default: check examples

check:
	mypy payoff.py
	mypy lalloc.py

examples:
	cd example && make

.PHONY: examples
