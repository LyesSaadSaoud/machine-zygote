# Machine Zygote — Paper 1
PY ?= python3

.PHONY: all run analyze figures test smoke clean distclean help

help:
	@echo "make all       run everything, analyse, and make figures"
	@echo "make run       run the five experiments"
	@echo "make analyze   compute statistics -> outputs/paper1_summary.json + PAPER1_RESULTS.md"
	@echo "make figures   render Fig. P1-1 .. P1-8 as PDF and 600 dpi PNG"
	@echo "make test      run the test suite (no pytest required)"
	@echo "make smoke     fast end-to-end check on reduced sample sizes"
	@echo "make clean     delete generated outputs (keeps directory structure)"

all: run analyze figures

run:
	$(PY) scripts/run_all.py

analyze:
	$(PY) scripts/analyze_all.py

figures:
	$(PY) scripts/make_figures.py

test:
	$(PY) tests/run_tests.py

smoke:
	$(PY) scripts/run_all.py --smoke-test
	$(PY) scripts/analyze_all.py --smoke-test
	$(PY) scripts/make_figures.py --smoke-test

clean:
	rm -f outputs/raw/* outputs/processed/* outputs/figures/* outputs/logs/*
	rm -f outputs/paper1_summary*.json outputs/PAPER1_RESULTS*.md

distclean: clean
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
