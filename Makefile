.PHONY: help install sample-db spider test lint baseline splits tokens real-path optimize compare clean

help:
	@echo "Delta - self-improving text-to-SQL agent"
	@echo ""
	@echo "  make install     Create .venv and install the package (editable)"
	@echo "  make sample-db   Build the tiny offline SQLite fixture"
	@echo "  make spider      Download + checksum-verify the Spider benchmark (206 MB)"
	@echo "  make splits      Write data/splits.json (refuses to overwrite)"
	@echo "  make tokens      Measure Spider prompt token sizes (offline)"
	@echo "  make real-path   Live Groq headroom gate on 100 stratified examples"
	@echo "  make test        Run the test suite (no API calls, no network)"
	@echo "  make lint        Ruff check"
	@echo "  make baseline    Score the v0 prompt (use MOCK=1 for zero API calls)"
	@echo "  make optimize    Run the optimization loop"
	@echo "  make compare     Score all conditions on the held-out set"

install:
	uv venv
	uv pip install -e ".[dev]"
	@echo "Activate with: source .venv/bin/activate"

sample-db:
	.venv/bin/python scripts/make_sample_db.py

spider:
	.venv/bin/python scripts/download_spider.py

splits:
	.venv/bin/python scripts/make_splits.py

tokens:
	.venv/bin/python scripts/measure_tokens.py

real-path:
	.venv/bin/python scripts/run_real_path.py

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

baseline:
	.venv/bin/python scripts/run_baseline.py $(if $(MOCK),--mock,)

optimize:
	.venv/bin/python scripts/run_optimize.py $(if $(MOCK),--mock,)

compare:
	.venv/bin/python scripts/run_comparison.py $(if $(MOCK),--mock,)

clean:
	rm -rf runs/ .cache/ .pytest_cache/ .ruff_cache/
