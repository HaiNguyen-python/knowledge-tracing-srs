.PHONY: install test lint typecheck format report train serve check

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

# The same gate as CI
check: lint typecheck
	pytest --cov-fail-under=80

# Synthetic study: docs/results.md and docs/img/*.png (about 2 minutes on a laptop)
report:
	ktrace report

train:
	ktrace train

serve: train
	ktrace serve
