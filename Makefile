.PHONY: help install lint check test clean

UV := $(shell command -v uv 2> /dev/null)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install dependencies"
	@echo "  lint      Format and autofix, then type-check"
	@echo "  check     Verify formatting and lint (no changes), then type-check"
	@echo "  test      Run tests"
	@echo "  clean     Remove caches and build artifacts"

install:
	@echo ">>> Installing dependencies"
	@$(UV) sync

lint:
	@echo ">>> Formatting and autofixing"
	@$(UV) run ruff format .
	@$(UV) run ruff check . --fix
	@echo ">>> Type checking"
	@$(UV) run mypy src/chap_ar
	@$(UV) run pyright

check:
	@echo ">>> Checking formatting and lint"
	@$(UV) run ruff format --check .
	@$(UV) run ruff check .
	@echo ">>> Type checking"
	@$(UV) run mypy src/chap_ar
	@$(UV) run pyright

test:
	@echo ">>> Running tests"
	@$(UV) run pytest -q

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist build *.egg-info

.DEFAULT_GOAL := help
