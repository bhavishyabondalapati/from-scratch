# Usage:
#   make test              all tests, including slow training tests
#   make test-fast         skip tests marked @pytest.mark.slow (what CI runs)
#   make test-section N=3  one section (folders s3_*)
#   make test-topic T=s3_classical_ml/svm
#   make lint / make format

PYTEST ?= python -m pytest
SECTION_DIR = $(wildcard s$(N)_*)

.PHONY: test test-fast test-section test-topic lint format

test:
	$(PYTEST) -q

test-fast:
	$(PYTEST) -q -m "not slow"

test-section:
	@if [ -z "$(N)" ] || [ -z "$(SECTION_DIR)" ]; then echo "usage: make test-section N=<1-6>"; exit 1; fi
	$(PYTEST) -q $(SECTION_DIR)

test-topic:
	@if [ -z "$(T)" ]; then echo "usage: make test-topic T=<section/topic>"; exit 1; fi
	$(PYTEST) -q $(T)

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .
