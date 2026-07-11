PYTHON ?= python3

.PHONY: help gate test lint perf profile profile-rl docker-test docker-push clean

help: ## List available targets and their descriptions
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

gate: ## Lint and run the offline test gate (what CI runs)
	bash -c 'set -o pipefail; ruff check igc tests && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 $(PYTHON) -m pytest -q'

test: gate ## Alias for gate

lint: ## Run the ruff linter over igc and tests
	ruff check igc tests

perf: ## Run the performance budget tripwires (pytest -m perf)
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 $(PYTHON) -m pytest -m perf -q

profile: ## Profile all hot paths with cProfile critical sections
	$(PYTHON) scripts/bench_hot_paths.py --profile

profile-rl: ## Profile only the RL training hot paths
	$(PYTHON) scripts/bench_hot_paths.py --section rl --profile

docker-test: ## Build the CPU test image and run the gate inside it
	docker build -f docker/Dockerfile.test -t igc-test:cpu . && docker run --rm igc-test:cpu pytest -q

docker-push: ## Build and push the CPU test image (requires DOCKER_REPO=<user>/igc-test)
	@test "$${DOCKER_REPO}" || (echo "ERROR: set DOCKER_REPO, e.g. DOCKER_REPO=youruser/igc-test make docker-push"; exit 1)
	docker build -f docker/Dockerfile.test -t $${DOCKER_REPO}:latest .
	docker push $${DOCKER_REPO}:latest

clean: ## Remove Python/pytest/ruff cache directories
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
