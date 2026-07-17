PYTHON ?= python3

# GB300 training image (docker/Dockerfile.train) — multi-arch build knobs.
# The cluster is linux/arm64 (Grace) and so is an Apple-Silicon Mac; an x86 box or CI
# is linux/amd64. The NGC base is published for both, so buildx targets either from any
# host (native is fast; the other arch runs under QEMU — slow but portable).
TRAIN_IMAGE ?= igc-train
NGC_TAG     ?= 26.03-py3
TRAIN_TAG   ?= ngc$(NGC_TAG)
PLATFORM    ?= linux/arm64
SAVE        ?= /models/images/$(TRAIN_IMAGE)-$(TRAIN_TAG).tar.zst

.PHONY: help gate test lint perf coverage metrics profile profile-rl profile-dataset-cuda \
        docker-test docker-push \
        train-image train-image-arm64 train-image-amd64 train-image-multi train-image-save clean

help: ## List available targets and their descriptions
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

gate: ## Lint and run the offline test gate (what CI runs)
	bash -c 'set -o pipefail; ruff check igc tests && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 $(PYTHON) -m pytest -q'

test: gate ## Alias for gate

lint: ## Run the ruff linter over igc and tests
	ruff check igc tests

perf: ## Run the performance budget tripwires (pytest -m perf)
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 $(PYTHON) -m pytest -m perf -q

coverage: ## Run the offline gate with coverage (term + coverage.xml + coverage.json)
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 $(PYTHON) -m pytest -q --cov=igc --cov-report=term --cov-report=xml --cov-report=json

metrics: coverage ## Write a per-commit metrics snapshot (tests + coverage + hot-path timings) to metrics.json
	$(PYTHON) scripts/metrics_snapshot.py --coverage-json coverage.json --out metrics.json

profile: ## Profile all hot paths with cProfile critical sections
	$(PYTHON) scripts/bench_hot_paths.py --profile

profile-rl: ## Profile only the RL training hot paths
	$(PYTHON) scripts/bench_hot_paths.py --section rl --profile

profile-dataset-cuda: ## Profile Redfish corpus/tokenizer/DataLoader/H2D/CUDA train step
	@test "$${PROFILE_DATASET_ARGS}" || ( \
		echo "ERROR: set PROFILE_DATASET_ARGS='--corpus-dir /path/to/corpus"; \
		echo "       --output-dir /models/igc/profile_runs/<run_id> ...'"; \
		exit 2; \
	)
	$(PYTHON) scripts/profile_dataset_to_cuda.py $${PROFILE_DATASET_ARGS}

docker-test: ## Build the CPU test image and run the gate inside it
	docker build -f docker/Dockerfile.test -t igc-test:cpu . && docker run --rm igc-test:cpu python -m pytest -q

docker-push: ## Build and push the CPU test image (requires DOCKER_REPO=<user>/igc-test)
	@test "$${DOCKER_REPO}" || (echo "ERROR: set DOCKER_REPO, e.g. DOCKER_REPO=youruser/igc-test make docker-push"; exit 1)
	docker build -f docker/Dockerfile.test -t $${DOCKER_REPO}:latest .
	docker push $${DOCKER_REPO}:latest

train-image: ## Build the GB300 training image for PLATFORM (default linux/arm64), loaded into local docker
	docker buildx build --platform $(PLATFORM) --build-arg NGC_TAG=$(NGC_TAG) \
	    -f docker/Dockerfile.train -t $(TRAIN_IMAGE):$(TRAIN_TAG) --load .

train-image-arm64: ## Build the training image for linux/arm64 (the GB300 / Grace cluster arch)
	$(MAKE) train-image PLATFORM=linux/arm64

train-image-amd64: ## Build the training image for linux/amd64 (x86; runs under QEMU on an arm64 host, slow)
	$(MAKE) train-image PLATFORM=linux/amd64

train-image-multi: ## Build+push a multi-arch arm64+amd64 manifest (set REGISTRY=<user>/igc-train)
	@test "$(REGISTRY)" || (echo "ERROR: set REGISTRY, e.g. REGISTRY=youruser/igc-train make train-image-multi"; exit 1)
	docker buildx build --platform linux/arm64,linux/amd64 --build-arg NGC_TAG=$(NGC_TAG) \
	    -f docker/Dockerfile.train -t $(REGISTRY):$(TRAIN_TAG) --push .

train-image-save: ## Save the built training image to a zstd tarball (override SAVE=/path) for offline docker load
	@mkdir -p $(dir $(SAVE))
	docker save $(TRAIN_IMAGE):$(TRAIN_TAG) | zstd -T0 > $(SAVE)
	@echo "saved -> $(SAVE)"

clean: ## Remove Python/pytest/ruff cache directories
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
