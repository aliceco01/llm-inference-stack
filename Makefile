PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
RESULTS ?= results
MODEL ?= llama-3.1-8b
GPU ?= h100-sxm
PORT ?= 8000
BASE_URL ?= http://127.0.0.1:$(PORT)

.PHONY: help
help:
	@echo "setup          create the venv and install llmkit"
	@echo "test           run the core test suite (no GPU needed)"
	@echo "lint           ruff check"
	@echo "serve          start the simulated inference server (project 01)"
	@echo "bench          concurrency sweep against BASE_URL (project 02)"
	@echo "kv             KV cache plan for MODEL on GPU (project 03)"
	@echo "no-gpu         every analysis that needs no server at all"
	@echo "clean          remove venv, results and caches"
	@echo ""
	@echo "vars: MODEL=$(MODEL) GPU=$(GPU) PORT=$(PORT) RESULTS=$(RESULTS)"

.venv:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip

.PHONY: setup
setup: .venv
	$(PIP) install -q -e packages/llmkit
	$(PIP) install -q fastapi "uvicorn[standard]" httpx numpy matplotlib pyyaml \
	                  pytest pytest-asyncio pyarrow
	@echo "ready. try: make no-gpu"

.PHONY: test
test:
	$(PY) -m pytest packages/llmkit/tests -q

.PHONY: test-kernels
test-kernels:
	cd projects/07-triton-kernels && ../../$(PY) -m pytest test_kernels.py -q

.PHONY: lint
lint:
	.venv/bin/ruff check packages/llmkit projects || true

.PHONY: compile
compile:
	@$(PY) -m compileall -q packages/llmkit/llmkit projects && echo "syntax OK"

# --- running things --------------------------------------------------------
.PHONY: serve
serve:
	$(PY) projects/01-inference-server/mock_server.py \
	  --model $(MODEL) --gpu $(GPU) --port $(PORT) --max-model-len 4096

.PHONY: bench
bench:
	$(PY) projects/02-benchmark-suite/bench.py sweep \
	  --base-url $(BASE_URL) --model $(MODEL) \
	  --concurrency 1,2,4,8,16,32,64 --workload balanced \
	  --engine simulator --simulated --out $(RESULTS)

.PHONY: kv
kv:
	$(PY) projects/03-kv-calculator/kvcalc.py plan --model $(MODEL) --gpu $(GPU) \
	  --max-model-len 131072

# --- everything that needs no GPU and no server ---------------------------
.PHONY: no-gpu
no-gpu:
	@echo "=== 03 KV cache budget ==="
	@$(PY) projects/03-kv-calculator/kvcalc.py plan --model $(MODEL) --gpu $(GPU) --max-model-len 131072
	@echo "\n=== 03 where TP stops buying KV ==="
	@$(PY) projects/03-kv-calculator/kvcalc.py tp-scan --model llama-3.1-70b --gpu $(GPU) --seq-len 32768 --max-model-len 32768
	@echo "\n=== 05 quantization prediction ==="
	@$(PY) projects/05-quantization-lab/quantlab.py predict --model $(MODEL) --gpu $(GPU)
	@echo "\n=== 06 speculative decoding plan ==="
	@$(PY) projects/06-speculative-decoding/specdec.py plan --batch-size 1
	@$(PY) projects/06-speculative-decoding/specdec.py plan --batch-size 64
	@echo "\n=== 07 kernel algorithm validation ==="
	@$(PY) projects/07-triton-kernels/reference.py verify
	@echo "\n=== 07 memory traffic model ==="
	@$(PY) projects/07-triton-kernels/reference.py traffic
	@echo "\n=== 09 fragmentation: paged vs contiguous ==="
	@$(PY) projects/09-paged-attention/experiment.py fragmentation --out $(RESULTS)
	@echo "\n=== 10 disaggregation: where it wins ==="
	@$(PY) projects/10-disaggregated/experiment.py sweep --out $(RESULTS)
	@echo "\n=== 11 autoscaling signals ==="
	@$(PY) projects/11-autoscaler/autoscaler.py signals --out $(RESULTS)

.PHONY: experiments
experiments:
	$(PY) projects/08-chunked-prefill/experiment.py policy --out $(RESULTS)
	$(PY) projects/08-chunked-prefill/experiment.py timeline --out $(RESULTS)
	$(PY) projects/09-paged-attention/experiment.py all --out $(RESULTS)
	$(PY) projects/11-autoscaler/autoscaler.py policies --out $(RESULTS)

.PHONY: clean
clean:
	rm -rf .venv $(RESULTS) site
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +
