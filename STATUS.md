# Verification status

This repo argues that a benchmark you cannot reproduce is an anecdote. It would
be inconsistent to then ship code without saying which parts have actually been
executed. This file says so, precisely.

Last updated: 2026-09-19. Development machine: Apple M1, 8 GB RAM, **no NVIDIA
GPU**.

## Verified by execution

| Component | What was run |
|---|---|
| `llmkit` core | 54 unit tests, all passing |
| Whole tree | `compileall` clean, `ruff check` clean |
| 01 inference server | serves real SSE over HTTP, role-only first frame, server `usage`, vLLM-named `/metrics`, `/sim/reset`, `/sim/fault` |
| 02 benchmark suite | `single`, `sweep` (closed loop) against the simulator, charts and report generated |
| 03 KV calculator | `plan`, `tp-scan`, `table`, `oom`, `config`; `kvmon.py` against a live server |
| 04 prefix proxy | proxy serves and routes; `experiment.py` ran, then was rewritten after it exposed cache contamination (see below) |
| 05 quantization lab | `predict`, `plan` |
| 06 speculative decoding | `plan`, `curve`, `simulate` |
| 07 Triton kernels | `reference.py verify` and `traffic`; 13 NumPy algorithm tests pass, 20 CUDA tests skip cleanly |
| 08 chunked prefill | `policy`, `timeline` |
| 09 PagedAttention | `fragmentation`, `blocksize`, `eviction`, `preemption`, `cow` |
| 10 disaggregated | `split`, `transfer`, `sweep`, `compare` |
| 11 autoscaler | `signals`, `policies`, `flapping`, `keda` |
| 12 cost dashboard | `mfu`, `breakeven` |
| 13 AI gateway | end to end against a live backend: routing headers, tier selection, `/stats`, breaker opening on an unreachable replica |
| 14 chaos suite | `list` |

## Not yet verified

These are written and reviewed but have never been executed, because they need
hardware or services this machine does not have. Treat them as unproven.

| Component | Needs |
|---|---|
| 07 `kernels.py`, `bench.py` | a CUDA GPU. The Triton kernels have never been compiled or run. |
| 05 `quantlab.py run`, `quality.py` | live vLLM endpoints serving different quantizations |
| 14 chaos scenarios end to end | a gateway plus several backends running concurrently |
| 15 `teardown.py run/verify/publish` | live endpoints for three configurations |
| 04 `experiment.py` full sweep | several concurrent servers; the first attempt was killed by the OOM killer on an 8 GB machine, which is why it now runs one switchable proxy instead of one per strategy |
| 12 `dashboard.py serve` | a live metrics source over time |
| All `k8s/*.yaml` | a Kubernetes cluster. Never applied anywhere. |
| `serve_vllm.sh`, `serve_sglang.sh` | real vLLM or SGLang. Never executed. |

## Which numbers in the docs are real

Numbers quoted in the READMEs come from one of three places, and each is
labelled where it appears:

1. **Measured on this machine against the simulator.** Real HTTP, real timing,
   simulated engine. The TTFT comparison (43.5 ms vs a naive 15.7 ms), the
   load-curve tables, and the scheduler timeline figure are all of this kind.
2. **Computed analytically.** The KV cache budgets, the memory-traffic model in
   project 07, the speculative-decoding acceptance arithmetic, and the
   consistent-hash rebalancing measurements are deterministic outputs of the
   code, not timing measurements.
3. **Cited as representative, not measured.** Acceptance rates per draft
   strategy in project 06 and GPU list prices in project 12 are starting points
   for ranking options. They are flagged as such at the point of use.

**No number in this repo was measured on real GPU hardware.** Every chart from
a simulated run is watermarked `SIMULATED`, and project 15's publication gate
blocks any run whose provenance is incomplete.

## Known issues found during development

Kept here because they are more informative than a clean history would be.

- **Timestamp sentinel.** `0` is a legal value on the simulator's virtual
  clock, so using falsiness to mean "unset" silently NaN'd every simulated
  metric. Fixed with an explicit `-1` sentinel; there is a regression test.
- **Token release timing.** The real-time server published a step's tokens at
  the *start* of its wall-clock delay rather than the end, under-reporting TTFT
  by exactly one scheduler step.
- **Benchmark cross-contamination.** The prefix-routing experiment ran four
  strategies against shared backends. The `unique` control workload, which must
  show a 0% cache hit rate, showed 33-57% for every strategy after the first:
  they were inheriting a warm cache. Fixed with `/sim/reset` between strategies.
- **Warmup scaling.** Fixed 4-request warmup left the startup transient in the
  sample at high concurrency and produced a non-monotonic load curve that looked
  like engine pathology. Warmup now scales to one full concurrency wave.
- **Gateway health asymmetry.** The health loop only ever marked backends
  healthy, never unhealthy, so an unreachable replica reported `available: true`
  until a user request found it. Probe failures now count toward the breaker,
  with a higher threshold than request failures.
- **Module-scope `importorskip` skipped a whole test file.** The kernel tests
  used `pytest.importorskip("torch")` at module scope, which raises during
  collection and skips *every* test in the file, including the thirteen NumPy
  algorithm tests that need no GPU and exist precisely so the algorithms can be
  validated on a laptop. They had never run. CI caught it as exit code 5, "no
  tests collected". Replaced with a lazy `skipif` so the NumPy tests always run
  and only the CUDA tests skip.

## Next steps, in value order

1. **Rent an H100 for a few hours.** Every harness already points at a real
   endpoint through one flag. This converts roughly nine projects from
   simulated to measured and is by far the highest-value action.
2. Run `projects/15-public-teardown` against three real configurations and
   publish the result through the gate.
3. Run the Triton benchmarks and record achieved bandwidth against device peak.
4. Apply the k8s manifests to a real cluster, even a single-node one, to
   validate them.
