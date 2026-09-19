# Contributing

## Setup

```bash
make setup      # venv + llmkit (editable) + deps
make test       # 54 core tests, no GPU needed
make lint       # ruff
make no-gpu     # every analysis that needs no server at all
```

## The one rule that matters

**Do not add a number to this repo that you cannot reproduce.**

Every measurement carries provenance: git SHA and dirty flag, host and GPU
identity, the model the server actually reported serving, engine flags, the
workload fingerprint, the SLO, and whether token counts were exact or
estimated. `RunMeta.validate()` enforces this and
`projects/15-public-teardown/teardown.py verify` blocks publication of runs
that fail it.

If you produce a number from the simulator, it must be tagged
`simulated: true`. Charts from simulated runs are watermarked automatically by
`llmkit.report`. Update [STATUS.md](STATUS.md) when you execute something that
was previously unverified.

## Where code goes

`packages/llmkit/` is the shared core. Anything used by more than one project
belongs there, and the import surface in `llmkit/__init__.py` is deliberately
small. Fifteen projects depend on it; a churny core makes them drift apart.

`projects/NN-name/` holds one project's CLI, config and README. A project
should contain orchestration and presentation, not reusable logic.

## Style

- `ruff.toml` pins the rule set so local and CI agree. Ignores are documented
  individually; if you add one, say why.
- Comments explain *why*, especially where a design choice exists to prevent a
  specific wrong result. "Reduce in fp32" is not useful; "reduce in fp32
  because an fp16 sum over 4096 terms loses several bits, and this is where
  quantized attention goes subtly wrong" is.
- Prefer a failing test that encodes a bug you found over a comment describing
  it. Several tests in `packages/llmkit/tests/` exist because the behaviour
  they assert was once wrong.

## Testing

```bash
pytest packages/llmkit/tests -v          # core, no GPU
pytest projects/07-triton-kernels -v     # CUDA tests self-skip without a GPU
```

GPU tests must skip cleanly on machines without a GPU. A skipped test is
honest; a GPU test that silently passes on CPU is not.

## GPU work

Most of this repo has never run on real hardware (see [STATUS.md](STATUS.md)).
If you have access to one, the highest-value contribution is running an
existing harness against real vLLM and recording the result with full
provenance. Every harness already takes `--base-url`, so this is a flag change
rather than new code.
