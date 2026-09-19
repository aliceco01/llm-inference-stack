"""Accelerator specs used for memory budgets and the roofline cost model.

Numbers are vendor dense (non-sparse) figures. Achieved bandwidth on real
kernels is typically 70-85% of peak and achieved FLOPs 50-75%, so the
simulator applies explicit efficiency factors rather than pretending peak is
attainable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    vram_gb: float               # marketing GB (1e9), not GiB
    bandwidth_gb_s: float        # peak HBM bandwidth
    bf16_tflops: float           # dense
    fp8_tflops: float = 0.0
    nvlink_gb_s: float = 0.0     # per-GPU bidirectional, 0 = PCIe only
    usd_per_hour: float = 0.0    # representative on-demand list price

    @property
    def vram_bytes(self) -> float:
        return self.vram_gb * 1e9

    @property
    def vram_gib(self) -> float:
        return self.vram_bytes / 1024 ** 3

    def tflops(self, dtype: str = "bf16") -> float:
        if dtype in ("fp8", "int8") and self.fp8_tflops:
            return self.fp8_tflops
        if dtype in ("int4", "fp4") and self.fp8_tflops:
            return self.fp8_tflops * 2
        return self.bf16_tflops


GPUS: dict[str, GPUSpec] = {
    g.name: g
    for g in [
        GPUSpec("a10g", 24, 600, 125, 250, 0, 1.01),
        GPUSpec("l4", 24, 300, 121, 242, 0, 0.80),
        GPUSpec("l40s", 48, 864, 362, 733, 0, 1.96),
        GPUSpec("rtx4090", 24, 1008, 165, 330, 0, 0.44),
        GPUSpec("a100-40gb", 40, 1555, 312, 0, 600, 3.67),
        GPUSpec("a100-80gb", 80, 2039, 312, 0, 600, 4.10),
        GPUSpec("h100-pcie", 80, 2000, 756, 1513, 0, 5.50),
        GPUSpec("h100-sxm", 80, 3350, 989, 1979, 900, 6.75),
        GPUSpec("h200", 141, 4800, 989, 1979, 900, 8.50),
        GPUSpec("b200", 180, 8000, 2250, 4500, 1800, 12.00),
        GPUSpec("mi300x", 192, 5300, 1307, 2614, 0, 5.90),
    ]
}


def get_gpu(name: str) -> GPUSpec:
    key = name.lower().strip().replace("_", "-")
    if key in GPUS:
        return GPUS[key]
    for k in GPUS:
        if k in key or key in k:
            return GPUS[k]
    raise KeyError(f"unknown GPU {name!r}. Known: {', '.join(sorted(GPUS))}")
