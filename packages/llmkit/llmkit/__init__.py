"""llmkit: the shared core of the inference stack.

Fifteen projects sit on top of this. The import surface is deliberately small
and stable, because a churny core would let them drift apart and the whole
point is that a measurement from the real client and one from the simulator are
the same type, analysed by the same code.

Submodules not re-exported here (import them directly):
    llmkit.simulator   paged KV cache, scheduler, roofline cost model
    llmkit.quant       quantization schemes and their predicted effects
    llmkit.specdec     speculative decoding arithmetic
    llmkit.cluster     colocated vs disaggregated topologies
    llmkit.autoscale   autoscaling policies and their simulation
    llmkit.costs       $/token, MFU and MBU
    llmkit.gateway     rate limits, retry budgets, degradation chains
    llmkit.chaos       fault injection and SLO burn accounting
"""

from . import report
from .client import EndpointConfig, Request, StreamingClient, probe
from .gpus import GPUS, GPUSpec, get_gpu
from .kvcache import MemoryBudget, ServingConfig, compute_budget, required_gpus
from .load import concurrency_ladder, run_closed_loop, run_open_loop
from .metrics import Dist, RunSummary, find_knee, percentile, summarize, sweep_table
from .modelspec import REGISTRY, ModelSpec, get_model
from .prom import parse_labeled, parse_prometheus, scrape_sync
from .results import Run, RunMeta, new_run_id, results_dir
from .routing import Replica, make_router, prefix_key
from .types import (
    SLO,
    FinishReason,
    Phase,
    RequestRecord,
    TokenEvent,
    now_ns,
)
from .workload import PRESETS, LengthSpec, WorkloadGenerator, WorkloadSpec, make_text

__all__ = [
    # measurement
    "FinishReason", "Phase", "RequestRecord", "SLO", "TokenEvent", "now_ns",
    "Dist", "RunSummary", "find_knee", "percentile", "summarize", "sweep_table",
    "EndpointConfig", "Request", "StreamingClient", "probe",
    "concurrency_ladder", "run_closed_loop", "run_open_loop",
    "PRESETS", "LengthSpec", "WorkloadGenerator", "WorkloadSpec", "make_text",
    "Run", "RunMeta", "new_run_id", "results_dir",
    # modelling
    "ModelSpec", "REGISTRY", "get_model",
    "GPUS", "GPUSpec", "get_gpu",
    "MemoryBudget", "ServingConfig", "compute_budget", "required_gpus",
    # serving infrastructure
    "parse_prometheus", "parse_labeled", "scrape_sync",
    "Replica", "make_router", "prefix_key",
    "report",
]
__version__ = "0.1.0"
