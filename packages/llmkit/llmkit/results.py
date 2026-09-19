"""Run persistence with enough provenance to be reproducible.

A benchmark result that does not record what produced it is an anecdote. Every
run written here carries: git SHA and dirty flag, hostware, engine identity as
reported by the server itself (not as the operator believes it to be), the
exact workload fingerprint, the SLO, and whether token counts were exact or
estimated. Project 15 refuses to publish any run missing these.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import RunSummary
from .types import RequestRecord

SCHEMA_VERSION = 2


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    # GPU identity, when there is one. Absence is recorded explicitly so a
    # reader never has to guess whether a run was on CPU.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["gpus"] = [l.strip() for l in out.stdout.strip().splitlines()]
    except Exception:
        pass
    info.setdefault("gpus", [])
    return info


def git_info() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"sha": sha, "dirty": dirty, "branch": _git("rev-parse", "--abbrev-ref", "HEAD")}


@dataclass
class RunMeta:
    """Everything a reader needs to trust or reproduce the numbers."""

    run_id: str = ""
    project: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    created_by: str = field(default_factory=lambda: getpass.getuser())

    engine: str = "unknown"          # vllm | sglang | simulator | provider name
    engine_version: str = ""
    model: str = ""
    served_model_id: str = ""        # what /v1/models actually reported
    engine_flags: dict[str, Any] = field(default_factory=dict)
    endpoint: str = ""

    workload: str = ""
    workload_fingerprint: str = ""
    driver: str = ""                 # closed_loop | open_loop
    slo: dict[str, Any] = field(default_factory=dict)
    warmup_s: float = 0.0

    token_source: str = "unknown"    # server_usage | exact | heuristic
    simulated: bool = False          # loud flag: no real hardware was involved

    host: dict[str, Any] = field(default_factory=host_info)
    git: dict[str, Any] = field(default_factory=git_info)
    schema_version: int = SCHEMA_VERSION
    notes: str = ""

    def validate(self) -> list[str]:
        """Publication gate. Returns reasons this run must not be published."""
        problems: list[str] = []
        if not self.model:
            problems.append("no model recorded")
        if self.engine == "unknown":
            problems.append("engine not identified")
        if not self.git.get("sha"):
            problems.append("no git SHA (code state unknown)")
        if self.git.get("dirty"):
            problems.append("working tree dirty at run time (code state not reproducible)")
        if self.token_source == "heuristic":
            problems.append(
                "token counts are estimated, not server-reported; "
                "throughput figures carry that error"
            )
        if not self.simulated and not self.host.get("gpus"):
            problems.append("run claims to be real but no GPU was detected on the host")
        return problems


@dataclass
class Run:
    meta: RunMeta = field(default_factory=RunMeta)
    summaries: list[RunSummary] = field(default_factory=list)
    records: list[RequestRecord] = field(default_factory=list)

    def to_json(self, include_records: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "meta": asdict(self.meta),
            "summaries": [s.as_dict() for s in self.summaries],
        }
        if include_records:
            d["records"] = [r.to_row() for r in self.records]
        return d

    def save(self, path: str | Path, *, records: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(include_records=False), indent=2, default=str))
        if records and self.records:
            self._save_records(path.with_suffix(".records.parquet"))
        return path

    def _save_records(self, path: Path) -> None:
        rows = [r.to_row() for r in self.records]
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            pq.write_table(pa.Table.from_pylist(rows), path)
        except Exception:
            # Parquet is an optimisation, not a requirement.
            path.with_suffix(".jsonl").write_text(
                "\n".join(json.dumps(r, default=str) for r in rows)
            )

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())


def results_dir() -> Path:
    d = Path(os.environ.get("LLMKIT_RESULTS", "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
