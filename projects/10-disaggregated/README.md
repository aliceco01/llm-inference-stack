# 10 - Disaggregated Prefill/Decode Cluster

> This is the architecture behind every frontier serving stack.

Splits prefill and decode onto separate GPU pools and measures the throughput
and latency deltas, including the KV transfer cost that decides whether the
split is worth making.

```bash
./experiment.py split    --gpus 8 --input-len 4096 --output-len 512
./experiment.py compare  --gpus 8 --out results/
./experiment.py sweep    --out results/      # where disaggregation actually wins
./experiment.py transfer --out results/      # KV transfer cost model
```

## Why split the phases

Prefill is compute bound; decode is memory-bandwidth bound. Co-locating them
means two workloads with opposite resource profiles share one scheduler and one
machine. Splitting buys three things:

1. **No interference.** Decode never stalls behind a prefill step, without
   needing chunked prefill (project 08) to mitigate it.
2. **Independent scaling.** Prefill demand tracks input-token rate; decode
   demand tracks concurrent generations. Those move independently, so any fixed
   ratio inside a single replica is wrong most of the time.
3. **Heterogeneous hardware.** Prefill wants FLOPs, decode wants HBM bandwidth
   and capacity. Disaggregation lets you buy each separately, for example
   prefill on compute-dense parts and decode on high-capacity ones.

## What it costs, which is the whole question

The KV cache produced during prefill must reach the decode worker. That is not
a small payload:

| Model | Context | KV to transfer | Over 400G RDMA (~50 GB/s) |
|---|---|---|---|
| Llama-3.1-8B | 8k | 1.0 GiB | ~21 ms |
| Llama-3.1-8B | 32k | 4.0 GiB | ~82 ms |
| Llama-3.1-70B | 8k | 2.5 GiB | ~52 ms |

(`kv_bytes_per_token` from project 03: 128 KiB/token for 8B, 320 KiB/token for
70B at fp16. Run `./experiment.py transfer` for the full matrix across
interconnects.)

Every one of those milliseconds lands directly in TTFT. So disaggregation pays
only when **interference costs more than the transfer**, which makes it a
workload question, not an architecture preference.

`./experiment.py sweep` maps that boundary across workload shapes. The pattern:
long prompts with substantial generations win, short-prompt/short-generation
traffic loses because the transfer dominates a cheap prefill, and well-balanced
mid-size workloads are close to a wash.

**Disaggregation is not a universal upgrade.** It is a fleet-shaping tool that
pays at scale and on skewed workloads. A two-GPU deployment almost certainly
should not use it, and saying so is more useful than the usual framing.

### Three mitigations that change the arithmetic

- **Layer-wise streaming.** Send each layer's KV as it is computed instead of
  the whole cache at the end, overlapping transfer with prefill. This is what
  makes long-context disaggregation viable at all.
- **fp8 KV cache.** Halves bytes on the wire as well as in memory, which is a
  second reason project 05 treats KV quantization as its own axis.
- **Prefix-aware placement.** If the decode worker already holds the prefix,
  only the suffix needs transferring. Project 04's routing applies here.

And note that intra-node disaggregation over NVLink (~400 GB/s) is a completely
different proposition from cross-node over RDMA. At NVLink speeds the transfer
is close to free, which is why intra-node splits are far easier to justify.

## Sizing the pools

`./experiment.py split` computes the ratio analytically: prefill work per
request scales with input length, decode work with output length times per-step
cost at the achieved batch size, and the pool ratio should match the ratio of
those totals.

`compare` then runs the suggested split plus its neighbours over an identical
request stream, and reports per-pool utilisation. Utilisation is the signal that
tells you the split is wrong: a prefill pool at 40% next to a saturated decode
pool means move GPUs across, and the simulator emits that note explicitly.

## Kubernetes

**`k8s/disaggregated.yaml`** runs the pools as separate Deployments, which is
the point: they scale on different signals via separate KEDA `ScaledObject`s
(prompt-token rate for prefill, running sequences for decode). Three details
that matter and are easy to miss:

- **Prefill needs little KV cache** (`--gpu-memory-utilization=0.60`). It holds
  a prompt only long enough to compute and ship its KV. Giving prefill workers
  90% of VRAM for cache starves the pool that actually needs it.
- **Pod affinity between the pools.** The KV transfer crosses that link on
  every request; a rack or zone hop can cost more than the prefill being
  offloaded.
- **Asymmetric drain.** Decode holds long-lived streams and needs a real
  `preStop` and grace period; prefill holds a request for milliseconds.

**`k8s/multinode-lws.yaml`** handles the case where the model does not fit on
one node. Tensor parallelism across nodes makes a replica a *group* of pods that
must be scheduled, started, restarted and scaled together, and a Deployment
cannot express that: it treats pods as interchangeable, which a TP group is not.
LeaderWorkerSet makes the group the unit, with `RecreateGroupOnPodRestart`
because a TP group missing a rank is dead rather than degraded. It is paired
with a `PodGroup` for gang scheduling, without which partially-placed groups
hold GPUs while waiting for their remaining nodes and can deadlock each other
under contention.

The `NCCL_IB_DISABLE=0` / `NCCL_NET_GDR_LEVEL=5` environment is load-bearing:
misconfigured cross-node NCCL does not error, it silently falls back to TCP and
the replica runs several times slower with no obvious symptom.

## Real implementations

The production stacks that do this are vLLM's disaggregated serving with
NixlConnector, NVIDIA Dynamo, llm-d, and Mooncake, with DistServe and
Splitwise as the research antecedents. The manifests here use vLLM's
`--kv-transfer-config` with `kv_producer` / `kv_consumer` roles, which is the
mechanism those stacks build on.
