# Qwen3-30B-A3B Pre-training Benchmark Report: Performance Optimization on B200 GPUs

**Date**: March 27, 2026
**Cluster**: 8x p6-b200.48xlarge (64x NVIDIA B200 SXM 192GB HBM3e)
**Model**: Qwen3-30B-A3B MoE (BF16)
**Software**: NeMo 25.11.01, Megatron-Bridge, NCCL 2.29.3

---

## 1. Executive Summary

This report documents the performance optimization of Qwen3-30B-A3B Mixture-of-Experts (MoE) pre-training on an 8-node Amazon SageMaker HyperPod cluster equipped with 64 NVIDIA B200 SXM GPUs (192GB HBM3e each). Starting from the dgxc-benchmarking B200 baseline profile, we achieved a **2.11x throughput improvement** -- from **232 to 489 MODEL TFLOPS/GPU** in BF16 precision.

### Key Results

| Configuration | TFLOPS/GPU | Step Time | Peak Memory (GB) | vs Baseline |
|---|---|---|---|---|
| B200 Baseline (PP=1, MBS=1, CUDA graphs) | 232 | 26.0s | 47.5 / 192 | -- |
| + MBS=2 (CUDA graphs + nsys profiling) | 429 | 14.0s | 71.6 / 192 | +85% |
| **+ MBS=4 (CUDA graphs)** | **489** | **12.3s** | **119.8 / 192** | **+111%** |
| + MBS=4, expanded CUDA graph scope (+attn) | 488 | 12.4s | 119.8 / 192 | +110% |

### What Worked

1. **Micro-batch size 1 to 4** (+111%): The B200's 192GB HBM3e provides sufficient headroom to quadruple the micro-batch size from the baseline, dramatically improving GPU arithmetic intensity. Each doubling of MBS roughly doubles the batch dimension in expert FFN GEMMs, improving tensor core utilization.

2. **CUDA graphs (moe_router + moe_preprocess scopes)**: Unlike the H200 where CUDA graphs caused OOM at PP=1, the B200's extra 51GB of memory accommodated both MBS=4 and CUDA graph capture simultaneously. CUDA graphs eliminated per-kernel CPU dispatch overhead for the many small MoE routing and preprocessing kernels across 48 layers.

### What Did Not Work

| Experiment | Result | Reason |
|---|---|---|
| MBS=8 | OOM | Would require ~168 GB activations, exceeding 178 GB usable |
| MBS=6 | Config error | GBS=4096 not divisible by MBS*DP (6*64=384) |
| CUDA graphs + `attn` scope | 488 TFLOPS (-0.2%) | 139s graph capture (vs 24s), no throughput benefit |
| MoE A2A overlap (MBS=4) | OOM | Deferred wgrad + double-buffered A2A adds ~47 GB |
| MoE A2A overlap (MBS=2) | 343 TFLOPS (-20%) | Overlap doubled comm volume; net exposed comm increased 37% |
| DeepEP flex dispatcher | 323 TFLOPS (-34%) | LD/ST comms inefficient on EFA; SM reservation hurts compute |
| HybridEP flex dispatcher | Fell back to AlltoAll | Megatron-Core in NeMo 25.11.01 lacks flex dispatcher module |

### Bottleneck Analysis

nsys profiling reveals the workload is **58% compute, 42% communication**. MoE Expert Parallel all-to-all (SendRecv) accounts for the majority of communication time at 23.3s per profiled window. Only 4% of communication is hidden behind compute, indicating poor overlap. Attempting to improve this with Megatron-Core's `moe_a2a_overlap` feature backfired -- it doubled the communication volume while only achieving 28% overlap, resulting in a 20% throughput regression (see Sections 4.5 and 6.4). The primary path to further BF16 optimization is FP8 precision, which would halve the A2A payload size.

---

## 2. Environment and Setup

### 2.1 Cluster Architecture

| Component | Specification |
|---|---|
| Platform | Amazon SageMaker HyperPod (us-west-2) |
| Instance type | ml.p6-b200.48xlarge |
| Compute nodes | 8 |
| GPUs per node | 8x NVIDIA B200 SXM 192GB HBM3e |
| Total GPUs | 64 |
| Usable GPU memory | 178.35 GiB per GPU |
| Intra-node interconnect | NVLink / NVSwitch (5th generation) |
| Inter-node network | Elastic Fabric Adapter (EFA) |
| Shared filesystem | Amazon FSx for Lustre (`/fsx/`, 38 TB) |
| Operating system | Ubuntu 22.04 |
| Workload manager | Slurm with Pyxis/Enroot container support |

### 2.2 Software Stack

| Component | Version |
|---|---|
| Container | `nvidia+nemo+25.11.01-efa-nccl29.sqsh` (EFA-upgraded) |
| NeMo Framework | 25.11.01 |
| Megatron-Bridge | Commit 4df8c97 (NVIDIA dgxc-benchmarking) |
| NCCL | 2.29.3 |
| AWS OFI NCCL plugin | Custom override at `/fsx/ubuntu/ofi-override/` |
| Benchmark framework | NVIDIA dgxc-benchmarking v25.12.02 |

### 2.3 AWS EFA Networking Patches

The B200 cluster required the same EFA fixes as the H200 cluster (see companion report), plus additional container mount configuration for the OFI override libraries. The key differences from the H200 setup:

1. **OFI override libraries**: The B200 cluster uses separately compiled libraries at `/fsx/ubuntu/ofi-override/` rather than baking them into the container. The `executors.py` patch mounts these directories into the container at runtime:
   - `aws-ofi-nccl/lib/` -- provides `libnccl-net-aws-ofi.so` and `libnccl-tuner-aws-ofi.so`
   - `efa/lib/` -- provides `libfabric.so.1.29.1`
   - NCCL 2.29.3 library at `/fsx/ubuntu/nccl29/libnccl.so.2.29.3`

2. **Environment variables**: Same EFA variables as H200 (`FI_PROVIDER=efa`, `NCCL_SOCKET_IFNAME`, `NCCL_BUFFSIZE=8388608`, etc.), plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `NCCL_GRAPH_REGISTER=0` for CUDA graph compatibility.

---

## 3. CUDA Graphs: What They Are and Why They Were Fundamental

### 3.1 The Problem: CPU Launch Overhead

In standard PyTorch execution, every GPU kernel (GEMM, activation, normalization, etc.) is dispatched individually from the CPU. Each dispatch involves CPU-side work: argument marshaling, driver calls, and synchronization checks. For large kernels that run for milliseconds, this overhead is negligible. But MoE models like Qwen3-30B-A3B have an unusual kernel profile: 128 experts per layer means each forward pass through a single MoE layer generates hundreds of small kernels -- expert routing, token permutation, per-expert FFN GEMMs (with only ~256 tokens per expert at MBS=1), and token unpermutation. Across 48 layers, forward and backward passes, and 16 gradient accumulation steps, a single training iteration launches millions of GPU kernels. The cumulative CPU dispatch overhead becomes significant.

### 3.2 How CUDA Graphs Work

CUDA graphs address this by recording a sequence of GPU operations into a static graph during a "capture" phase, then replaying that graph with a single CPU-side launch call. During capture, PyTorch executes the target code region normally, but instead of dispatching kernels immediately, the CUDA runtime records them into a graph structure. On subsequent iterations, the entire captured sequence -- potentially hundreds of kernels -- is replayed with one `cudaGraphLaunch()` call.

The key constraints are:

- **Static shapes**: All tensor shapes must be identical across replays. Dynamic shapes (like variable-length sequences or changing batch sizes) break graph replay.
- **Memory overhead**: The graph captures memory allocations as part of its structure. A private memory pool is reserved for graph execution, which adds to peak GPU memory usage.
- **Capture granularity**: Not all operations can be captured (e.g., CPU-dependent control flow, certain NCCL collectives). NeMo uses "scopes" to capture specific subgraphs rather than the entire training step.

### 3.3 CUDA Graph Scopes in NeMo

NeMo's Megatron-Core supports scoped CUDA graph capture through the `cuda_graph_scope` configuration. The available scopes for MoE models are:

| Scope | What It Captures | Kernels Covered |
|---|---|---|
| `moe_router` | Expert routing computation | Top-K gating, softmax, capacity factor |
| `moe_preprocess` | Token dispatch preprocessing | sort_chunks_by_map, permutation indices |
| `attn` | Full attention block | QKV projection, Flash Attention, output projection |

For Qwen3-30B-A3B, the B200 baseline profile uses `moe_router` and `moe_preprocess` scopes. These capture the many small MoE-specific kernels that are most affected by launch overhead, while leaving the larger GEMMs and NCCL collectives outside the graph.

### 3.4 Why CUDA Graphs Were Fundamental on B200

CUDA graphs were a key differentiator between B200 and H200 performance:

| Factor | H200 (141 GB) | B200 (192 GB) |
|---|---|---|
| Best MBS | 2 | 4 |
| Peak memory at best MBS | 119 GB (84% utilization) | 120 GB (62% utilization) |
| CUDA graph memory pool | OOM (+14.77 GB exceeds 141 GB) | Fits comfortably (72 GB headroom) |
| CUDA graphs at best config | **Not possible** | **Enabled** |
| Best TFLOPS/GPU | 315 (no graphs) | 489 (with graphs) |

On the H200, the best configuration (PP=1, MBS=2) used 119 GB of the 141 GB available. Attempting CUDA graph capture required an additional 14.77 GB private pool, pushing total memory past the 141 GB limit. The H200 achieved its best result *without* CUDA graphs.

On the B200, the same parallelism configuration with MBS=4 used 120 GB of 192 GB -- leaving 72 GB of headroom, more than enough for the CUDA graph memory pool. This meant the B200 could benefit from both increased MBS *and* CUDA graph optimizations simultaneously.

The impact of CUDA graphs is visible in the step time: each training step begins with two slow iterations during graph capture (66s and 65s for Job 198), then steady-state iterations run at full speed. The 24-second capture overhead for `moe_router` + `moe_preprocess` scopes is amortized over the remaining iterations.

---

## 4. Optimization Journey

### 4.1 Starting Point: B200 Baseline Profile

The dgxc-benchmarking repository includes a B200-specific profile for Qwen3-30B-A3B at 64 GPUs:

- **TP=1, PP=1, CP=1, EP=8** -- no pipeline parallelism (unlike the H100 profile which uses PP=2)
- **MBS=1, GBS=4096**
- **CUDA graphs**: `moe_router, moe_preprocess` scopes
- **ZeRO-3**: optimizer+gradient+parameter sharding across DP=64

**Baseline result**: 232 MODEL TFLOPS/GPU, 26.0s per iteration, 47.5 GB peak memory.

With only 47.5 GB of 192 GB utilized (25%), the B200 has substantial memory headroom for larger micro-batches.

### 4.2 Optimization 1: MBS 1 to 2 (+85%)

**Change**: `MBS=1` to `MBS=2`, all other parameters unchanged. nsys profiling enabled on steps 46-50.

**Result**: 429 MODEL TFLOPS/GPU (steady state without profiling), 14.0s per iteration, 71.6 GB peak memory.

Doubling MBS doubles the token count per micro-batch (4096 to 8192 tokens), which directly improves GEMM efficiency. With MBS=1, each expert processes ~32 tokens per micro-batch (4096 tokens / 128 experts). At MBS=2, this doubles to ~64 tokens, significantly increasing the M-dimension of the expert FFN GEMMs and improving tensor core saturation.

Memory increased by 24.1 GB (47.5 to 71.6 GB) -- the cost of doubled activation memory. Still well within the 192 GB capacity with 120 GB headroom.

### 4.3 Optimization 2: MBS 2 to 4 (+14% additional, +111% total)

**Change**: `MBS=2` to `MBS=4`.

**Result**: 489 MODEL TFLOPS/GPU, 12.33s per iteration, 119.8 GB peak memory.

The second doubling of MBS provides a further 14% gain, following the same principle of improved arithmetic intensity. Each expert now processes ~128 tokens per micro-batch, and the gradient accumulation steps decrease from 32 to 16 (GBS=4096 / (MBS=4 * DP=64) = 16).

Memory increased by 48.2 GB (71.6 to 119.8 GB). At 62% utilization of 192 GB, the B200 still has 72 GB of headroom -- enough for both the model and CUDA graph capture pools.

**Why MBS=4 works on B200 but failed on H200**: The H200's 141 GB capacity could not accommodate MBS=4 regardless of parallelism configuration. With PP=2, MBS=4 OOMed at 130-141 GB. With PP=1, the base memory footprint of 119 GB at MBS=2 left no room to double activations again. The B200's additional 51 GB of HBM3e is the decisive factor.

### 4.4 Failed Attempts

**MBS=8** (Job 200): OOM. Extrapolating the MBS=4 memory of 120 GB, MBS=8 would require ~168 GB -- within the 192 GB physical capacity but exceeding the ~178 GB usable after OS/driver allocations.

**MBS=6** (Job 201): GBS divisibility error. With DP=64, `GBS / (MBS * DP) = 4096 / (6 * 64) = 10.67` -- not an integer, so Megatron rejects the configuration.

**CUDA graphs with `attn` scope** (Job 202): Adding the attention scope to the existing `moe_router` + `moe_preprocess` scopes. Graph capture time increased from 24s to 139s (5.8x longer), but steady-state performance was 488 TFLOPS -- a marginal 0.2% regression from 489. The attention GEMMs are large enough that kernel launch overhead is negligible; the additional graph capture complexity adds overhead without reducing any meaningful bottleneck.

### 4.5 Optimization 3 (Failed): MoE A2A Communication Overlap

The nsys analysis of Job 198 showed that MoE all-to-all SendRecv accounts for 23.3s of the 80.8s profiled window (29%), with only 4% of communication hidden behind compute. This poor overlap suggested that enabling Megatron-Core's `moe_a2a_overlap` feature could yield significant gains by pipelining A2A communication with expert computation.

The `moe_a2a_overlap` feature (set via `--moe_a2a_overlap true` in `setup_experiment.py`) activates two Megatron-Core flags:
- `overlap_moe_expert_parallel_comm=True` -- pipelines A2A dispatch/combine with expert FFN computation
- `delay_wgrad_compute=True` -- defers weight gradient GEMMs to overlap with the A2A combine phase

**Job 228 (MBS=4 + overlap): OOM**

Enabling overlap at MBS=4 immediately OOMed at 167+ GB, exceeding the ~178 GB usable. The `delay_wgrad_compute` mechanism requires buffering activation tensors for deferred backward computation, and `overlap_moe_expert_parallel_comm` requires double-buffered A2A communication tensors. Together, these added ~47 GB to the MBS=4 memory footprint (119.8 + ~47 = ~167 GB), pushing past the limit.

**Job 232 (MBS=2 + overlap): 343 TFLOPS (-20% regression)**

Falling back to MBS=2 allowed the overlap buffers to fit in memory (119.1 GB peak). However, performance *regressed* from 429 to 343 TFLOPS/GPU -- a 20% slowdown. nsys profiling confirmed the root cause (see Section 6.4): while overlap improved from 4% to 28%, the `delay_wgrad_compute` mechanism nearly doubled the total communication volume. The additional A2A operations required for the deferred wgrad path overwhelmed the benefit of overlapping them.

The fundamental problem is architectural: with 128 experts and a small per-expert hidden dimension of 768, the individual wgrad GEMMs are too small to meaningfully hide the A2A communication behind them. The overlap mechanism generates more communication than compute can absorb.

Notably, the dgxc-benchmarking H100 profile enables `moe_a2a_overlap` (with PP=2), but the B200 profile does **not** -- and our testing confirmed this was the correct design choice. The B200's higher tensor core throughput makes the compute-side of the overlap equation even less favorable, as the small wgrad GEMMs complete faster, leaving less time to hide communication.

---

## 5. Baseline vs Best: Detailed Comparison

### 5.1 Configuration Side-by-Side

| Parameter | Baseline | Best | Change |
|---|---|---|---|
| Pipeline Parallelism (PP) | 1 | 1 | Unchanged |
| Tensor Parallelism (TP) | 1 | 1 | Unchanged |
| Context Parallelism (CP) | 1 | 1 | Unchanged |
| Expert Parallelism (EP) | 8 | 8 | Unchanged |
| Data Parallelism (DP) | 64 | 64 | Unchanged |
| Micro-Batch Size (MBS) | 1 | **4** | 4x increase |
| Global Batch Size (GBS) | 4096 | 4096 | Unchanged |
| Gradient Accumulation Steps | 64 | **16** | 4x fewer |
| CUDA Graph Scope | moe_router, moe_preprocess | moe_router, moe_preprocess | Unchanged |
| ZeRO-3 | Yes | Yes | Unchanged |
| MoE A2A Overlap | No | No | Unchanged (see Section 4.5) |
| MoE Dispatcher | NCCL AlltoAll | NCCL AlltoAll | Unchanged (see Section 9) |

The optimization was purely a micro-batch size change -- no parallelism or algorithmic modifications were needed. The B200 baseline profile already used PP=1 (unlike the H100/H200 profile which used PP=2), so pipeline elimination was not a factor here.

### 5.2 Memory Breakdown

| Metric | Baseline (MBS=1) | Best (MBS=4) |
|---|---|---|
| `mem-max-allocated` (Rank 0) | 47.5 GB | 119.8 GB |
| `mem-max-reserved` (Rank 0) | 49.2 GB | 122.2 GB |
| Post-iteration `mem-allocated` | 29.4 GB | 30.3 GB |
| Memory utilization | 25% of 192 GB | **62% of 192 GB** |
| Headroom | 145 GB | 72 GB |

Post-iteration memory (model weights + optimizer state) is nearly identical at ~30 GB, confirming that the MBS increase only affects activation memory. The 72.4 GB difference in peak memory (119.8 - 47.5) is 4x the per-micro-batch activation footprint increase from MBS=1 to MBS=4, as expected.

### 5.3 Performance Breakdown

| Metric | Baseline | Best | Improvement |
|---|---|---|---|
| MODEL TFLOPS/GPU | 232 | 489 | **2.11x** |
| Step time (steady state) | 26,000 ms | 12,330 ms | **2.11x faster** |
| Tokens per second (cluster) | 644,677 | 1,358,475 | **2.11x** |
| Tokens per second per GPU | 10,073 | 21,226 | **2.11x** |

*Tokens/sec calculated as: GBS * seq_length / step_time = 4096 * 4096 / step_time_seconds*

---

## 6. nsys Profile Analysis

nsys profiling was collected on Job 198 (MBS=2, 429 TFLOPS/GPU, CUDA graphs enabled) for the last 5 training steps across all 64 ranks. Analysis was performed using the [nsys-profiling-analyser-tool](https://gitlab.aws.dev/paragao/nsys-profiling-analyser-tool) on 3 strategic ranks: Rank 0 (EP0/DP0), Rank 1 (EP1/DP0), and Rank 8 (EP0/DP1).

Note: Profiling adds overhead -- the profiled steps ran at 338 TFLOPS (17.8s) compared to 429 TFLOPS (14.0s) in steady state. The kernel time *ratios* remain representative.

### 6.1 Communication vs Compute

| Category | Rank 0 Time (s) | % of GPU Time |
|---|---|---|
| **Compute Kernels** | **48.5** | **60%** |
| NCCL Communication | 32.3 | 40% |
| **Grand Total** | **80.8** | **100%** |

Averaged across the 3 profiled ranks: **58% compute, 42% communication**. This is a more favorable ratio than the H200 (49% compute, 51% communication), suggesting the B200's higher tensor core throughput shifts the balance toward compute.

### 6.2 Communication Breakdown

| NCCL Operation | Time (s) | Purpose |
|---|---|---|
| **SendRecv (MoE A2A)** | **23.3** | Expert Parallel token dispatch + combine |
| AllGather/ReduceScatter (ZeRO-3) | ~6.9 | Parameter all-gather + gradient reduce-scatter |

MoE A2A dominates communication, consistent with the 128-expert architecture. With EP=8, each layer performs all-to-all dispatch and combine operations across 8 GPU groups for both forward and backward passes.

### 6.3 Compute-Communication Overlap

| Metric | Value |
|---|---|
| Total communication time | 29.6s |
| Communication hidden behind compute | 4% |
| Exposed (serialized) communication | 28.5s |
| Effective compute efficiency | 59% |
| Idle / bubble time | 31% |

Only 4% of communication is overlapped with compute. The 31% "idle" time represents gaps between kernels within training steps -- a combination of synchronization barriers, memory allocation overhead, and CUDA graph replay scheduling. This poor overlap is the primary efficiency bottleneck.

### 6.4 MoE A2A Overlap Analysis (Job 232 vs Job 198)

To understand why `moe_a2a_overlap` regressed performance, we collected nsys profiles on Job 232 (MBS=2, overlap enabled) and compared them against Job 198 (MBS=2, no overlap) using the nsys-profiling-analyser-tool's diff mode.

| Metric | Job 198 (no overlap) | Job 232 (with overlap) | Delta |
|---|---|---|---|
| Step time (profiled) | 17.87s | 16.94s | -5.2% |
| Compute time | 41.1s (58%) | 34.1s (39%) | -7.0s (-17%) |
| Communication time | 29.6s (42%) | 54.3s (61%) | **+24.7s (+83%)** |
| MoE A2A SendRecv | 23.3s | 42.5s | +19.2s (+82%) |
| ZeRO AllGather/RS | ~6.9s | ~14.2s | +7.3s (+106%) |
| Overlap (comm hidden) | 4% | 28% | +24pp |
| Idle / bubble | 31% | 24% | -7pp |
| Effective compute | 59% | 47% | -12pp |

The diff reveals the core problem: `delay_wgrad_compute` nearly doubled the total communication budget (29.6s to 54.3s). While overlap improved from 4% to 28%, the *exposed* (non-overlapped) communication actually increased from 28.5s to 39.1s (+37%). The net effect was a worse compute-to-communication ratio (58:42 degraded to 39:61) and a 12-point drop in effective compute efficiency. The root cause is described in Section 4.5: the per-expert wgrad GEMMs are too small to hide the additional A2A transfers that the overlap mechanism introduces.

### 6.5 Implications

1. **MoE A2A remains the dominant cost**: At 23.3s of the 80.8s profiled window, reducing this communication would yield the largest gains. FP8 precision would halve the token payload size, directly reducing A2A volume.

2. **Megatron-Core's overlap strategy is counterproductive here**: The `moe_a2a_overlap` feature doubled communication volume for a net 20% regression (Section 6.4). Improving overlap for this model likely requires a fundamentally different approach -- such as reducing communication volume first via FP8, or using a future NVLink-aware dispatcher (see Section 9 for HybridEP investigation).

3. **B200 is more compute-efficient than H200**: The 58/42 compute-to-comm ratio (vs H200's 49/51) reflects the B200's higher tensor core throughput. The same communication volume represents a smaller fraction of total time because compute is faster.

---

## 7. Cross-Platform Comparison: H200 vs B200

| Metric | H200 Best | B200 Best | B200 Advantage |
|---|---|---|---|
| GPU | H200 SXM 141 GB HBM3e | B200 SXM 192 GB HBM3e | +36% memory |
| Peak BF16 TFLOPS (spec) | 989 | 2250 | +128% |
| MODEL TFLOPS/GPU achieved | 315 | 489 | **+55%** |
| % of peak (MFU) | 31.8% | 21.7% | H200 better utilized |
| Step time | 9.57s | 12.33s | H200 faster per step* |
| Tokens/sec/GPU | 1,369 | 21,226 | N/A (different GBS)** |
| Best MBS | 2 | 4 | B200 enables 2x larger |
| CUDA graphs | Not possible (OOM) | Enabled | B200 has headroom |
| Pipeline parallelism | PP=1 | PP=1 | Same |
| Peak memory used | 119 GB / 141 GB (84%) | 120 GB / 192 GB (62%) | B200 has 72 GB headroom |
| Improvement over baseline | 2.13x | 2.11x | Similar optimization gain |
| Compute : Comm ratio | 49% : 51% | 58% : 42% | B200 more compute-efficient |

*H200 uses GBS=2048, B200 uses GBS=4096. H200's faster step time processes half the tokens per step.

**Per-step token counts differ (GBS=2048*4096=8.4M vs GBS=4096*4096=16.8M), so raw tokens/sec/GPU is not directly comparable.

### Key Takeaways

1. **Memory capacity is the decisive factor**: Both GPUs achieved their best results at PP=1 with the highest MBS that fits in memory. The B200's extra 51 GB enabled MBS=4 (vs MBS=2 on H200) and CUDA graphs simultaneously -- a combination impossible on H200.

2. **The B200 has lower MFU**: Despite higher absolute TFLOPS, the B200 achieves 21.7% of its 2250 TFLOPS peak, while the H200 achieves 31.8% of its 989 TFLOPS peak. This is characteristic of MoE workloads where active parameters per token (~3B) are far smaller than total parameters (233B) -- the communication overhead prevents the faster compute hardware from being fully utilized.

3. **Communication is the equalizer**: Both platforms spend 40-51% of GPU time on NCCL communication, dominated by MoE A2A. Since the communication volume is identical (same model, same EP=8), the B200's faster compute simply shifts the ratio without reducing communication time. FP8 precision would benefit both platforms by reducing communication volume.

---

## 8. Appendix

### 8.1 Complete Experiment Log

| Job ID | MBS | CUDA Graph Scope | moe_a2a_overlap | TFLOPS/GPU | Step Time | Peak Mem (GB) | Status |
|---|---|---|---|---|---|---|---|
| 197 | 1 | moe_router, moe_preprocess | No | 232 | 26.0s | 47.5 | Baseline |
| 198 | 2 | moe_router, moe_preprocess | No | 429 | 14.0s | 71.6 | +nsys profiling |
| **199** | **4** | **moe_router, moe_preprocess** | **No** | **489** | **12.3s** | **119.8** | **Best** |
| 200 | 8 | moe_router, moe_preprocess | No | -- | -- | >168 (OOM) | OOM |
| 201 | 6 | moe_router, moe_preprocess | No | -- | -- | -- | GBS not divisible |
| 202 | 4 | moe_router, moe_preprocess, attn | No | 488 | 12.4s | 119.8 | No improvement |
| 228 | 4 | moe_router, moe_preprocess | **Yes** | -- | -- | >167 (OOM) | OOM (overlap buffers) |
| 232 | 2 | moe_router, moe_preprocess | **Yes** | 343 | 17.6s | 119.1 | Regression (-20%) +nsys |
| 390 | 4 | moe_router, moe_preprocess | No | 323 | 18.65s | 120.2 | DeepEP flex dispatcher (-34%) |
| 391 | 4 | moe_router, moe_preprocess | No | 491 | 12.28s | 119.8 | HybridEP config (fell back to AlltoAll) |

All jobs: PP=1, TP=1, CP=1, EP=8, GBS=4096, 64 GPUs (8 nodes). Dispatcher is NCCL AlltoAll unless noted.

### 8.2 Launch Commands

**Best configuration (MBS=4)**:

```bash
export LLMB_INSTALL=/fsx/dgxc-benchmark
export HF_TOKEN=<your_hf_token>
export SBATCH_ACCOUNT=root
export SBATCH_PARTITION=b200
export NEMORUN_HOME=$LLMB_INSTALL/workloads/pretrain_qwen3
source $LLMB_INSTALL/venvs/venv_3872b31f7bed/bin/activate
cd $LLMB_INSTALL/llmb_repo/qwen3/pretrain/

MBS=4 \
  JOB_TOTAL_GPUS=64 \
  GPU_TYPE=b200 \
  MODEL_SIZE=30b \
  DTYPE=bf16 \
  ./launch.sh
```

**With nsys profiling**:

```bash
MBS=2 \
  ENABLE_PROFILE=true PROFILE_START_STEP=46 PROFILE_STOP_STEP=50 \
  JOB_TOTAL_GPUS=64 \
  GPU_TYPE=b200 \
  MODEL_SIZE=30b \
  DTYPE=bf16 \
  ./launch.sh
```

### 8.3 Key File Locations (on B200 cluster)

| Path | Description |
|---|---|
| `/fsx/dgxc-benchmark/` | LLMB install root |
| `/fsx/dgxc-benchmark/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh` | EFA-upgraded container |
| `/fsx/dgxc-benchmark/llmb_repo/qwen3/pretrain/launch.sh` | Launch script |
| `.../Megatron-Bridge/scripts/performance/utils/executors.py` | EFA env vars + mount patch |
| `.../configs/qwen3/workload_base_configs.py` | Parallelism presets |
| `/fsx/dgxc-benchmark/workloads/pretrain_qwen3/experiments/` | All experiment outputs |
| `/fsx/ubuntu/ofi-override/` | EFA OFI override libraries |

### 8.4 Potential Future Optimizations

| Optimization | Expected Impact | Notes |
|---|---|---|
| FP8 precision | 30-50% gain | 2x tensor core throughput + 2x less A2A comm volume |
| Compute-communication overlap tuning | Unlikely in BF16 | Tested: `moe_a2a_overlap` regressed 20%; per-expert GEMMs too small to hide A2A |
| HybridEP/DeepEP dispatcher | **Not viable in NeMo 25.11.01** | Tested: see Section 9. Requires newer Megatron-Core with flex dispatcher |
| Longer sequence length | 5-10% gain | Better compute/comm ratio per token |
| MBS=5 with GBS=5120 | ~5% gain | Requires adjusting GBS for divisibility |

---

## 9. MoE Dispatcher Investigation: DeepEP and HybridEP

### 9.1 Background

The NVIDIA dgxc-benchmarking framework's `workload_base_configs.py` specifies `moe_flex_dispatcher_backend="hybridep"` for GB200 and GB300 profiles, targeting the Megatron-Core "flex" token dispatcher. This dispatcher replaces standard NCCL AlltoAll with the DeepSeek [DeepEP library](https://github.com/deepseek-ai/DeepEP), which provides fused permute + all-to-all operations using TMA instructions and optional IBGDA RDMA. The B200 profile does not set this flag.

We investigated whether enabling DeepEP or HybridEP on B200 with EFA could improve upon the NCCL AlltoAll dispatcher.

### 9.2 Dispatcher Types in Megatron-Core

| Dispatcher | `--moe-token-dispatcher-type` | Backend | Communication |
|---|---|---|---|
| AllGather | `allgather` | NCCL | AllGather / ReduceScatter |
| **AlltoAll** | **`alltoall`** | **NCCL** | **All-to-All (our baseline)** |
| Flex (DeepEP) | `flex` | `deep_ep.Buffer` | LD/ST instructions, NVSHMEM |
| Flex (HybridEP) | `flex` | `deep_ep.HybridEPBuffer` | TMA instructions, IBGDA RDMA |

### 9.3 Discovery: All Benchmarks Used NCCL AlltoAll

Analysis of the training logs for both our best B200 run (Job 199) and the best H200 run (Job 3185) confirmed that **both clusters used the NCCL AlltoAll dispatcher**, not DeepEP or HybridEP:

```
# From both logs:
moe_token_dispatcher_type: alltoall
moe_flex_dispatcher_backend: deepep  # (Megatron-Core default value, not active)
```

Both logs also contained the warning:
```
Not a valid flex dispatcher backend. Skipping flex dispatcher backend configuration.
```

**Root cause**: The B200 config did not set `moe_flex_dispatcher_backend` at all (default `None`). The validation function in `Megatron-Bridge/src/megatron/bridge/training/flex_dispatcher_backend.py` hit the `else` branch and silently fell back to AlltoAll.

### 9.4 Container Analysis

We verified the NeMo 25.11.01 container's capabilities:

| Component | Status | Location |
|---|---|---|
| `deep_ep` Python package | **Installed** | `/opt/venv/lib/python3.12/site-packages/deep_ep/` |
| `deep_ep.Buffer` (DeepEP) | **Available** | For LD/ST-based dispatch |
| `deep_ep.HybridEPBuffer` (HybridEP) | **Available** | For TMA-based dispatch |
| `flex_token_dispatcher.py` in Megatron-Core | **Does not exist** | Not in `/opt/megatron-lm/megatron/core/transformer/moe/` |
| `deepep_dispatcher/` directory | **Does not exist** | Not in Megatron-Core MoE module |

The `deep_ep` library is installed, but the Megatron-Core version in NeMo 25.11.01 predates the flex dispatcher integration. The MoE module only contains `token_dispatcher.py` (AlltoAll + AllGather implementations).

### 9.5 Experiment: DeepEP Backend (Job 390)

We set `moe_flex_dispatcher_backend="deepep"` in `workload_base_configs.py` for the B200 BF16 profile. The `flex_dispatcher_backend.py` validation passed (B200 is in the DeepEP allowlist), and the training log confirmed:

```
moe_token_dispatcher_type: flex
moe_flex_dispatcher_backend: deepep
```

**Result**: 323 MODEL TFLOPS/GPU, 18.65s per step -- a **34% regression** from the 489 TFLOPS AlltoAll baseline.

| Metric | AlltoAll (Job 199) | DeepEP (Job 390) | Delta |
|---|---|---|---|
| TFLOPS/GPU | 489 | 323 | **-34%** |
| Step time | 12.3s | 18.65s | +52% |
| Peak memory | 119.8 GB | 120.2 GB | +0.3% |
| CUDA_DEVICE_MAX_CONNECTIONS | 8 | 32 | Forced by DeepEP path |

The regression is expected: DeepEP's LD/ST instructions are designed for NVSwitch/NVLink topologies where GPU-initiated memory operations can leverage high-bandwidth intra-node links. On EFA (RDMA over Ethernet), these operations fall back to suboptimal paths. Additionally, the DeepEP code path sets `CUDA_DEVICE_MAX_CONNECTIONS=32` and reserves 20 SMs for communication kernels (`moe_deepep_num_sms=20`), reducing the compute resources available.

### 9.6 Experiment: HybridEP Backend (Job 391)

We patched `flex_dispatcher_backend.py` to add `"NVIDIA B200"` to the HybridEP GPU allowlist and set `moe_flex_dispatcher_backend="hybridep"` in the B200 config. The Megatron-Bridge validation passed, but the training log showed:

```
moe_token_dispatcher_type: alltoall  # Fell back!
moe_flex_dispatcher_backend: deepep  # Megatron-Core default
moe_hybridep_num_sms: 16            # Default, not the 32 set by helpers.py
```

**Result**: 491 MODEL TFLOPS/GPU, 12.28s per step -- matching the AlltoAll baseline.

HybridEP silently fell back to AlltoAll because `flex_token_dispatcher.py` does not exist in this Megatron-Core version. When `apply_flex_dispatcher_backend()` set `model_config.moe_token_dispatcher_type = "flex"`, Megatron's argument parser overrode it with the default `"alltoall"` since the flex module couldn't be loaded.

### 9.7 Conclusions

1. **NCCL AlltoAll is the optimal MoE dispatcher for B200 on EFA in NeMo 25.11.01.** The flex dispatcher (DeepEP/HybridEP) is not implemented in this version's Megatron-Core.

2. **DeepEP causes a 34% regression when force-enabled.** The LD/ST communication kernels are inefficient on EFA and the SM reservation + CUDA connection changes hurt overall throughput.

3. **The dgxc-benchmarking `workload_base_configs.py` configs for GB200/GB300 that set `moe_flex_dispatcher_backend="hybridep"` are forward-looking.** They target a newer Megatron-Core version that includes `flex_token_dispatcher.py`. In NeMo 25.11.01, these settings are silently ignored.

4. **To test flex dispatchers, a newer NeMo container is required** -- one built from a Megatron-Core commit that includes the flex token dispatcher module, the DeepEP dispatcher integration, and NVSHMEM support for EFA-based RDMA.

