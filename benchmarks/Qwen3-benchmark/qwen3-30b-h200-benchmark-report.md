---
title: "Qwen3-30B-A3B Pre-training Benchmark Report"
subtitle: "Performance Optimization on H200 GPUs"
theme: aws
sidebar: true
meta:
  Date: "March 24, 2026"
  Cluster: "8x p5en.48xlarge (64x NVIDIA H200 SXM 141GB HBM3e)"
  Model: "Qwen3-30B-A3B MoE (BF16)"
  Software: "NeMo 25.11.01, Megatron-Bridge, NCCL 2.29.3"
---

# Qwen3-30B-A3B Pre-training Benchmark Report: Performance Optimization on H200 GPUs

---

## 1. Executive Summary

This report documents the performance optimization of Qwen3-30B-A3B Mixture-of-Experts (MoE) pre-training on an 8-node Amazon EC2 p5en.48xlarge cluster equipped with 64 NVIDIA H200 SXM GPUs (141GB HBM3e each). Starting from the NVIDIA dgxc-benchmarking H100 baseline profile, we achieved a **2.13x throughput improvement** -- from **148 to 315 MODEL TFLOPS/GPU** in BF16 precision.

### Key Results

| Configuration | TFLOPS/GPU | Iter Time (ms) | Peak Memory (GB) | vs Baseline |
|---|---|---|---|---|
| H100 Baseline (PP=2, MBS=1, CUDA graphs) | 148 | 20,400 | 38 / 141 | -- |
| + MBS=2 (PP=2, CUDA graphs) | 275 | 10,930 | 64 / 141 | +86% |
| + PP=1, no CUDA graphs | **315** | 9,570 | 119 / 141 | **+113%** |
| + GBS=4096 | 317 | 19,050* | 119 / 141 | +114% |

*GBS=4096 processes 2x tokens per iteration, so per-token throughput is comparable at 317 vs 315 TFLOPS/GPU.

### What Worked

1. **Micro-batch size 1 to 2** (+86%): The H100 profile uses MBS=1 because 80GB HBM3 constrains activation memory. The H200's 141GB HBM3e provides sufficient headroom to double the micro-batch size, dramatically improving GPU arithmetic intensity.

2. **Pipeline parallelism 2 to 1** (+14.5% additional): Eliminating the 2-stage pipeline removes all pipeline bubble overhead. With ZeRO-3 (optimizer+gradient+parameter sharding) active, doubling data parallelism from 32 to 64 keeps per-GPU optimizer memory constant. The trade-off is that CUDA graph capture no longer fits in memory, but removing graphs has no negative impact -- the kernel launch overhead is negligible relative to the large MoE layer compute.

### What Did Not Work

| Experiment | Result | Reason |
|---|---|---|
| MBS=4 (any PP setting) | OOM | 128 experts create excessive activation memory |
| CUDA graphs + PP=1 | OOM | 48-layer graph capture uses 14.77GB private pool, exceeding 141GB |
| CUDA graphs (attn scope) + PP=2 MBS=2 | OOM | Attention graph capture pushes past 141GB at 121GB allocated |
| CUDA graphs (moe_router only) + PP=1 | 208 TFLOPS (-34%) | Partial graph capture introduces sync overhead, major regression |
| NCCL_NVLS_ENABLE=1 | OOM | NVLink SHARP allocates 512MB+ multicast buffers, no headroom |
| GBS=4096 | 317 TFLOPS (+0.6%) | Marginal gain from amortizing gradient all-reduce |

### Bottleneck Analysis

nsys profiling reveals the workload is **50.8% communication-bound**. MoE Expert Parallel all-to-all (token dispatch/combine) alone accounts for 38% of GPU time. This is inherent to the 128-expert architecture with EP=8 and represents the primary barrier to further BF16 optimization.

---

## 2. Environment and Setup

### 2.1 Cluster Architecture

| Component | Specification |
|---|---|
| Instance type | Amazon EC2 p5en.48xlarge |
| Nodes | 8 (of 16 total in shared capacity reservation) |
| GPUs per node | 8x NVIDIA H200 SXM 141GB HBM3e |
| Total GPUs | 64 |
| Intra-node interconnect | NVLink / NVSwitch (900 GB/s bidirectional per GPU) |
| Inter-node network | Elastic Fabric Adapter (EFA), 4x 400Gbps per node (3.2 Tbps aggregate) |
| Shared filesystem | Amazon FSx for Lustre (`/fsx/`) |
| Operating system | Ubuntu 24.04 (head node), container-based compute |
| Slurm | Workload manager with Pyxis/Enroot container support |

### 2.2 Software Stack

| Component | Version |
|---|---|
| Container | `nvidia+nemo+25.11.01` (EFA-upgraded, 36GB squashfs) |
| NeMo Framework | 25.11.01 |
| Megatron-Core | Bundled with NeMo 25.11.01 |
| Megatron-Bridge | Commit 4df8c97 (from NVIDIA dgxc-benchmarking) |
| PyTorch | 2.x (bundled) |
| NCCL | 2.29.3 |
| CUDA | 12.x (bundled) |
| AWS OFI NCCL plugin | From EFA installer (bundled in upgraded container) |
| Benchmark framework | NVIDIA dgxc-benchmarking v25.12.02 |

### 2.3 Model Architecture: Qwen3-30B-A3B

| Parameter | Value |
|---|---|
| Total parameters | 233.46 billion |
| Active parameters per token | ~3 billion |
| Architecture | Mixture-of-Experts (MoE) Transformer |
| Number of layers | 48 |
| Hidden size | 2048 |
| Attention heads | 32 (GQA with 4 KV groups) |
| Dense FFN hidden size | 6144 |
| MoE FFN hidden size (per expert) | 768 |
| Number of experts | 128 |
| Top-K routing | 8 |
| Sequence length | 4096 |
| Vocabulary size | ~152K (Qwen3 tokenizer) |
| Precision | BF16 |

### 2.4 AWS EFA Networking Patches

Running the NeMo container on AWS EFA-equipped instances requires several modifications. The stock NeMo 25.11.01 container has a **NCCL plugin naming mismatch**: NCCL 2.29+ searches for `libnccl-net-aws-ofi.so`, but the EFA installer names it `libnccl-net-ofi.so`. We used a pre-built EFA-upgraded container (`nvidia+nemo+25.11.01-efa-nccl29.sqsh`) that resolves this by including the correctly-named plugin.

Beyond the container fix, the Megatron-Bridge `executors.py` file requires a patch to inject EFA-specific environment variables into every Slurm job. The patch adds:

```diff
--- a/scripts/performance/utils/executors.py
+++ b/scripts/performance/utils/executors.py
 PERF_ENV_VARS = {
     "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
     "TRANSFORMERS_OFFLINE": "1",
     "TOKENIZERS_PARALLELISM": "False",
     "NCCL_NVLS_ENABLE": "0",
+    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
+    "NCCL_GRAPH_REGISTER": "0",
     "NVTE_NORM_FWD_USE_CUDNN": "1",
     "NVTE_NORM_BWD_USE_CUDNN": "1",
     "TORCH_NCCL_HIGH_PRIORITY": "1",
     "HF_HUB_OFFLINE": "1",
+    # EFA networking
+    "FI_PROVIDER": "efa",
+    "NCCL_SOCKET_IFNAME": "^docker,lo,veth",
+    "NCCL_BUFFSIZE": "8388608",
+    "NCCL_P2P_NET_CHUNKSIZE": "8388608",
+    "NCCL_TUNER_PLUGIN": "/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so",
 }
```

**Explanation of each variable:**

| Variable | Value | Purpose |
|---|---|---|
| `FI_PROVIDER` | `efa` | Forces libfabric to use the EFA provider for NCCL inter-node communication |
| `NCCL_SOCKET_IFNAME` | `^docker,lo,veth` | Excludes virtual network interfaces from NCCL socket communication |
| `NCCL_BUFFSIZE` | `8388608` (8MB) | Increases NCCL buffer size for higher throughput on EFA |
| `NCCL_P2P_NET_CHUNKSIZE` | `8388608` (8MB) | Sets P2P chunk size for EFA transfers |
| `NCCL_TUNER_PLUGIN` | `/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so` | Loads the AWS OFI NCCL tuner plugin for algorithm selection |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Enables PyTorch's expandable memory segments, critical for BF16 workloads to avoid fragmentation-induced OOM |
| `NCCL_GRAPH_REGISTER` | `0` | Required when CUDA graphs and expandable_segments are both enabled; prevents NCCL from attempting to register graph memory regions that conflict with expandable segments |
| `NCCL_NVLS_ENABLE` | `0` | Disables NVLink SHARP to conserve GPU memory (NVLS allocates large multicast buffers) |

---

## 3. Optimization Journey

### 3.1 Starting Point: The H100 Baseline Profile

The NVIDIA dgxc-benchmarking repository provides pre-tuned parallelism profiles for each GPU type. The H100 profile for Qwen3-30B-A3B at 64 GPUs uses:

- **TP=1, PP=2, CP=1, VP=12, EP=8, ETP=1**
- **MBS=1, GBS=2048**
- **CUDA graphs**: `transformer_engine` implementation, scope: `moe_router, moe_preprocess`
- **ZeRO-3**: `data_parallel_sharding_strategy=optim_grads_params`, `use_distributed_optimizer=True`
- **MoE A2A overlap**: enabled with `delay_wgrad_compute=True`
- **Dispatcher**: DeepEP backend

This profile is designed for H100 SXM GPUs with **80GB HBM3**. The key constraint is memory: MBS=1 and PP=2 are necessary to fit the 128-expert MoE model within 80GB. CUDA graphs for MoE router and preprocess kernels reduce kernel launch overhead.

**Baseline result on H200**: 148 MODEL TFLOPS/GPU, 20.4s per iteration, 38GB peak memory.

> [!IMPORTANT]
> The H200 uses only 38 of 141GB -- **73% of GPU memory is unused**. This immediately signals that the H100 profile is heavily under-utilizing the H200's capacity.

### 3.2 Optimization 1: Micro-Batch Size 1 to 2 (+86%)

**Change**: `MBS=1` to `MBS=2`, all other parameters unchanged.

**Result**: 275 MODEL TFLOPS/GPU, 10.93s per iteration, 64GB peak memory.

**Why it works**: With MBS=1, each GPU processes one sample (4096 tokens) per micro-batch forward/backward pass. The GPU's tensor cores are severely underutilized because the batch dimension is too small to saturate the compute units, especially for the many small expert FFN GEMMs (768 hidden size per expert). Doubling to MBS=2 doubles the matrix dimensions in the batch axis, improving GEMM efficiency.

The memory impact is significant but manageable: peak memory increased from 38GB to 64GB (26GB increase), well within the 141GB capacity. The 26GB increase comes from doubled activation memory during the forward pass -- each of the 48 layers stores activations for 2 samples instead of 1.

**Why MBS=4 fails**: We tested MBS=4 with PP=2/VP=4 and PP=2/VP=2. Both OOM'd at 130-141GB. The 128-expert MoE architecture is uniquely memory-hungry: each layer's forward pass dispatches tokens to 8 of 128 experts, and the per-expert activations accumulate across the top-8 routing. MBS=4 roughly doubles the MBS=2 activation footprint again (~128GB for activations alone), exceeding capacity.

### 3.3 Optimization 2: Pipeline Parallelism 2 to 1 (+14.5% additional)

**Change**: `PP=2, VP=12` to `PP=1, VP=None`, CUDA graphs disabled (`cuda_graph_impl=none`).

**Result**: 315 MODEL TFLOPS/GPU, 9.57s per iteration, 119GB peak memory.

**Why it works -- pipeline bubble elimination**: With PP=2 and VP=12 (interleaved 1F1B schedule), the pipeline has bubble overhead at the start and end of each iteration. The bubble fraction for interleaved 1F1B is approximately `(PP - 1) / (num_microbatches * VP)`. With PP=2, VP=12, and num_microbatches = GBS/(DP*MBS) = 2048/(32*2) = 32: bubble fraction ~= 1/(32*12) = 0.26%. While the bubble fraction appears small, the practical overhead is larger because each pipeline stage processes only 24 of 48 layers, creating stage imbalances and additional communication for pipeline send/recv.

With PP=1, every GPU holds all 48 layers and processes complete forward/backward passes without any pipeline coordination. This eliminates:

- Pipeline send/recv communication between stages
- Pipeline bubble idle time
- Stage imbalance overhead from uneven layer distribution
- Virtual pipeline interleaving overhead (VP=12 means 12 virtual chunks with 2 layers each)

**Memory math -- why PP=1 fits**: When PP changes from 2 to 1, data parallelism (DP) changes from 32 to 64 because `DP = total_GPUs / (TP * PP * CP) = 64 / (1 * 1 * 1) = 64`.

- **Model weights**: With PP=2, each GPU holds 24 layers (~2.58B params). With PP=1, each GPU holds 48 layers (~5.16B params). Weight memory doubles from ~5GB to ~10GB in BF16.
- **Optimizer state**: ZeRO-3 shards optimizer states across all DP ranks. With DP doubling from 32 to 64, per-GPU optimizer memory is halved. Net effect: roughly constant.
- **Activations**: With PP=1 and MBS=2, only 1 micro-batch is in-flight at a time (simple data-parallel training). With PP=2/VP=12, the interleaved schedule keeps up to 24 micro-batches partially in-flight. However, activation memory per micro-batch is smaller with PP=2 (24 layers vs 48). The net result is that PP=1 uses moderately more activation memory.
- **Total**: Peak memory increased from 64GB (PP=2, MBS=2) to 119GB (PP=1, MBS=2). The 55GB increase is accommodated by the 141GB HBM3e capacity with 22GB headroom remaining.

**Why CUDA graphs cannot be used with PP=1**: CUDA graph capture for the `moe_router` and `moe_preprocess` scopes across all 48 layers (vs 24 with PP=2) requires 14.77GB of private pool memory. With 119GB already allocated for model/optimizer/activations, adding 14.77GB exceeds the 141GB limit. Job 3181 confirmed this: it completed iteration 1 at 119GB, then OOM'd during graph capture.

**Code fix required**: The `helpers.py` file needed a patch to auto-disable virtual pipeline parallelism when PP=1. Without this fix, the launch script would attempt to pass VP=12 with PP=1, causing Megatron to error with "pipeline-model-parallel size should be greater than 1 with interleaved schedule". The fix adds:

```python
# In set_user_overrides(), after pipeline_model_parallel_size is set:
if recipe.model.pipeline_model_parallel_size <= 1:
    recipe.model.virtual_pipeline_model_parallel_size = None
```

### 3.4 Diminishing Returns: Further Optimization Attempts

With the best config at 315 TFLOPS/GPU using 119 of 141GB, the remaining optimization space is narrow. We systematically tested several additional configurations:

**CUDA graphs with reduced scope (moe_router only)**: We hypothesized that capturing only the `moe_router` scope (excluding `moe_preprocess`) would reduce graph memory enough to fit with PP=1. Graph capture succeeded, but steady-state throughput dropped to **208 TFLOPS/GPU** -- a 34% regression. The likely cause is that partial CUDA graph capture introduces synchronization barriers between graphed and non-graphed regions, and the `moe_router` alone is too small a portion of the compute to benefit from graph replay. The overhead of entering/exiting graph execution dominates.

**NCCL NVLink SHARP (NVLS)**: NVLink SHARP enables hardware-accelerated intra-node reductions using NVSwitch multicast. However, NVLS requires NCCL to allocate dedicated multicast buffers in GPU memory. At 119GB already used, the attempted allocation of 512MB+ buffer chunks immediately triggered OOM before the first iteration.

**GBS=4096**: Increasing the global batch size from 2048 to 4096 doubles the number of gradient accumulation steps from 16 to 32 (with DP=64, MBS=2). This amortizes the gradient all-reduce communication over twice as many compute steps. The result was **317 TFLOPS/GPU** -- a 0.6% improvement. The gain is real but marginal because the gradient all-reduce (ReduceScatter) is only 2.3% of total GPU time (see Section 5). The dominant communication cost is MoE A2A, which scales with the number of micro-batches, not with gradient synchronization frequency.

---

## 4. Baseline vs Best Configuration: Detailed Comparison

### 4.1 Configuration Side-by-Side

| Parameter | H100 Baseline | Best (H200-Optimized) | Change |
|---|---|---|---|
| Pipeline Parallelism (PP) | 2 | **1** | Eliminated pipeline stages |
| Virtual Pipeline (VP) | 12 | **None** | Removed (requires PP > 1) |
| Tensor Parallelism (TP) | 1 | 1 | Unchanged |
| Context Parallelism (CP) | 1 | 1 | Unchanged |
| Expert Parallelism (EP) | 8 | 8 | Unchanged |
| Expert Tensor Parallelism (ETP) | 1 | 1 | Unchanged |
| Data Parallelism (DP) | 32 | **64** | Doubled (freed by PP reduction) |
| Micro-Batch Size (MBS) | 1 | **2** | Doubled for H200 memory |
| Global Batch Size (GBS) | 2048 | 2048 | Unchanged |
| Gradient Accumulation Steps | 32 | **16** | Halved (DP doubled, MBS doubled) |
| CUDA Graph Implementation | transformer_engine | **none** | Disabled (no memory for graphs) |
| CUDA Graph Scope | moe_router, moe_preprocess | **N/A** | N/A |
| ZeRO-3 (Distributed Optimizer) | Yes | Yes | Unchanged |
| MoE A2A Overlap | Yes | Yes | Unchanged |
| MoE Dispatcher | DeepEP | DeepEP | Unchanged |
| Recompute | None | None | Unchanged |

### 4.2 Memory Breakdown

| Metric | Baseline (PP=2, MBS=1) | Best (PP=1, MBS=2) |
|---|---|---|
| Parameters per GPU | 2.58B (24 layers) | 5.16B (48 layers) |
| `mem-max-allocated` (Rank 0) | 38.3 GB | 119.2 GB |
| `mem-max-reserved` (Rank 0) | 41.8 GB | 124.3 GB |
| Post-iteration `mem-allocated` | 19.5 GB | 33.0 GB |
| Memory utilization | 27% of 141GB | **85% of 141GB** |
| Headroom | 103 GB | 22 GB |

The post-iteration allocated memory (model weights + optimizer state in steady state) grew from 19.5GB to 33.0GB -- a 13.5GB increase explained by holding all 48 layers instead of 24, partially offset by ZeRO-3 sharding across 2x more DP ranks.

The peak allocated memory (during forward/backward) grew from 38.3GB to 119.2GB -- an 80.9GB increase. This comes from: (a) doubled activation memory from MBS=2 vs MBS=1, (b) all 48 layers' activations stored simultaneously instead of 24, and (c) ZeRO-3 all-gather materializing full parameters for all layers.

### 4.3 Performance Breakdown

| Metric | Baseline | Best | Improvement |
|---|---|---|---|
| MODEL TFLOPS/GPU | 148 | 315 | **+113%** |
| Iteration time (steady state) | 20,400 ms | 9,570 ms | **2.13x faster** |
| Tokens per second (cluster) | 41,100 | 87,600 | **2.13x** |
| Tokens per second per GPU | 643 | 1,369 | **2.13x** |

*Tokens/sec calculated as: GBS * seq_length / iter_time = 2048 * 4096 / iter_time_seconds*

### 4.4 Why Performance Improved: Mechanistic Explanation

The 2.13x improvement decomposes into two multiplicative factors:

**Factor 1 -- MBS doubling (1.86x)**: The primary gain comes from improved arithmetic intensity. With MBS=1, the GEMMs in each expert FFN have shape `[tokens_per_expert, 768] x [768, hidden]` where `tokens_per_expert` is small (4096 tokens routed across 128 experts with top-8 means ~256 tokens per expert per sample). With MBS=2, this doubles to ~512 tokens per expert, significantly improving GEMM efficiency on the H200's tensor cores. The attention GEMMs similarly benefit from the larger batch dimension.

Additionally, with PP=2 and VP=12, the interleaved 1F1B schedule processes 32 micro-batches through 12 virtual pipeline chunks. Each virtual chunk contains only 2 layers, meaning the GPU rapidly alternates between tiny compute kernels and pipeline communication. MBS=2 doubles the compute per kernel invocation, improving the compute-to-overhead ratio.

**Factor 2 -- PP elimination (1.15x)**: Removing the 2-stage pipeline provides a compounding benefit:

1. **Zero pipeline bubbles**: No idle time waiting for pipeline warmup/cooldown.
2. **No pipeline send/recv**: Eliminates inter-stage point-to-point communication entirely.
3. **Simpler scheduling**: Replaces VP=12 interleaved 1F1B (which processes 12 virtual chunks of 2 layers each, cycling through forward and backward passes) with straightforward data-parallel training (48 layers, forward then backward).
4. **Better memory access patterns**: All 48 layers are processed sequentially without context switching between virtual pipeline chunks, improving GPU cache utilization.

---

## 5. nsys Profile Analysis

Nsys profiling was collected on the best configuration (Job 3185: PP=1, MBS=2, GBS=2048, no CUDA graphs) for steps 45-50 across all 64 ranks. The analysis below is from Rank 0 (node 0, GPU 0).

### 5.1 Communication vs Compute

| Category | Time (s) | % of GPU Time |
|---|---|---|
| **NCCL Communication (total)** | **48.5** | **50.8%** |
| Compute Kernels (total) | 47.1 | 49.2% |
| **Grand Total** | **95.6** | **100%** |

The workload is effectively split 50/50 between communication and compute. This is characteristic of MoE models with high expert counts: the all-to-all communication required to route tokens to experts across GPUs is proportional to the number of forward and backward passes.

### 5.2 NCCL Communication Breakdown

| NCCL Operation | Time (s) | % of GPU | Instances | Purpose |
|---|---|---|---|---|
| **SendRecv** | **36.3** | **37.9%** | 23,040 | MoE Expert Parallel all-to-all (token dispatch + combine) |
| AllGather (RING_LL) | 10.0 | 10.4% | 4,030 | ZeRO-3 parameter all-gather (forward pass) |
| ReduceScatter (RING_LL) | 2.2 | 2.3% | 175 | ZeRO-3 gradient reduce-scatter (backward pass) |
| AllReduce (TREE_LL) | 0.04 | 0.0% | 25 | Auxiliary loss reduction |

**MoE A2A dominates**: The SendRecv operations (37.9%) are the token dispatch and combine all-to-all collectives in the Expert Parallel dimension. With EP=8, each of the 48 layers performs a dispatch (forward) and combine (forward), plus the corresponding backward operations. Over 5 profiled iterations with 16 micro-batches each, this generates 23,040 SendRecv kernel invocations.

**ZeRO-3 AllGather is significant**: At 10.4%, the parameter all-gather (materializing full model parameters from sharded optimizer state) is the second-largest communication cost. With DP=64, each parameter shard must be gathered across all 64 ranks before use.

**Gradient ReduceScatter is small**: At 2.3%, the gradient reduction is well-amortized -- it happens once per parameter bucket per iteration, overlapped with backward compute where possible.

### 5.3 Top Compute Kernels

| Kernel | Time (s) | % of GPU | Description |
|---|---|---|---|
| cuDNN SDPA bprop | 5.3 | 5.5% | Flash Attention backward pass |
| cuDNN SDPA fprop | 2.0 | 2.1% | Flash Attention forward pass |
| nvjet GEMMs (various) | ~25 | ~26% | Expert FFN + attention projection matmuls |
| sort_chunks_by_map | 2.1 | 2.2% | MoE token routing sort |
| triton fused kernels | 1.3 | 1.3% | MoE gating (SiLU + sigmoid fusion) |
| RMSNorm (fwd + bwd) | 1.3 | 1.4% | Layer normalization |
| permute/unpermute | 1.7 | 1.7% | MoE token permutation |
| RoPE (fwd + bwd) | 0.7 | 0.7% | Rotary position embeddings |
| Cross-entropy | 0.2 | 0.2% | Loss computation |

The GEMMs (labeled `nvjet_tst_*`) comprise the bulk of compute, spanning expert FFN projections, attention QKV/output projections, and dense FFN layers. The cuDNN Flash Attention operations are efficient but represent only 7.6% of total time due to the model's relatively small attention hidden size (2048) and short sequence length (4096).

### 5.4 Implications for Further Optimization

The nsys analysis reveals three conclusions:

1. **MoE A2A is the bottleneck**: At 38% of GPU time, reducing this communication would yield the largest gains. Options include: using the HybridEP dispatcher (NVLink-aware routing), reducing EP (requires more memory per GPU), or moving to FP8 (reduces token payload size by 2x).

2. **ZeRO-3 AllGather overhead is non-trivial**: At 10.4%, the parameter all-gather is partially overlapped with compute but remains visible. This is the cost of sharding parameters across 64 DP ranks.

3. **Compute is well-optimized**: The GEMM kernels are running on tensor cores via cuBLAS/NVJet, and attention uses cuDNN Flash Attention. There is limited room to improve the compute side without changing precision (BF16 to FP8 would double tensor core throughput).

---

## 6. Appendix

### 6.1 Complete Experiment Log

| Job ID | PP | VP | MBS | GBS | CUDA Graphs | Special | TFLOPS/GPU | Iter (ms) | Peak Mem (GB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 3141 | 2 | 12 | 1 | 2048 | moe_router, moe_preprocess | Baseline | 148 | 20,400 | 38 | Success |
| 3148 | 2 | 12 | 2 | 2048 | moe_router, moe_preprocess | MBS=2 | 275 | 10,930 | 64 | Success |
| 3149 | 2 | 12 | 2 | 2048 | +attn scope | MBS=2+attn graphs | -- | -- | 121 (OOM) | OOM during graph capture |
| 3152 | 2 | 4 | 4 | 2048 | moe_router, moe_preprocess | MBS=4, VP=4 | -- | -- | 141 (OOM) | OOM |
| 3154 | 2 | 2 | 4 | 2048 | moe_router, moe_preprocess | MBS=4, VP=2 | -- | -- | 130 (OOM) | OOM |
| 3179 | 1 | -- | 4 | 2048 | none | PP=1, MBS=4 | -- | -- | 130 (OOM) | OOM (ZeRO all-gather spike) |
| 3181 | 1 | -- | 2 | 2048 | moe_router, moe_preprocess | PP=1+graphs | -- | -- | 119+14.77 (OOM) | OOM during graph capture |
| **3185** | **1** | **--** | **2** | **2048** | **none** | **PP=1, nsys profiling** | **315** | **9,570** | **119** | **Success (best)** |
| 3187 | 1 | -- | 2 | 2048 | none | NCCL_NVLS_ENABLE=1 | -- | -- | 119+ (OOM) | OOM (NVLS buffer alloc) |
| 3188 | 1 | -- | 2 | 2048 | moe_router only | Reduced graph scope | 208 | 14,500 | 125 | Success (regression) |
| 3190 | 1 | -- | 2 | 4096 | none | GBS=4096 | 317 | 19,050 | 119 | Success |

### 6.2 Launch Command: Best Configuration

```bash
export PATH=/opt/slurm/bin:$PATH
export LLMB_INSTALL=/fsx/paragao/testing-pr
export HF_TOKEN=<your_hf_token>
export SBATCH_ACCOUNT=smml
export SBATCH_PARTITION=p5en
export NEMORUN_HOME=$LLMB_INSTALL/workloads/pretrain_qwen3
source $LLMB_INSTALL/venvs/venv_3872b31f7bed/bin/activate
cd $LLMB_INSTALL/llmb_repo/qwen3/pretrain/

MBS=2 PP=1 VP=1 \
  CUDA_GRAPH="--cuda_graph_impl=none" \
  JOB_TOTAL_GPUS=64 \
  GPU_TYPE=h100 \
  MODEL_SIZE=30b \
  DTYPE=bf16 \
  ./launch.sh
```

**Notes**:
- `GPU_TYPE=h100` is used because no H200-specific profile exists; the H100 profile is compatible.
- `VP=1` triggers the `> 1` guard in `launch.sh`, preventing `-vp 1` from being passed. The `helpers.py` patch then auto-sets VP=None when PP=1.
- To enable nsys profiling, add: `ENABLE_PROFILE=true PROFILE_START_STEP=45 PROFILE_STOP_STEP=50`

### 6.3 Launch Command: Baseline Configuration

```bash
MBS=1 PP=2 VP=12 \
  JOB_TOTAL_GPUS=64 \
  GPU_TYPE=h100 \
  MODEL_SIZE=30b \
  DTYPE=bf16 \
  ./launch.sh
```

### 6.4 Key File Locations (on cluster)

| Path | Description |
|---|---|
| `/fsx/paragao/testing-pr/` | LLMB install root |
| `/fsx/paragao/testing-pr/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh` | EFA-upgraded container (36GB) |
| `/fsx/paragao/testing-pr/llmb_repo/qwen3/pretrain/launch.sh` | Launch script |
| `/fsx/paragao/testing-pr/workloads/pretrain_qwen3/Megatron-Bridge/` | Megatron-Bridge (patched) |
| `.../scripts/performance/utils/executors.py` | EFA environment variable patch |
| `.../scripts/performance/utils/helpers.py` | VP auto-disable patch (line 248-249) |
| `.../scripts/performance/configs/qwen3/workload_base_configs.py` | Parallelism presets |
| `/fsx/paragao/testing-pr/workloads/pretrain_qwen3/experiments/` | All experiment outputs |

### 6.5 nsys Profile Locations

Profiles from the best run (Job 3185, steps 45-50, all 64 ranks):

```
/fsx/paragao/testing-pr/workloads/pretrain_qwen3/experiments/
  pretrain_qwen3_30b_a3b_bf16_gpus64_tp1_pp1_cp1_vp12_ep8_mbs2_gbs2048/
    pretrain_qwen3_30b_a3b_bf16_gpus64_tp1_pp1_cp1_vp12_ep8_mbs2_gbs2048_1774307163/
      pretrain_qwen3_30b_a3b_bf16_gpus64_tp1_pp1_cp1_vp12_ep8_mbs2_gbs2048/
        nsys_profile/
          profile_{pid}_{jobid}_node{N}_rank{R}.nsys-rep  (64 files, ~137MB each)
```

To analyze: `nsys stats --report cuda_gpu_kern_sum --format csv <profile.nsys-rep>`

### 6.6 Potential Future Optimizations

| Optimization | Expected Impact | Feasibility |
|---|---|---|
| FP8 precision | 30-50% gain (2x tensor core throughput, 2x less comm volume) | High -- requires FP8-compatible model recipe |
| HybridEP dispatcher | 5-15% gain (NVLink-aware A2A routing) | Medium -- not tuned for H200, may need NVLINK_DOMAIN_SIZE=8 |
| Longer sequence length | 5-10% gain (better compute/comm ratio) | Medium -- increases activation memory |
| 16-node scaling (128 GPUs) | Enables EP=16, potentially different parallelism | Medium -- shared capacity nodes may be unreliable |
| MoE token dropping | 5-10% gain (reduces A2A volume) | Low -- changes model convergence behavior |

