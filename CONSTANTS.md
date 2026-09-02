# AI Model Calculator -- Magic Constants Reference

This document catalogs all empirical constants, magic numbers, and tunable
parameters used in the inference calculator, with sources and configurability
status. Constants in the training calculator (Phases 1-6) are listed at the
end for completeness.

---

## Phase 7: KV Cache & Memory

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| Engine overhead | 2.0 GB | vLLM block manager (~0.5 GB) + NCCL workspace (~0.25-0.5 GB) + PyTorch allocator fragmentation (~0.5-1.0 GB) + CUDA graph captures (~0.1-0.3 GB) | Yes -- `ENGINE_OVERHEAD_GB` in `inference_config.py` | `phase7_kv_cache.py:169` |
| GPU memory utilization | 0.90 | vLLM default `--gpu-memory-utilization` | Yes -- `GPU_MEMORY_UTILIZATION` in `inference_config.py`, or `--gpu-util` CLI flag | `inference_config.py:30` |
| KV heads min per GPU | 1 | GQA: if `kv_heads < TP`, replicate rather than shard below 1 head/GPU | No (architecture constraint) | `phase7_kv_cache.py:79` |

### Engine overhead component breakdown

| Component | Typical Size | Notes |
|-----------|-------------|-------|
| CUDA context | ~0.3-0.5 GB | Driver + runtime, per-GPU fixed |
| NCCL workspace | ~0.25-0.5 GB | Ring buffers for collectives, scales with TP |
| PagedAttention block tables | ~0.05-0.3 GB | vLLM: metadata for virtual blocks, scales with `max_num_seqs` |
| CUDA graph captures | ~0.1-0.3 GB | Captured kernels for decode, varies by model size |
| PyTorch allocator fragmentation | ~0.5-1.0 GB | Non-deterministic, depends on allocation patterns |

**Tuning guidance**: Use 1.0 GB for TP=1 on small models (<13B), 2.0 GB
(default) for most configurations, 3.0 GB for large CUDA graph captures
with TP>=4.

---

## Phase 8: Prefill / TTFT

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| x 1000 (s -> ms) | 1000 | Unit conversion: seconds to milliseconds | No (fundamental) | `phase8_prefill.py:280,289` |
| Chunked prefill overhead | 0.05 (5%) | vLLM/SGLang empirical scheduling overhead | No (hardcoded) | `phase8_prefill.py:305` |
| Prefill MFU (vLLM) | 0.20 | Observed on H100/L40S with TRITON_ATTN backend | Yes -- `ENGINE_MFU["vllm"]["prefill_mfu"]` | `inference_config.py:41` |
| Prefill MFU (SGLang) | 0.25 | RadixAttention advantage over TRITON_ATTN | Yes -- `ENGINE_MFU["sglang"]["prefill_mfu"]` | `inference_config.py:47` |
| Prefill MFU (TRT-LLM) | 0.35 | Optimized fused CUDA kernels | Yes -- `ENGINE_MFU["trtllm"]["prefill_mfu"]` | `inference_config.py:53` |
| TP comm overlap (vLLM) | 0.0 | vLLM does not overlap TP comm with compute | Yes -- `ENGINE_MFU["vllm"]["tp_comm_overlap"]` | `inference_config.py:43` |
| TP comm overlap (TRT-LLM) | 0.30 | Partial overlap via CUDA streams | Yes -- `ENGINE_MFU["trtllm"]["tp_comm_overlap"]` | `inference_config.py:55` |
| Activation dtype | 2 bytes (BF16) | Activations are BF16 even with FP8 weights | No (architecture constraint) | `phase8_prefill.py:157` |
| FLOPs per token | 2N | Kaplan et al., "Scaling Laws for Neural Language Models" | No (fundamental) | `phase8_prefill.py:52` |

---

## Phase 9: Decode Throughput / ITL

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| x 1000 (s -> ms) | 1000 | Unit conversion: seconds to milliseconds | No (fundamental) | `phase9_decode.py:179,338` |
| HBM BW efficiency (vLLM) | 0.70 | Observed decode bandwidth utilization on H100/L40S | Yes -- `ENGINE_MFU["vllm"]["decode_bw_eff"]` | `inference_config.py:42` |
| HBM BW efficiency (SGLang) | 0.72 | Slightly better memory access patterns | Yes -- `ENGINE_MFU["sglang"]["decode_bw_eff"]` | `inference_config.py:48` |
| HBM BW efficiency (TRT-LLM) | 0.80 | Best memory access patterns (fused ops) | Yes -- `ENGINE_MFU["trtllm"]["decode_bw_eff"]` | `inference_config.py:54` |
| Batching efficiency (vLLM) | 0.85 | Continuous batching overhead | Yes -- `ENGINE_MFU["vllm"]["batching_eff"]` | `inference_config.py:44` |
| Batching efficiency (SGLang) | 0.87 | Slightly lower scheduling overhead | Yes -- `ENGINE_MFU["sglang"]["batching_eff"]` | `inference_config.py:50` |
| Batching efficiency (TRT-LLM) | 0.90 | Inflight batching | Yes -- `ENGINE_MFU["trtllm"]["batching_eff"]` | `inference_config.py:56` |
| Kernels per layer | 5 | QKV proj, attention, output proj, gate+up MLP, down MLP | No (architecture-dependent) | `phase9_decode.py:161,303` |
| Kernel launch latency (TP=1) | 2.0 us | CUDA graph pipelining, minimal overhead | No (hardcoded) | `phase9_decode.py:164` |
| Kernel launch latency (TP>1) | 10.0 us | Sync barriers between compute & NCCL kernels | No (hardcoded) | `phase9_decode.py:168` |
| Midpoint approximation | ISL + OSL/2 | Average sequence length during decode generation | No (mathematical) | `phase9_decode.py:141,277` |

---

## Phase 10: TP Communication

| Constant | Default | Source | Configurable? | Location |
|----------|---------|--------|---------------|----------|
| PCIe all-reduce factor (TP=2) | 0.60 | nccl-tests: same-socket PCIe P2P, ~38 GB/s on 64 GB/s link | Yes -- `pcie_allreduce_factor_2gpu` per instance in `hardware_config.py` | `phase10_tp_comm.py:_detect_interconnect` |
| PCIe all-reduce factor (TP=4) | 0.15 | nccl-tests: cross-socket ring, NUMA penalty, ~9.6 GB/s on 64 GB/s | Yes -- `pcie_allreduce_factor_4gpu` per instance | `hardware_config.py` |
| PCIe all-reduce factor (TP=8) | 0.05 | nccl-tests: full 8-GPU ring, dual-socket, ~3.2 GB/s on 64 GB/s | Yes -- `pcie_allreduce_factor_8gpu` per instance | `hardware_config.py` |
| Launch latency (NVLink) | 8.0 us | nccl-tests H100/H200 intra-node, small messages | Yes -- `launch_latency_us_nvlink` per instance | `hardware_config.py` |
| Launch latency (PCIe) | 40.0 us | nccl-tests 8xGPU PCIe Gen4/5, ring all-reduce | Yes -- `launch_latency_us_pcie` per instance | `hardware_config.py` |
| Launch latency (EFA) | 75.0 us | EFA RTT + NCCL proxy thread overhead | Yes -- `launch_latency_us_efa` per instance | `hardware_config.py` |
| NVLink threshold | 200 GB/s | intra_node_bw_gbps > 200 implies NVLink (vs PCIe) | No (detection heuristic) | `phase10_tp_comm.py:_detect_interconnect` |
| All-reduces per layer | 2 | 1 after attention output, 1 after MLP output | No (architecture constraint) | `phase10_tp_comm.py:180` |
| Ring all-reduce volume | 2*(TP-1)/TP * msg | Standard NCCL ring all-reduce model | No (algorithm) | `phase10_tp_comm.py:125` |

### PCIe factor measurement methodology

Factors were measured using `nccl-tests` on AWS instances:

```bash
# Build nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=1

# Measure effective bus bandwidth for ring all-reduce
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <num_gpus> -n 100
```

The "busbw" column from nccl-tests output divided by per-link PCIe peak
gives the all-reduce factor. On dual-socket Intel Xeon systems (g6e):

| TP | busbw (GB/s) | PCIe peak (GB/s) | Factor |
|----|-------------|-------------------|--------|
| 2 | ~38 | 64 | 0.60 |
| 4 | ~9.6 | 64 | 0.15 |
| 8 | ~3.2 | 64 | 0.05 |

The steep drop-off at TP=4+ is due to cross-socket UPI/QPI hops and
shared PCIe root complex contention.

---

## Phase 11: EP Communication (MoE only)

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| All-to-all algorithm | Ring / recursive doubling | NCCL default selection | No | `phase11_ep_comm.py` |

---

## Phase 12: Scheduler & Batching

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| Max num seqs | 256 | vLLM default `--max-num-seqs` | Yes -- `MAX_NUM_SEQS` in `inference_config.py` | `inference_config.py:27` |

---

## Phase 13: Cost Efficiency

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| Instance pricing | Various | AWS on-demand us-east-1 (as of 2026-08) | Yes -- `INSTANCE_PRICING` in `inference_config.py` | `inference_config.py:79` |
| x 3600 (hr -> sec) | 3600 | Unit conversion: 1 hour = 3600 seconds | No (fundamental) | `phase13_cost.py:56` |
| x 3.6 (M tok factor) | 3.6 | = 3600/1e6 * 1e3, simplification of hourly_cost / (tok/s * 3600 / 1e6) | No (derived) | `phase13_cost.py:62` |

---

## Training Calculator Constants (Phases 1-6)

These are from the training calculator for reference. Not all are documented
in detail here.

| Constant | Value | Source | Configurable? | Location |
|----------|-------|--------|---------------|----------|
| Training MFU | 0.40 | Typical observed on H100 clusters | Yes -- `MFU` in `project_config.py` | `project_config.py` |
| Fragmentation factor | 1.24 | ZeRO memory analysis empirical | No (hardcoded) | `phase1_memory.py` |
| NCCL workspace multiplier | 2x | PyTorch DDP default | No (hardcoded) | `phase1_memory.py` |

---

## How to Override

All **configurable** constants can be modified without code changes:

1. **Engine/workload params** -- Edit `configuration/inference_config.py`
   - `ENGINE_OVERHEAD_GB`, `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`
   - `ENGINE_MFU` dict (prefill_mfu, decode_bw_eff, batching_eff, tp_comm_overlap)
   - `INSTANCE_PRICING` dict
2. **Hardware specs & PCIe factors** -- Edit `configuration/hardware_config.py`
   - `pcie_allreduce_factor_{2,4,8}gpu` per INFERENCE_HARDWARE entry
   - `launch_latency_us_{nvlink,pcie,efa}` per entry
3. **Model architectures** -- Edit `configuration/variants_config.py`
4. **CLI overrides** -- Use flags on `inference_calculations.py`:
   - `--gpu-util 0.85` overrides GPU_MEMORY_UTILIZATION
   - `--tp 4` overrides INFERENCE_TP
   - `--ep 2` overrides INFERENCE_EP
   - `--engine trtllm` overrides ENGINE

---

## References

1. NCCL User Guide -- Collective Operations:
   https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
2. nccl-tests repository:
   https://github.com/NVIDIA/nccl-tests
3. Kaplan et al., "Scaling Laws for Neural Language Models" (2020):
   https://arxiv.org/abs/2001.08361
4. Pope et al., "Efficiently Scaling Transformer Inference" (2022):
   https://arxiv.org/abs/2211.05102
5. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (2023):
   https://arxiv.org/abs/2309.06180
