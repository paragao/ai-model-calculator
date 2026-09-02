"""
Inference workload configuration.

Edit these settings for each new inference analysis. These control
the workload profile, engine selection, quantization, and KV cache parameters.
"""

# Workload profile
WORKLOAD_MODE = "realtime"          # "realtime" (latency-critical) | "batch" (throughput-critical)
INFERENCE_CLASS = "conversational"  # "conversational" | "streaming" | "agentic" | "embedded" | "multimodal"

# Sequence lengths
INPUT_SEQ_LENS = [14000, 33000, 49000]  # ISL values to sweep (prompt tokens)
OUTPUT_SEQ_LEN = 1000                    # OSL (generated tokens per request)

# Concurrency sweep
CONCURRENCY_LEVELS = [1, 8, 32, 64, 128]  # Concurrent requests to test

# Engine selection
ENGINE = "vllm"  # "vllm" | "sglang" | "trtllm" | "all"

# Quantization & KV cache
QUANTIZATION = "fp8"         # "fp8" | "awq-int4" | "gptq-int4" | "nvfp4" | "none"
KV_CACHE_DTYPE = "fp8"       # "fp8" | "fp16" | "bf16" | "auto" (auto = same as quantization)

# Engine-specific parameters
MAX_NUM_SEQS = 256                # Max concurrent sequences in engine scheduler
ENABLE_CHUNKED_PREFILL = True     # Chunked prefill (vLLM/SGLang)
ENABLE_PREFIX_CACHING = True      # Prefix/radix caching
GPU_MEMORY_UTILIZATION = 0.90     # Fraction of GPU memory available for model + KV cache

# Engine overhead: fixed memory reservation per GPU (GB) for non-model, non-KV allocations.
# Includes: CUDA context (~0.5 GB), NCCL workspace/buffers (~0.25-0.5 GB),
# PagedAttention block tables / radix tree metadata (~0.1-0.3 GB),
# PyTorch allocator fragmentation (~0.5-1.0 GB), CUDA graph captures (~0.1-0.3 GB).
# Conservative default of 2.0 GB covers worst-case across vLLM/SGLang/TRT-LLM.
# Reduce to ~1.0 GB for TP=1 on small models; increase to ~3.0 GB for large CUDA graphs.
ENGINE_OVERHEAD_GB = 2.0

# Parallelism for inference (override from project_config if needed)
INFERENCE_TP = 1      # Tensor parallelism for inference
INFERENCE_PP = 1      # Pipeline parallelism for inference
INFERENCE_EP = 1      # Expert parallelism for inference (MoE only)

# Engine MFU estimates (prefill phase, empirical)
# These reflect observed compute utilization during the prefill (prompt processing) phase
ENGINE_MFU = {
    "vllm": {
        "prefill_mfu": 0.20,       # vLLM prefill MFU (TRITON_ATTN, typical)
        "decode_bw_eff": 0.70,     # HBM bandwidth efficiency during decode
        "tp_comm_overlap": 0.0,    # Fraction of TP comm overlapped with compute
        "batching_eff": 0.85,      # Batching overhead (continuous batching)
    },
    "sglang": {
        "prefill_mfu": 0.25,       # SGLang prefill MFU (RadixAttention)
        "decode_bw_eff": 0.72,     # Slightly better memory access patterns
        "tp_comm_overlap": 0.0,
        "batching_eff": 0.87,
    },
    "trtllm": {
        "prefill_mfu": 0.35,       # TRT-LLM prefill MFU (optimized CUDA kernels)
        "decode_bw_eff": 0.80,     # Best memory access patterns
        "tp_comm_overlap": 0.30,   # Partial TP comm overlap with compute
        "batching_eff": 0.90,      # Inflight batching
    },
}

# Precision bytes mapping for inference
QUANT_BYTES = {
    "fp8": 1,
    "awq-int4": 0.5,
    "gptq-int4": 0.5,
    "nvfp4": 0.5,
    "none": 2,       # BF16/FP16 default
}

# KV cache bytes per element
KV_DTYPE_BYTES = {
    "fp8": 1,
    "fp16": 2,
    "bf16": 2,
    "auto": None,     # Resolved at runtime from QUANTIZATION
}

# On-demand hourly pricing (USD, us-east-1, as of 2026-08)
# Used for cost efficiency calculations
INSTANCE_PRICING = {
    "p5.48xlarge": 98.32,
    "p5en.48xlarge": 120.14,
    "p6-b200.48xlarge": 152.89,
    "p6-b300.48xlarge": 198.45,
    "p6e-gb200.36xlarge": 165.00,
    "g5.48xlarge": 16.29,
    "g5.12xlarge": 5.67,
    "g6.48xlarge": 13.35,
    "g6.12xlarge": 4.60,
    "g6e.48xlarge": 30.13,
    "g6e.24xlarge": 16.29,
    "g6e.12xlarge": 8.15,
    "g7e.48xlarge": 42.00,
    "g7e.12xlarge": 8.29,
}
