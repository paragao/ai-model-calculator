#!/usr/bin/env python3
"""
Validation script for MoE memory estimation fixes.
Tests Qwen3-235B-A22B on 64 B300 GPUs (512 GPUs) against nsys profiling data.

Validation targets (from actual B300 profiling):
- PP=8, VP=4, EP=8, TP=1, MBS=1, seq=4096: should estimate ~140-160 GB (actual: 150 GB)
- PP=8, VP=2, EP=8, TP=1, MBS=1, seq=4096: should estimate ~160-180 GB (actual: 171 GB)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration.variants_config import QWEN3_MOE_VARIANTS
from configuration.project_config import *
from configuration.advanced_config import *
from utils.core_calculations import calculate_model_memory_gb, calculate_memory_for_micro

# Find Qwen3-235B-A22B
variant = next(v for v in QWEN3_MOE_VARIANTS if v["name"] == "Qwen3-235B-A22B")

# Common config (override global PP/EP/TP for this test)
_SEQ_LEN = 4096
_TOPK = 8
_N_EXPERTS = 128
_TP = 1
_EP = 8
_PP = 8


def run_scenario(vp, mbs, label, target_low, target_high):
    """Run a single memory estimation scenario."""
    layers_per_gpu = variant["layers"] // _PP  # 94 // 8 = 11 (rounds to 11)
    experts_per_gpu = _N_EXPERTS // _EP  # 128 // 8 = 16

    # 64 GPUs total (8 nodes * 8 GPUs), DP = GPUs / (PP * TP * CP * EP)
    total_gpus = 64
    dp = total_gpus // (_PP * _TP * 1 * _EP)  # = 64 / 64 = 1

    model_mem_gb = calculate_model_memory_gb(
        variant, layers_per_gpu, experts_per_gpu,
        PARAM_BYTES, VOCAB, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES,
        tp=_TP,
    )

    mem = calculate_memory_for_micro(
        micro=mbs,
        model_mem_gb=model_mem_gb,
        variant=variant,
        dp=dp,
        layers_per_gpu=layers_per_gpu,
        zero_strategy=1,
        seq_len=_SEQ_LEN,
        topk=_TOPK,
        param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF,
        optim_bytes=OPTIM_BYTES,
        ep=_EP,
        pp=_PP,
        vp=vp,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=experts_per_gpu,
    )

    total = mem["total"]
    in_range = target_low <= total <= target_high
    status = "✓ PASS" if in_range else "✗ FAIL"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  PP={_PP}, VP={vp}, EP={_EP}, TP={_TP}, MBS={mbs}, seq={_SEQ_LEN}")
    print(f"{'='*60}")
    print(f"  Model memory:     {mem['model']:.1f} GB")
    print(f"  Gradients:        {mem['grad']:.1f} GB")
    print(f"  Optimizer:        {mem['optim']:.1f} GB")
    print(f"  Activation:       {mem['activation']:.1f} GB")
    print(f"  Buffers:          {mem['buffer']:.1f} GB")
    print(f"  ---")
    print(f"  Total estimated:  {total:.1f} GB")
    print(f"  Target range:     {target_low}-{target_high} GB")
    print(f"  {status}")

    return in_range


print("=" * 60)
print(" MoE MEMORY ESTIMATION VALIDATION")
print(f" Model: Qwen3-235B-A22B (128 experts, 94 layers)")
print(f" Hardware: 64 B300 GPUs (8 nodes, 268 GB/GPU)")
print("=" * 60)

results = []
results.append(run_scenario(vp=4, mbs=1, label="Scenario 1: VP=4, MBS=1",
                           target_low=140, target_high=160))
results.append(run_scenario(vp=2, mbs=1, label="Scenario 2: VP=2, MBS=1",
                           target_low=160, target_high=180))

print(f"\n{'='*60}")
print(f" SUMMARY: {sum(results)}/{len(results)} scenarios passed")
print("=" * 60)

if not all(results):
    sys.exit(1)
