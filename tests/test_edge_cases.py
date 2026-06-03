"""
Edge case tests for memory estimation and communication modeling.
Tests PP=1, EP=1, VP=1, and dense (non-MoE) models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration.advanced_config import *
from configuration.variants_config import LLAMA_DENSE_VARIANTS, QWEN3_MOE_VARIANTS
from utils.core_calculations import calculate_model_memory_gb, calculate_memory_for_micro
from utils.phase6_pp_comm import calculate_pp_communication

# Test variants
DENSE_VARIANT = next(v for v in LLAMA_DENSE_VARIANTS if v["name"] == "Llama-3.1-8B")
MOE_VARIANT = next(v for v in QWEN3_MOE_VARIANTS if v["name"] == "Qwen3-235B-A22B")


def test_dense_model_no_moe_overhead():
    """Dense models should have zero MoE dispatch buffers and moe_frac=0."""
    layers_per_gpu = DENSE_VARIANT["layers"]  # PP=1
    model_mem = calculate_model_memory_gb(
        DENSE_VARIANT, layers_per_gpu, 0,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    mem = calculate_memory_for_micro(
        micro=1, model_mem_gb=model_mem, variant=DENSE_VARIANT, dp=8,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=0, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=1, pp=1, vp=1,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=0,
    )
    # Optimizer should be fully sharded across dp=8 (no MoE split)
    expected_optim = model_mem * OPTIM_BYTES / 8
    assert abs(mem["optim"] - expected_optim) < 0.01, f"Optim {mem['optim']} != {expected_optim}"
    # No MoE dispatch buffer (topk=0 and no experts)
    # Buffer should just be nccl base (no EP scaling since ep=1)
    assert mem["buffer"] < NCCL_MEM_BUF + 0.1, f"Buffer too high for dense: {mem['buffer']}"
    print("  ✓ Dense model: no MoE overhead")


def test_pp1_no_vp_overhead():
    """PP=1 should produce zero VP activation overhead regardless of VP setting."""
    layers_per_gpu = MOE_VARIANT["layers"]  # All layers on one GPU
    model_mem = calculate_model_memory_gb(
        MOE_VARIANT, layers_per_gpu, 16,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    mem_vp1 = calculate_memory_for_micro(
        micro=1, model_mem_gb=model_mem, variant=MOE_VARIANT, dp=1,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=8, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=8, pp=1, vp=4,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=16,
    )
    mem_novp = calculate_memory_for_micro(
        micro=1, model_mem_gb=model_mem, variant=MOE_VARIANT, dp=1,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=8, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=8, pp=1, vp=1,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=16,
    )
    # With PP=1, VP should have no effect on activation
    assert mem_vp1["activation"] == mem_novp["activation"], \
        f"PP=1: VP should not affect activation ({mem_vp1['activation']} vs {mem_novp['activation']})"
    print("  ✓ PP=1: no VP activation overhead")


def test_ep1_no_ep_scaling():
    """EP=1 should produce no NCCL EP scaling."""
    layers_per_gpu = DENSE_VARIANT["layers"]
    model_mem = calculate_model_memory_gb(
        DENSE_VARIANT, layers_per_gpu, 0,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    mem = calculate_memory_for_micro(
        micro=1, model_mem_gb=model_mem, variant=DENSE_VARIANT, dp=8,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=0, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=1, pp=1, vp=1,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=1.0,  # disable fragmentation for precise check
        experts_per_gpu=0,
    )
    # Buffer should be exactly NCCL base + 0 EP scaling + 0 a2a + 0 dispatch + 0 pp
    assert abs(mem["buffer"] - NCCL_MEM_BUF) < 0.01, \
        f"EP=1 buffer should be {NCCL_MEM_BUF}, got {mem['buffer']}"
    print("  ✓ EP=1: no NCCL EP scaling")


def test_pp1_no_pp_communication():
    """PP=1 should return zero PP communication."""
    result = calculate_pp_communication(
        layers=32, hidden_dim=4096, seq_len=4096, mbs=1,
        gbs=1024, pp=1, vp=1, dp=8, ep=1,
        dtype_bytes=2, inter_node_bw_gb=400, gpus_per_node=8
    )
    assert result["total_sends_per_step"] == 0
    assert result["total_traffic_gb"] == 0
    assert result["pipeline_bubble_pct"] == 0
    assert result["estimated_pp_time_ms"] == 0
    print("  ✓ PP=1: zero PP communication")


def test_pp_intra_node_detection():
    """PP <= gpus_per_node should be intra-node (NVLink)."""
    # PP=4 with 8 GPUs per node -> intra-node
    result = calculate_pp_communication(
        layers=32, hidden_dim=4096, seq_len=4096, mbs=1,
        gbs=1024, pp=4, vp=1, dp=2, ep=1,
        dtype_bytes=2, inter_node_bw_gb=400, gpus_per_node=8
    )
    assert result["comm_type"] == "intra-node (NVLink)", \
        f"PP=4 with 8 GPUs/node should be intra-node, got {result['comm_type']}"

    # PP=16 with 8 GPUs per node -> inter-node
    result = calculate_pp_communication(
        layers=32, hidden_dim=4096, seq_len=4096, mbs=1,
        gbs=1024, pp=16, vp=1, dp=1, ep=1,
        dtype_bytes=2, inter_node_bw_gb=400, gpus_per_node=8
    )
    assert result["comm_type"] == "inter-node (EFA)", \
        f"PP=16 with 8 GPUs/node should be inter-node, got {result['comm_type']}"
    print("  ✓ PP inter/intra-node detection correct")


def test_vp1_standard_schedule():
    """VP=1 should use standard (non-interleaved) send count: 2*(pp-1)."""
    result = calculate_pp_communication(
        layers=32, hidden_dim=4096, seq_len=4096, mbs=1,
        gbs=1024, pp=8, vp=1, dp=1, ep=1,
        dtype_bytes=2, inter_node_bw_gb=400, gpus_per_node=8
    )
    assert result["sends_per_microbatch"] == 2 * (8 - 1), \
        f"VP=1 sends should be 14, got {result['sends_per_microbatch']}"
    print("  ✓ VP=1: standard schedule send count")


def test_vp4_pp8_adds_activation_overhead():
    """VP=4, PP=8 should produce significantly more activation than VP=1, PP=8."""
    layers_per_gpu = MOE_VARIANT["layers"] // 8  # PP=8 -> 11 layers
    model_mem = calculate_model_memory_gb(
        MOE_VARIANT, layers_per_gpu, 16,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    common = dict(
        micro=1, model_mem_gb=model_mem, variant=MOE_VARIANT, dp=1,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=8, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=8, pp=8,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=1.0, experts_per_gpu=16,
    )
    mem_vp4 = calculate_memory_for_micro(**common, vp=4)
    mem_vp1 = calculate_memory_for_micro(**common, vp=1)
    diff = mem_vp4["activation"] - mem_vp1["activation"]
    assert diff >= 3.0, f"VP=4 PP=8 should add ≥3 GB activation, got +{diff:.1f} GB"
    print(f"  ✓ VP=4, PP=8: +{diff:.1f} GB activation over VP=1")


def test_vp2_more_activation_than_vp4():
    """VP=2 should produce MORE activation than VP=4 (14*5=70 > 28*2=56 layer-slots)."""
    layers_per_gpu = MOE_VARIANT["layers"] // 8
    model_mem = calculate_model_memory_gb(
        MOE_VARIANT, layers_per_gpu, 16,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    common = dict(
        micro=1, model_mem_gb=model_mem, variant=MOE_VARIANT, dp=1,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=8, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=8, pp=8,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=1.0, experts_per_gpu=16,
    )
    mem_vp2 = calculate_memory_for_micro(**common, vp=2)
    mem_vp4 = calculate_memory_for_micro(**common, vp=4)
    assert mem_vp2["activation"] > mem_vp4["activation"], \
        f"VP=2 activation ({mem_vp2['activation']:.1f}) should exceed VP=4 ({mem_vp4['activation']:.1f})"
    print(f"  ✓ VP=2 ({mem_vp2['activation']:.1f} GB) > VP=4 ({mem_vp4['activation']:.1f} GB) activation")


def test_vp4_pp8_total_memory_difference():
    """VP=4 PP=8 should add meaningful total memory vs VP=1 PP=8 (≥20 GB)."""
    layers_per_gpu = MOE_VARIANT["layers"] // 8
    model_mem = calculate_model_memory_gb(
        MOE_VARIANT, layers_per_gpu, 16,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    common = dict(
        micro=1, model_mem_gb=model_mem, variant=MOE_VARIANT, dp=1,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096,
        topk=8, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=8, pp=8,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR, experts_per_gpu=16,
    )
    mem_vp4 = calculate_memory_for_micro(**common, vp=4)
    mem_vp1 = calculate_memory_for_micro(**common, vp=1)
    diff = mem_vp4["total"] - mem_vp1["total"]
    assert diff >= 20.0, f"VP=4 PP=8 should add ≥20 GB total, got +{diff:.1f} GB"
    print(f"  ✓ VP=4, PP=8 total: +{diff:.1f} GB over VP=1")


if __name__ == "__main__":
    print("Running edge case tests...")
    test_dense_model_no_moe_overhead()
    test_pp1_no_vp_overhead()
    test_ep1_no_ep_scaling()
    test_pp1_no_pp_communication()
    test_pp_intra_node_detection()
    test_vp1_standard_schedule()
    test_vp4_pp8_adds_activation_overhead()
    test_vp2_more_activation_than_vp4()
    test_vp4_pp8_total_memory_difference()
    print("\nAll edge case tests passed! ✓")
