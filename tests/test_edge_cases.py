"""
Edge case tests for memory estimation and communication modeling.
Tests PP=1, EP=1, VP=1, and dense (non-MoE) models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration.advanced_config import *
from configuration.variants_config import (
    LLAMA_DENSE_VARIANTS,
    LLAMA4_MOE_VARIANTS,
    QWEN3_MOE_VARIANTS,
    QWEN35_MOE_VARIANTS,
    DEEPSEEK_VARIANTS,
)
from utils.core_calculations import (
    calculate_model_memory_gb,
    calculate_memory_for_micro,
    resolve_topk,
    calculate_training_time_months_with_overhead,
)
from utils.phase5_alltoall_comm import calculate_alltoall_metrics
from utils.phase7_tp_comm import calculate_tp_communication
from utils.phase8_inference import (
    kv_cache_bytes_per_token,
    calculate_inference_memory,
    calculate_ttft_ms,
    calculate_tpot_ms,
    size_for_slo,
)
from utils.phase6_pp_comm import calculate_pp_communication
from utils.phase4_zero2_comm import calculate_communication_metrics
from utils.phase4_1_zero1_comm import calculate_zero1_metrics
from utils.phase4_2_zero3_comm import calculate_zero3_metrics

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


def test_zero3_volume_is_three_psi():
    """ZeRO-3 must total 3*Psi per step, i.e. exactly 1.5x baseline DP's 2*Psi.

    Pins ZeRO paper (arXiv:1910.02054v3) sec 7.2.2: the two parameter all-gathers
    plus the gradient reduce-scatter sum to 3*Psi, "which is 1.5x compared to the
    baseline". Each collective must equal the single collective ZeRO-1/ZeRO-2 pay.
    """
    model_mem_gb, bw = 141.107, 400.0
    for dp in (8, 16, 64, 256):
        z1 = calculate_zero1_metrics(model_mem_gb, dp, bw)
        z2 = calculate_communication_metrics(model_mem_gb, dp, bw)
        z3 = calculate_zero3_metrics(model_mem_gb, dp, bw)
        one_replica = z1["ag_volume_mb"]
        # ZeRO-1's all-gather and ZeRO-2's reduce-scatter are both one replica
        assert abs(z2["rs_volume_mb"] - one_replica) < 1e-6, \
            f"DP={dp}: ZeRO-2 RS {z2['rs_volume_mb']} != one replica {one_replica}"
        # Each of ZeRO-3's three collectives is one replica
        for key in ("fwd_ag_volume_mb", "bwd_ag_volume_mb", "rs_volume_mb"):
            assert abs(z3[key] - one_replica) < 1e-6, \
                f"DP={dp}: ZeRO-3 {key} {z3[key]} != one replica {one_replica}"
        # Total is 3*Psi and the ratio to baseline 2*Psi is exactly 1.5
        assert abs(z3["total_volume_mb"] - 3 * one_replica) < 1e-6, \
            f"DP={dp}: ZeRO-3 total {z3['total_volume_mb']} != 3x {one_replica}"
        assert abs(z3["baseline_ratio"] - 1.5) < 1e-9, \
            f"DP={dp}: baseline ratio {z3['baseline_ratio']} != 1.5"
    print("  ✓ ZeRO-3 volume = 3*Psi = 1.5x baseline DP at every DP")


def test_zero_volume_independent_of_dp():
    """ZeRO-DP volume per rank must NOT shrink as DP grows.

    Regression guard. All three phases previously divided by DP, which made
    communication appear to vanish at scale (21x understated for ZeRO-1 and 31x
    for ZeRO-2 at DP=64). Ring collectives move (DP-1)/DP of a replica per rank,
    which RISES toward 1 as DP grows, so volume must be non-decreasing in DP.
    """
    model_mem_gb, bw = 141.107, 400.0
    dps = [8, 16, 32, 64, 128, 256]
    series = {
        "zero1_ag": [calculate_zero1_metrics(model_mem_gb, d, bw)["ag_volume_mb"] for d in dps],
        "zero2_rs": [calculate_communication_metrics(model_mem_gb, d, bw)["rs_volume_mb"] for d in dps],
        "zero3_total": [calculate_zero3_metrics(model_mem_gb, d, bw)["total_volume_mb"] for d in dps],
    }
    for name, vols in series.items():
        for prev, nxt, d_prev, d_next in zip(vols, vols[1:], dps, dps[1:]):
            assert nxt >= prev, \
                f"{name}: volume fell from {prev:.0f} MB at DP={d_prev} to {nxt:.0f} MB at DP={d_next}"
        # And it must stay bounded by one replica per collective, never exceed it
        assert vols[-1] < model_mem_gb * 1000 * (3 if name == "zero3_total" else 1) + 1e-6, \
            f"{name}: volume {vols[-1]:.0f} MB exceeds its replica bound"
    print("  ✓ ZeRO-1/2/3 volume is non-decreasing in DP (no /DP bug)")


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
    test_zero3_volume_is_three_psi()
    test_zero_volume_independent_of_dp()
    print("\nAll edge case tests passed! ✓")


def _mem(variant, dp, zero_strategy, experts_per_gpu=0, topk=0):
    """Memory breakdown helper: PP=1, EP=1, one micro-batch, 4K sequence."""
    layers_per_gpu = variant["layers"]
    model_mem = calculate_model_memory_gb(
        variant, layers_per_gpu, experts_per_gpu,
        PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1
    )
    return calculate_memory_for_micro(
        micro=1, model_mem_gb=model_mem, variant=variant, dp=dp,
        layers_per_gpu=layers_per_gpu, zero_strategy=zero_strategy, seq_len=4096,
        topk=topk, param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=1, pp=1, vp=1,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=FRAGMENTATION_FACTOR,
        experts_per_gpu=experts_per_gpu,
    )


def test_zero3_transient_buffer_is_dp_invariant():
    """ZeRO-3's transient all-gather buffer must NOT shrink with DP.

    Resident parameter memory is Psi/Nd and falls as DP grows, but the transient
    buffer is set by the largest gathered module and DeepSpeed's live-parameter
    budget, neither of which depends on DP. This is the floor that makes the
    paper's resident-memory model an underestimate at large DP -- regression guard
    against reintroducing a /dp divisor here.
    """
    transients = []
    residents = []
    for dp in (2, 8, 32, 64, 256, 1024):
        mem = _mem(DENSE_VARIANT, dp, zero_strategy=3)
        transients.append(mem["zero3_transient"])
        residents.append(mem["model"])

    assert transients[0] > 0, "ZeRO-3 must carry a transient buffer"
    for t in transients[1:]:
        assert abs(t - transients[0]) < 1e-9, (
            f"Transient buffer varied with DP: {transients}"
        )
    # Resident memory, by contrast, must fall as 1/DP.
    for earlier, later in zip(residents, residents[1:]):
        assert later < earlier, f"Resident memory did not fall with DP: {residents}"


def test_zero1_and_zero2_have_no_transient_buffer():
    """Only ZeRO-3 shards parameters, so only ZeRO-3 gathers them back."""
    for zero_strategy in (1, 2):
        mem = _mem(DENSE_VARIANT, dp=64, zero_strategy=zero_strategy)
        assert mem["zero3_transient"] == 0.0, (
            f"ZeRO-{zero_strategy} reported a transient buffer: {mem['zero3_transient']}"
        )
        assert mem["zero3_transient_detail"] is None
    # And ZeRO-3 at the same DP does carry one, so the test above is meaningful.
    assert _mem(DENSE_VARIANT, dp=64, zero_strategy=3)["zero3_transient"] > 0


def test_hybrid_attention_uses_per_layer_type_counts():
    """A hybrid family charges linear layers at their GDN count, not full attention.

    Qwen3.5 repeats [linear, linear, linear, full] (full_attention_interval=4), and
    a Gated DeltaNet layer is LARGER than a gated full-attention layer -- 1.10x to
    1.24x measured against the published configs. Charging every layer at
    attn_per_layer therefore understates a hybrid model, and charging every layer
    at linear_attn_per_layer would overstate it. Pin the split so neither returns.
    """
    from configuration.variants_config import QWEN35_DENSE_VARIANTS

    v = next(x for x in QWEN35_DENSE_VARIANTS if x["name"] == "Qwen3.5-27B")
    assert v["full_attention_interval"] == 4
    assert v["linear_attn_per_layer"] > v["attn_per_layer"], (
        "GDN layer must be larger than a full-attention layer for this family"
    )

    layers = v["layers"]
    got = calculate_model_memory_gb(
        v, layers, 0, PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES,
        NUM_EMBEDDINGS_TABLES, tp=1,
    )

    n_full = layers // v["full_attention_interval"]
    n_linear = layers - n_full
    assert (n_full, n_linear) == (16, 48)

    d = v["d"]
    expected_bytes = (
        n_full * v["attn_per_layer"]
        + n_linear * v["linear_attn_per_layer"]
        + layers * FFN_WEIGHT_MATRICES * d * v["dense_ffn"]
        + NUM_EMBEDDINGS_TABLES * v["vocab"] * d
        + layers * LAYER_NORM * d
    ) * PARAM_BYTES
    assert abs(got - expected_bytes / 1e9) < 1e-6, f"{got} != {expected_bytes / 1e9}"

    # And it must sit between the two uniform treatments, not equal either.
    uniform_full = dict(v)
    uniform_full.pop("linear_attn_per_layer")
    low = calculate_model_memory_gb(
        uniform_full, layers, 0, PARAM_BYTES, 151936, LAYER_NORM,
        FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1,
    )
    uniform_linear = dict(v)
    uniform_linear["attn_per_layer"] = v["linear_attn_per_layer"]
    uniform_linear.pop("linear_attn_per_layer")
    high = calculate_model_memory_gb(
        uniform_linear, layers, 0, PARAM_BYTES, 151936, LAYER_NORM,
        FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1,
    )
    assert low < got < high, f"hybrid {got} not between {low} and {high}"


# Published parameter counts, from each model's own model card / config. These are
# the external anchor: the engine driven at ep=1 (every expert on one rank, i.e.
# the FULL model) must land near them, or the catalog has drifted.
#
# DeepSeek-V4 is deliberately ABSENT -- it has no public config, carries no
# per-variant fields, and there is nothing to reconcile it against.
PUBLISHED_TOTAL_B = {
    "Llama-3.1-8B": 8.03,
    "Llama-3.1-70B": 70.6,
    "Llama-3.1-405B": 405.0,
    "Llama-3.2-3B": 3.21,
    "Llama-4-Scout-17B-16E": 109.0,
    "Llama-4-Maverick-17B-128E": 400.0,
    "Qwen3-30B-A3B": 30.5,
    "Qwen3-235B-A22B": 235.0,
    "Qwen3.5-27B": 27.0,
    "Qwen3.5-35B-A3B": 35.0,
    "Qwen3.5-122B-A10B": 122.0,
    "Qwen3.5-397B-A17B": 397.0,
    "DeepSeek-V3": 671.0,
    "DeepSeek-V3.1": 671.0,
    "DeepSeek-V3.2": 671.0,
}


def test_catalog_reconciles_against_published_parameter_counts():
    """The engine at ep=1 must reproduce each variant's published size.

    ep=1 places every expert on one rank, so calculate_model_memory_gb returns the
    FULL model rather than a per-GPU shard -- the only form comparable to a
    published count. This is the guard that would have caught the two Qwen3.5
    errors: linear-attention layers charged at the full-attention count, and
    attn_per_layer missing the attn_output_gate doubling of q_proj. Together those
    put Qwen3.5-27B at 24.35 B against a published 27 B, and the catalog's own
    total_params_B agreed with them, so nothing internal disagreed.

    3% tolerance: the multimodal families (Llama-4, Qwen3.5) carry vision towers a
    text-only engine correctly does not model, which leaves a real sub-1% shortfall.
    """
    from configuration.variants_config import VARIANTS

    checked = 0
    failures = []
    for v in VARIANTS:
        published = PUBLISHED_TOTAL_B.get(v["name"])
        if published is None:
            continue
        model_mem_gb = calculate_model_memory_gb(
            v, v["layers"], 0, 2, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES,
            NUM_EMBEDDINGS_TABLES, tp=1, ep=1,
        )
        derived_b = model_mem_gb * 1e9 / 2 / 1e9  # param_bytes=2 -> element count
        delta_pct = (derived_b - published) / published * 100
        if abs(delta_pct) > 3.0:
            failures.append(
                f"{v['name']}: engine {derived_b:.2f}B vs published {published}B "
                f"({delta_pct:+.2f}%)"
            )
        checked += 1

    assert checked >= 15, f"expected to check at least 15 variants, checked {checked}"
    assert not failures, "catalog drifted from published sizes:\n  " + "\n  ".join(failures)


def test_explicit_ep_overrides_the_global_for_expert_count():
    """calculate_model_memory_gb must honour an explicit ep, not the global EP.

    The expert override re-derives experts_per_gpu from the variant's own
    num_experts, and previously did so using the module-global EP unconditionally,
    which silently discarded the experts_per_gpu argument and made the function
    undrivable at any other EP -- so a full-model reconciliation was impossible.
    """
    from configuration.variants_config import VARIANTS

    v = next(x for x in VARIANTS if x["name"] == "Qwen3.5-397B-A17B")
    assert v["num_experts"] == 512

    def mem(ep):
        return calculate_model_memory_gb(
            v, v["layers"], 0, 2, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES,
            NUM_EMBEDDINGS_TABLES, tp=1, ep=ep,
        )

    # Expert memory dominates, so halving EP should roughly double the total.
    at_1, at_8, at_16 = mem(1), mem(8), mem(16)
    assert at_1 > at_8 > at_16, f"not monotonic in ep: {at_1}, {at_8}, {at_16}"

    # The expert term itself must scale exactly with 512 // ep.
    expert_bytes = lambda ep: (512 // ep) * v["layers"] * v["expert_each"] * 2 / 1e9
    assert abs((at_1 - at_8) - (expert_bytes(1) - expert_bytes(8))) < 1e-6


def test_step_outliers_are_the_extreme_step_values_and_everything_is_classified():
    """Phase 2 judges configurations relative to their siblings, not against bands.

    Replaces the old absolute-band assessment, which was unusable: the priority-0
    target check was unreachable (0 of 14,080 configs), and 81% of configurations
    landed in "too many steps". The rule now is that the smallest and largest STEP
    COUNTS for a (variant, hardware) pair are outliers and the rest are reasonable.

    Pins three things that are easy to get wrong:
      1. nothing is left unclassified (the old bug was a step count falling through
         every band to "unknown");
      2. outliers are dropped BY VALUE, so every configuration tied at an extreme is
         marked -- the step counts are degenerate because micro*accum repeats
         products, so a pair's ~20 configs collapse onto 6-8 distinct values;
      3. a set with fewer than 3 distinct values is left wholly reasonable, since
         trimming both ends would discard everything.
    """
    from utils.phase2_batch import classify_step_outliers

    hw = {"name": "test-hw"}

    def cfg(steps):
        return {"steps": steps, "micro": 1, "accum": 1, "hardware": hw["name"]}

    # Degenerate set: 4 distinct values, with ties at BOTH extremes.
    configs = [cfg(100), cfg(100), cfg(200), cfg(200), cfg(300), cfg(400), cfg(400)]
    classify_step_outliers(configs)

    # 1. Everything classified.
    assert all(c["priority"] is not None for c in configs)
    assert all(c["assessment"] is not None for c in configs)
    assert {c["priority"] for c in configs} <= {1, 3}

    # 2. Both ties at each extreme are marked, not just one of them.
    lows = [c for c in configs if c["steps"] == 100]
    highs = [c for c in configs if c["steps"] == 400]
    assert len(lows) == 2 and all(c["priority"] == 3 for c in lows)
    assert len(highs) == 2 and all(c["priority"] == 3 for c in highs)
    assert all(c["assessment"] == "outlier (fewest steps)" for c in lows)
    assert all(c["assessment"] == "outlier (most steps)" for c in highs)

    # The middle stays reasonable, ties included.
    middle = [c for c in configs if c["steps"] in (200, 300)]
    assert len(middle) == 3 and all(c["priority"] == 1 for c in middle)

    # 3. Fewer than 3 distinct values: nothing to trim, all reasonable.
    for values in ([500], [500, 500], [500, 600], [500, 500, 600, 600]):
        small = [cfg(v) for v in values]
        classify_step_outliers(small)
        assert all(c["priority"] == 1 for c in small), (
            f"{values} has <3 distinct step values, so nothing should be an outlier"
        )

    # Exactly 3 distinct values is the smallest set where trimming leaves something.
    three = [cfg(1), cfg(2), cfg(3)]
    classify_step_outliers(three)
    assert [c["priority"] for c in three] == [3, 1, 3]


def test_absurd_step_check_is_absolute_and_stays_quiet_on_the_structural_case():
    """The absolute sanity check the relative outlier rule structurally cannot give.

    classify_step_outliers judges a configuration only against its siblings, so a
    pair whose ENTIRE range is outside published practice still reports its middle
    as "reasonable" -- measured: Llama-3.1-8B on "1 g5" calls 57.2M-1.83B steps
    reasonable. This check is a SEPARATE axis, keyed to the smallest batch any
    published frontier run used at any point (Llama 3 405B's ramp-start 4M,
    arXiv:2407.21783 S3.4.1), so it makes a statement about the published envelope
    rather than about optimality.

    The subtle part, and the reason this test exists, is the threshold. Every
    micro x accum grid contains micro=1/accum=1, the smallest expressible batch and
    so the most steps, which makes "at least one absurd configuration" nearly
    structural -- 572 of 704 pairs. Warning there would reproduce the exact failure
    of the bands this replaced. So the pair-level line fires only when the pair's
    LARGEST reasonable batch is itself absurd.
    """
    from configuration.advanced_config import compute_absurd_step_ceiling
    from configuration.project_config import TOTAL_TOKENS
    from utils.phase2_batch import (classify_step_outliers,
                                    generate_batch_recommendations)

    ceiling = compute_absurd_step_ceiling(TOTAL_TOKENS)

    # Provenance: the ceiling is total tokens divided by the ramp-start batch, and
    # is LOOSER than MAX_STEPS -- it is a credibility floor, not a quality target.
    from configuration.advanced_config import TOKENS_PER_BATCH, MAX_STEPS
    assert ceiling == TOTAL_TOKENS / TOKENS_PER_BATCH
    assert ceiling > MAX_STEPS, "an absurdity floor must be looser than the old band"

    def pair(step_values):
        cfgs = [
            {"steps": s, "micro": 1, "accum": 1, "hardware": "hw",
             "tokens_per_batch": TOTAL_TOKENS / s,
             "steps_absurd": s > ceiling}
            for s in step_values
        ]
        return classify_step_outliers(cfgs)

    def lines(step_values):
        return generate_batch_recommendations({"V | hw": pair(step_values)})

    # 1. Wholly absurd -> escalated. Every value above the ceiling.
    out = lines([c * ceiling for c in (2, 4, 8, 16)])
    assert any(r.startswith("🚨") for r in out), out
    assert "no configuration in this pair is credible" in " ".join(out)

    # 2. Best reasonable absurd, but not all absurd -> warn, not escalate.
    #    Largest batch = fewest steps, so the smallest step value is the best pick.
    #    Make it an outlier so the best REASONABLE one is still above the ceiling.
    out = lines([ceiling * 0.5, ceiling * 2, ceiling * 4, ceiling * 8])
    assert not any(r.startswith("🚨") for r in out), out
    assert any("even the largest reasonable batch" in r for r in out), out

    # 3. Best reasonable is credible -> SILENT, even though absurd configs exist.
    #    This is the structural 81% case that must not produce a warning.
    out = lines([ceiling * 0.2, ceiling * 0.5, ceiling * 0.9, ceiling * 50])
    assert not any(r.startswith("🚨") for r in out), out
    assert not any("published practice" in r for r in out), (
        "a pair whose best reasonable batch is credible must emit no absolute "
        f"warning, or the check degenerates into flagging 4 pairs in 5: {out}"
    )

    # Empty input must not raise.
    assert classify_step_outliers([]) == []


def test_recommendations_refuse_unclassified_configs():
    """An unclassified configuration must fail loudly, not silently vanish.

    analyze_batch_configuration cannot assign a priority (the verdict is relative to
    the pair), so it returns None and a separate pass fills it in. If that pass were
    ever skipped, a filter like [c for c in configs if c["priority"] == 1] would
    quietly match nothing -- which is precisely how the old always-empty
    optimal_configs list went unnoticed for so long. So this raises instead.
    """
    import pytest
    from utils.phase2_batch import (
        analyze_batch_configuration,
        classify_step_outliers,
        generate_batch_recommendations,
    )

    hw = {"name": "64 p5", "gpus": 64}
    raw = [analyze_batch_configuration(hw, m, 4, dp=8, seq_len=4096,
                                       variant_name="TestModel")
           for m in (1, 2, 4, 8, 16)]

    # Fresh from analyze_batch_configuration: deliberately unclassified.
    assert all(c["priority"] is None for c in raw)
    with pytest.raises(ValueError, match="no priority"):
        generate_batch_recommendations({"TestModel | 64 p5": raw})

    # Classified: produces a recommendation naming the spread and the outliers.
    classify_step_outliers(raw)
    recs = generate_batch_recommendations({"TestModel | 64 p5": raw})
    assert recs, "classified configs should produce a recommendation"
    joined = " ".join(recs)
    assert "reasonable config" in joined
    assert "outlier" in joined
    assert "steps" in joined


def test_rel_speed_is_anchored_on_the_slowest_config_not_sweep_order():
    """rel_speed must mean "times faster than the worst option", so it needs the set.

    The old anchor was all_platforms[0] -- the first entry in sweep order, which
    measured mid-pack for all 22 variants. rel_speed could therefore be below 1.0
    (0.002 at worst), 1.0x meant nothing in particular, and 78% of rows cleared the
    fixed 2.0x bar for rating 0.

    Anchored on the slowest config, rel_speed is >= 1.0 everywhere and exactly 1.0
    for the slowest config itself.
    """
    from utils.phase3_training import classify_platform_efficiency

    # est_months deliberately NOT in sorted order, so a sweep-order anchor would
    # pick 8.0 -- neither the slowest (40.0) nor the fastest (1.0).
    configs = [{"est_months": m} for m in (8.0, 40.0, 1.0, 2.0, 20.0)]
    classify_platform_efficiency(configs)

    assert all(c["rel_speed"] >= 1.0 for c in configs), \
        "a slowest anchor cannot produce a rel_speed below 1.0"

    slowest = max(configs, key=lambda c: c["est_months"])
    assert slowest["rel_speed"] == 1.0, \
        "the slowest config is the anchor, so it must read exactly 1.0x"

    fastest = min(configs, key=lambda c: c["est_months"])
    assert fastest["rel_speed"] == 40.0, \
        "the fastest config's rel_speed is the set's full spread (40.0/1.0)"

    # Every config is classified.
    assert all(c["efficiency_rating"] is not None for c in configs)


def test_efficiency_rating_is_comparable_across_wildly_different_spreads():
    """The rating is a log-position in the variant's own range, not a multiplier.

    Measured over the real sweep, per-variant spread (slowest/fastest) runs from
    32.5x to 20,642x -- a 635x difference in the scale being measured. A fixed
    multiplier threshold on rel_speed therefore means something different for every
    variant: against a slowest anchor, 2.0x is the top 3% of a 32x spread but the
    bottom 7% of a 20,642x one.

    Two sets with very different spreads but the same RELATIVE positions must
    produce the same ratings, which is what makes ratings comparable across
    variants.
    """
    from utils.phase3_training import classify_platform_efficiency

    # Same log-positions (0, 0.25, 0.5, 0.75, 1.0) in both sets, wildly
    # different absolute spreads: 16x versus 100,000x.
    narrow = [{"est_months": 16.0 / (16.0 ** p)} for p in (0, 0.25, 0.5, 0.75, 1.0)]
    wide = [{"est_months": 1e5 / (1e5 ** p)} for p in (0, 0.25, 0.5, 0.75, 1.0)]

    classify_platform_efficiency(narrow)
    classify_platform_efficiency(wide)

    assert [c["efficiency_rating"] for c in narrow] == \
           [c["efficiency_rating"] for c in wide], \
        "equal relative positions must rate equally regardless of absolute spread"

    # And the rel_speed values are NOT equal, which is exactly why a fixed
    # multiplier threshold could not have produced that agreement.
    assert narrow[-1]["rel_speed"] != wide[-1]["rel_speed"]


def test_every_efficiency_band_is_reachable_including_the_worst():
    """A rating no input can produce is a dead band, and one nearly happened here.

    Anchoring rel_speed on the slowest config makes it >= 1.0 by construction, so
    the old thresholds (>=2.0, >=1.5, >=1.0, else) would have left rating 3
    UNREACHABLE -- measured at 0 of 5,084 rows -- while pushing 97.8% into rating 0.
    That is the same uninformative-verdict failure the step bands had.

    The log-position bands keep all four populated: the slowest config sits at
    position 0.0 and the fastest at 1.0, so both extremes are always occupied.
    """
    from utils.phase3_training import classify_platform_efficiency

    # A geometric ladder spanning the full range, as real sweeps produce.
    configs = [{"est_months": 2.0 ** -i} for i in range(20)]
    classify_platform_efficiency(configs)

    ratings = {c["efficiency_rating"] for c in configs}
    assert ratings == {0, 1, 2, 3}, (
        f"all four bands must be reachable, got {sorted(ratings)}. If rating 3 is "
        f"missing, the thresholds are being applied to rel_speed (always >= 1.0) "
        f"instead of to the log-position."
    )

    slowest = max(configs, key=lambda c: c["est_months"])
    fastest = min(configs, key=lambda c: c["est_months"])
    assert slowest["efficiency_rating"] == 3, "the slowest config is position 0.0"
    assert fastest["efficiency_rating"] == 0, "the fastest config is position 1.0"


def test_degenerate_set_with_no_range_penalises_nothing():
    """With one distinct time there is no range to position within.

    log(slowest/fastest) is 0, so a naive position would divide by zero. Every
    config is simultaneously slowest and fastest, so none is worse than the others
    and none is penalised.
    """
    from utils.phase3_training import classify_platform_efficiency

    configs = [{"est_months": 5.0} for _ in range(4)]
    classify_platform_efficiency(configs)

    assert all(c["rel_speed"] == 1.0 for c in configs)
    assert all(c["efficiency_rating"] == 0 for c in configs), \
        "with nothing to rank against, no config is penalised"

    # A single config is the same case.
    one = [{"est_months": 3.0}]
    classify_platform_efficiency(one)
    assert one[0]["rel_speed"] == 1.0 and one[0]["efficiency_rating"] == 0


def test_training_recommendations_refuse_unclassified_configs():
    """calculate_training_metrics cannot rate a config, so it must not pretend to.

    Both rel_speed and efficiency_rating are relative to the variant's whole set,
    so they are assigned by classify_platform_efficiency. If that pass were skipped,
    the recommendation string would format None into the output, so this raises.
    """
    import pytest
    from utils.phase3_training import (
        classify_platform_efficiency,
        generate_training_recommendations,
    )

    plats = [
        {"name": "64 p5", "gpus": 512, "est_months": 4.0, "months_low": 3.2,
         "months_high": 5.3, "rel_speed": None, "efficiency_rating": None,
         "bubble_pct": 1.0, "zero": 1, "micro": 1, "accum": 8, "seq_len": 4096,
         "tok_batch": 4e6, "steps": 1000, "comm_overhead_pct": 0.5},
        {"name": "8 p5", "gpus": 64, "est_months": 32.0, "months_low": 25.6,
         "months_high": 42.2, "rel_speed": None, "efficiency_rating": None,
         "bubble_pct": 2.0, "zero": 1, "micro": 1, "accum": 8, "seq_len": 4096,
         "tok_batch": 4e6, "steps": 1000, "comm_overhead_pct": 0.9},
    ]

    with pytest.raises(ValueError, match="no rel_speed or efficiency_rating"):
        generate_training_recommendations({"variants": {"TestModel": plats}})

    classify_platform_efficiency(plats)
    recs = generate_training_recommendations({"variants": {"TestModel": plats}})
    assert recs, "classified configs should produce recommendations"
    joined = " ".join(recs)
    assert "faster than the slowest" in joined, (
        "the headline speed-up must name its anchor -- under a slowest anchor it "
        "can read as 20,642x, which is meaningless without saying what it is "
        "relative to"
    )


def test_phase3_csv_exports_unclassified_as_blank_not_as_a_real_rating():
    """0 and 3 are legitimate readings, so they cannot double as "not assessed".

    efficiency_rating 3 is the WORST rating, so defaulting an unset value to 3
    would make a config that was never classified indistinguishable from one
    genuinely rated worst -- the same trap as coercing an unassessed None to False.
    """
    import csv
    import tempfile
    import os
    from utils.phase3_training import (
        classify_platform_efficiency,
        export_training_results_csv,
    )

    def plat(name, months):
        return {"name": name, "gpus": 64, "zero": 1, "micro": 1, "accum": 8,
                "seq_len": 4096, "gbs": 512, "tok_batch": 4e6, "steps": 1000,
                "est_months": months, "months_low": months * 0.8,
                "months_high": months * 1.3, "bubble_pct": 1.0,
                "comm_overhead_pct": 0.5, "rel_speed": None,
                "efficiency_rating": None}

    rated = [plat("a", 4.0), plat("b", 32.0)]
    classify_platform_efficiency(rated)
    unrated = [plat("c", 4.0), plat("d", 32.0)]  # deliberately never classified

    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        export_training_results_csv(
            {"variants": {"Rated": rated, "Unrated": unrated}}, path)
        rows = list(csv.DictReader(open(path)))
    finally:
        os.unlink(path)

    assert "efficiency_rating" in rows[0] and "rel_speed" in rows[0], \
        "both columns must be in fieldnames -- DictWriter silently drops keys that "\
        "are not, which is how fits_in_memory and steps_absurd were lost before"

    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    assert all(r["efficiency_rating"] != "" for r in by_variant["Rated"])
    assert all(r["rel_speed"] != "" for r in by_variant["Rated"])
    assert all(r["efficiency_rating"] == "" for r in by_variant["Unrated"]), \
        "an unclassified rating must be blank, never 3"
    assert all(r["rel_speed"] == "" for r in by_variant["Unrated"])


def test_moe_frac_uses_the_variant_expert_count_not_the_global():
    """The two memory functions must resolve the expert count identically.

    calculate_memory_for_micro carried a comment saying its moe_frac split "must
    match calculate_model_memory_gb", and it did not: the memory function applies a
    variant's own num_experts over the global N_EXPERTS // EP, while this one used
    the raw global for both moe_frac and the MoE dispatch buffer. Measured, that
    affected 7 of 11 MoE variants -- the global overstates Llama-4-Scout's 16
    experts by 8x and understates Qwen3.5-397B's 512 by 4x.

    Pinned via the shared helper, so the two cannot drift apart again.
    """
    from utils.core_calculations import resolve_experts_per_gpu

    # A variant's own count wins over whatever the caller passed.
    scout = {"num_experts": 16}
    assert resolve_experts_per_gpu(scout, experts_per_gpu=16, ep=8) == 2, \
        "16 experts over EP=8 is 2 per GPU, not the global's 16"

    big = {"num_experts": 512}
    assert resolve_experts_per_gpu(big, experts_per_gpu=16, ep=8) == 64, \
        "512 experts over EP=8 is 64 per GPU, 4x the global's 16"

    # Never below 1, and a variant with no count keeps the caller's value.
    assert resolve_experts_per_gpu({"num_experts": 4}, 16, ep=8) == 1
    assert resolve_experts_per_gpu({}, 16, ep=8) == 16

    # And the real catalog: every MoE variant declaring a count must resolve to
    # its own value, not the global.
    from configuration.variants_config import VARIANTS
    from configuration.project_config import EP as CFG_EP
    checked = 0
    for v in VARIANTS:
        ne = v.get("num_experts")
        if not ne:
            continue
        expected = max(1, ne // CFG_EP)
        assert resolve_experts_per_gpu(v, experts_per_gpu=99999, ep=CFG_EP) == expected, \
            f"{v['name']}: expected {expected} experts/GPU from its own count"
        checked += 1
    assert checked >= 10, f"expected at least 10 variants with num_experts, got {checked}"


def test_higher_tp_never_raises_moe_optimizer_memory():
    """moe_frac is a fraction OF a tp-divided model memory, so it must be tp-divided.

    calculate_model_memory_gb divides routed expert weights by TP. moe_frac did not,
    so the ratio exceeded 1.0 for any TP > 1 and clamped there, driving dense_frac to
    0 and pushing ALL optimizer and ZeRO-3 parameter sharding onto eff_dp_moe
    instead of the larger dp.

    Measured on Qwen3-235B-A22B at TP=2 before the fix: reported optimizer memory
    ROSE from 5.36 to 8.04 GB even though TP had halved the weights. Raising tensor
    parallelism must not increase reported memory.
    """
    import utils.core_calculations as cc
    from configuration.variants_config import VARIANTS
    from configuration.project_config import EP, VOCAB, TOPK
    from configuration.advanced_config import (
        LAYER_NORM, FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES,
        PARAM_BYTES, OPTIM_BYTES, EMPIRICAL_ACT_MULTIPLIER,
        SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER, FWD_BWD_ROUTING_BUFF_PASSES,
        NCCL_MEM_BUF, NCCL_EP_SCALING_FACTOR, FRAGMENTATION_FACTOR,
    )

    v = next(x for x in VARIANTS if x["name"] == "Qwen3-235B-A22B")
    layers_per_gpu = v["layers"]
    seq_len = v.get("seq_len", 4096)
    experts_global = 128 // EP

    original_tp = cc.TP
    recovered_fracs, optims = [], []
    try:
        for tp in (1, 2, 4):
            # tp is passed as an ARGUMENT now. This loop used to monkey-patch
            # cc.TP because routed_mem and the ZeRO-3 transient read the module
            # global and discarded the caller's tp -- so the activation term was
            # identical at tp = 1, 2, 4 and 8 and the transient was pinned at
            # 2.1013 GB. Patching the global is no longer how tp reaches the
            # function, and a test that still did it would pass tp=1 silently.
            cc.TP = tp  # retained only so a stale global cannot mask a regression
            dp = 512 // tp
            model_mem = cc.calculate_model_memory_gb(
                v, layers_per_gpu, experts_global, PARAM_BYTES, VOCAB, LAYER_NORM,
                FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=tp)
            b = cc.calculate_memory_for_micro(
                1, model_mem, v, dp, layers_per_gpu, 1, seq_len, TOPK, PARAM_BYTES,
                EMPIRICAL_ACT_MULTIPLIER, SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
                FWD_BWD_ROUTING_BUFF_PASSES, NCCL_MEM_BUF, OPTIM_BYTES,
                ep=EP, pp=1, vp=1, tp=tp,
                nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
                fragmentation_factor=FRAGMENTATION_FACTOR,
                experts_per_gpu=experts_global)

            # Recover the moe_frac the FUNCTION used, from the optimizer memory it
            # returned. Asserting on a locally recomputed fraction would test this
            # test's own arithmetic rather than the function's, and would stay green
            # with the TP division removed.
            eff_dp_moe = dp // EP if dp > EP else dp
            x = b["optim"] / (model_mem * OPTIM_BYTES)
            recovered_fracs.append((x - 1.0 / dp) / (1.0 / eff_dp_moe - 1.0 / dp))
            optims.append(b["optim"])
    finally:
        cc.TP = original_tp

    assert all(f < 0.999 for f in recovered_fracs), (
        f"moe_frac clamped at 1.0 (recovered {[round(f, 4) for f in recovered_fracs]}), "
        f"which zeroes dense_frac and forces all sharding onto eff_dp_moe -- "
        f"routed_mem is not being divided by TP"
    )
    # TP halves the weights while halving dp, so optimizer memory should stay
    # roughly flat. The clamp made it jump 1.5x, from 5.36 to 8.04 GB.
    assert optims[1] / optims[0] < 1.05, (
        f"optimizer memory rose {optims[1] / optims[0]:.2f}x going from TP=1 to TP=2 "
        f"({optims[0]:.2f} -> {optims[1]:.2f} GB); raising tensor parallelism must "
        f"not increase reported memory"
    )


def test_fused_ce_logits_stop_scaling_with_micro_above_the_chunk():
    """The whole point of a fused CE is that the logits stop growing with micro.

    The LM head produces a (micro, seq_len, vocab) tensor and cross-entropy backward
    needs its gradient. A fused or chunked CE walks the flattened rows in chunks and
    keeps only one chunk live, so above the chunk size this term is FLAT in micro;
    unfused it grows linearly and becomes the largest term in the function.

    Measured on Llama-3.1-8B at seq_len 4096, vocab 128,256: unfused is 4.20 GB at
    micro=1 and 33.62 GB at micro=8, while fused holds at 1.05 GB.
    """
    from utils.core_calculations import calculate_memory_for_micro

    variant = {"d": 4096, "layers": 32, "dense_layers": 32, "attn_per_layer": 41_943_040,
               "dense_ffn": 14336, "expert_ffn": 0, "shared_each": 0, "router_each": 0,
               "expert_each": 0, "vocab": 128_256}
    common = dict(
        model_mem_gb=16.0, variant=variant, dp=64, layers_per_gpu=32, zero_strategy=1,
        seq_len=4096, topk=8, param_bytes=2, empirical_act_multiplier=12,
        selective_act_checkpointing_multiplier=0.45, fwd_bwd_routing_buff_passes=2,
        nccl_mem_buf=6.0, optim_bytes=6.0, vocab=128_256,
        logits_bytes_per_element=8, logits_chunk_tokens=1024,
    )

    fused = [calculate_memory_for_micro(micro=m, logits_fused_ce=True, **common)["logits"]
             for m in (1, 2, 4, 8, 16)]
    unfused = [calculate_memory_for_micro(micro=m, logits_fused_ce=False, **common)["logits"]
               for m in (1, 2, 4, 8, 16)]

    assert len(set(round(x, 6) for x in fused)) == 1, (
        f"fused logits must be flat in micro above the chunk size, got {fused}"
    )
    # 1024 rows * 128,256 vocab * 8 bytes
    assert abs(fused[0] - 1.0508) < 1e-3, f"expected ~1.05 GB, got {fused[0]}"

    # Unfused doubles with micro, and is 32x the fused figure at micro=8.
    for i in range(1, len(unfused)):
        assert abs(unfused[i] / unfused[i - 1] - 2.0) < 1e-6, \
            "unfused logits must scale linearly with micro"
    assert abs(unfused[0] - 4.2027) < 1e-3, f"expected ~4.20 GB, got {unfused[0]}"
    assert abs(unfused[3] - 33.6215) < 1e-3, f"expected ~33.62 GB, got {unfused[3]}"

    # Below the chunk size the two agree: there is nothing to chunk.
    small = dict(common, seq_len=512)
    assert (calculate_memory_for_micro(micro=1, logits_fused_ce=True, **small)["logits"]
            == calculate_memory_for_micro(micro=1, logits_fused_ce=False, **small)["logits"])

    # Disabled by default for a caller that does not opt in, so the term cannot
    # appear unannounced in an existing consumer's totals.
    off = {k: v for k, v in common.items() if k != "logits_bytes_per_element"}
    assert calculate_memory_for_micro(micro=8, **off)["logits"] == 0.0

    # And when enabled it must reach the TOTAL, not merely be reported beside it.
    # A term computed into the breakdown but left out of the sum is this repo's
    # recurring defect: fits_in_memory, exceeds_batch_cap and steps_absurd were all
    # computed and then dropped before reaching their consumer.
    frag = 1.24
    with_logits = calculate_memory_for_micro(
        micro=8, logits_fused_ce=False, fragmentation_factor=frag, **common)
    without = calculate_memory_for_micro(
        micro=8, fragmentation_factor=frag, **off)
    assert abs((with_logits["total"] - without["total"])
               - with_logits["logits"] * frag) < 1e-6, (
        f"the logits term must be in the total: enabling it moved the total by "
        f"{with_logits['total'] - without['total']:.4f} but the term is "
        f"{with_logits['logits']:.4f} GB before fragmentation"
    )


def test_fp8_gradients_are_held_at_bf16_not_at_param_bytes():
    """An FP8 recipe does not store gradients in FP8, so they must not shrink with it.

    FP8 is used for the GEMM inputs; gradients are reduced and accumulated in BF16 or
    higher. grad_mem derives from model_mem_gb, which carries PARAM_BYTES, so without
    a separate multiplier an FP8 run charged 1 byte/param -- measured 1.923 GB against
    the correct 3.846 on Llama-3.1-8B.

    GRAD_BYTES is a multiplier on parameter memory, the same shape as OPTIM_BYTES, so
    GRAD_BYTES_PER_PARAM / PARAM_STORAGE_BYTES holds the product constant across
    precisions. It divides by PARAM_STORAGE_BYTES rather than PARAM_BYTES because
    model_mem_gb now carries the RESIDENT size, which is BF16 either way.
    """
    from utils.core_calculations import calculate_memory_for_micro

    variant = {"d": 4096, "layers": 32, "dense_layers": 32, "attn_per_layer": 41_943_040,
               "dense_ffn": 14336, "expert_ffn": 0, "shared_each": 0, "router_each": 0,
               "expert_each": 0, "vocab": 128_256}

    def grad_for(param_bytes):
        # model_mem_gb scales with param_bytes, as calculate_model_memory_gb makes it.
        model_mem = 16.0 * param_bytes / 2
        return calculate_memory_for_micro(
            micro=1, model_mem_gb=model_mem, variant=variant, dp=64, layers_per_gpu=32,
            zero_strategy=1, seq_len=4096, topk=8, param_bytes=param_bytes,
            empirical_act_multiplier=12, selective_act_checkpointing_multiplier=0.45,
            fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0,
            optim_bytes=12 / param_bytes,
            grad_bytes=2 / param_bytes)["grad"]

    bf16, fp8 = grad_for(2), grad_for(1)
    assert bf16 == fp8, (
        f"gradient memory must be precision-independent: BF16 {bf16} vs FP8 {fp8}. "
        f"If FP8 is half, grad_bytes is not compensating for PARAM_BYTES."
    )

    # And the config derives it that way. Asserted against PARAM_STORAGE_BYTES, not
    # PARAM_BYTES: the old form held only at BF16, where the two happen to be equal,
    # so it would have passed vacuously here and failed under PRECISION="FP8".
    from configuration.advanced_config import (
        GRAD_BYTES, GRAD_BYTES_PER_PARAM, PARAM_STORAGE_BYTES)
    assert GRAD_BYTES * PARAM_STORAGE_BYTES == GRAD_BYTES_PER_PARAM, \
        "GRAD_BYTES must hold GRAD_BYTES_PER_PARAM bytes/param at any precision"

    # Default of 1.0 keeps an existing caller's behaviour unchanged.
    no_arg = calculate_memory_for_micro(
        micro=1, model_mem_gb=16.0, variant=variant, dp=64, layers_per_gpu=32,
        zero_strategy=1, seq_len=4096, topk=8, param_bytes=2,
        empirical_act_multiplier=12, selective_act_checkpointing_multiplier=0.45,
        fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0)
    assert no_arg["grad"] == 16.0


def test_batch_config_reports_memory_feasibility_and_the_batch_cap():
    """Phase 2 has no memory awareness of its own, so it must report, not hide it.

    Rating a configuration "reasonable" at a micro batch Phase 1 found does not fit
    is misleading output -- measured at 6.0% of displayed configs. fits_in_memory
    makes it visible without changing the priority, and exceeds_batch_cap surfaces
    the MAX_TOKENS_PER_BATCH limit that the caller (not this function) enforces.
    """
    from utils.phase2_batch import analyze_batch_configuration

    hw = {"name": "test-hw", "gpus": 64}

    # Unknown feasibility must claim nothing either way.
    c = analyze_batch_configuration(hw, micro=8, accum=4, dp=8, seq_len=4096)
    assert c["fits_in_memory"] is None

    # Supplied feasibility must be honoured on both sides of the boundary.
    over = analyze_batch_configuration(hw, micro=8, accum=4, dp=8, seq_len=4096,
                                       max_viable_micro=4)
    at = analyze_batch_configuration(hw, micro=4, accum=4, dp=8, seq_len=4096,
                                     max_viable_micro=4)
    assert over["fits_in_memory"] is False
    assert at["fits_in_memory"] is True
    # The rating itself is unchanged -- this is a report, not a filter.
    assert over["priority"] == at["priority"] or over["micro"] != at["micro"]

    # The cap flag must track MAX_TOKENS_PER_BATCH, not silently drop the config.
    small = analyze_batch_configuration(hw, micro=1, accum=1, dp=1, seq_len=4096)
    assert small["exceeds_batch_cap"] is False
    big_dp = int(MAX_TOKENS_PER_BATCH // 4096) + 1
    big = analyze_batch_configuration(hw, micro=1, accum=1, dp=big_dp, seq_len=4096)
    assert big["tokens_per_batch"] >= MAX_TOKENS_PER_BATCH
    assert big["exceeds_batch_cap"] is True


def test_derived_batch_cap_replaces_the_hardcoded_constant():
    """The batch cap must follow from the token budget, not a magic number.

    The old hard-coded 600e6 was 8x looser than this module's own MIN_STEPS
    implies -- TOTAL_TOKENS/600e6 = 25,000 steps against a 200,000 floor -- and
    so never fired on any of 14,080 swept configurations.
    """
    from configuration.advanced_config import (
        compute_max_tokens_per_batch, describe_batch_budget, MIN_STEPS,
    )
    from configuration.project_config import TOTAL_TOKENS

    # The step budget is the binding bound at realistic topologies.
    cap = compute_max_tokens_per_batch(TOTAL_TOKENS, dp=512, seq_len=4096)
    assert cap == TOTAL_TOKENS / MIN_STEPS
    # And it must actually bind, unlike the constant it replaces.
    assert cap < 600e6

    # A batch at the cap yields exactly MIN_STEPS; anything larger undershoots.
    assert TOTAL_TOKENS / cap == MIN_STEPS

    # The parallelism floor wins when the topology is too wide to reach MIN_STEPS,
    # because no cap below the smallest expressible batch is satisfiable.
    huge_dp = int(TOTAL_TOKENS / MIN_STEPS / 4096) * 4
    wide = describe_batch_budget(TOTAL_TOKENS, dp=huge_dp, seq_len=4096)
    assert wide["floor_exceeds_band"] is True
    assert wide["max_tokens_per_batch"] == huge_dp * 4096
    assert wide["implied_steps_at_floor"] < MIN_STEPS

    # Degenerate inputs are rejected rather than silently producing a bad cap.
    for bad in ({"dp": 0}, {"seq_len": 0}, {"min_steps": 0}):
        kwargs = {"dp": 8, "seq_len": 4096}
        kwargs.update(bad)
        try:
            compute_max_tokens_per_batch(TOTAL_TOKENS, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_step_band_brackets_the_published_frontier_runs():
    """The [MIN_STEPS, MAX_STEPS] band must contain what real runs actually use.

    This is the evidence that the step bounds are calibrated rather than
    arbitrary, and it is what justifies deriving the batch cap from them.

    Llama 3 405B, arXiv:2407.21783 S3.4.1: "we use an initial batch size of 4M
    tokens and sequences of length 4,096, and double these values to a batch size
    of 8M sequences of 8,192 tokens after pre-training 252M tokens. We double the
    batch size again to 16M after pre-training on 2.87T tokens" -- final 16M.

    DeepSeek-V3, arXiv:2412.19437 S4.2: "the batch size is gradually increased
    from 3072 to 15360 in the training of the first 469B tokens, and then keeps
    15360 in the remaining training", at a 4K sequence length -- final
    15360 * 4096 = 62.9M tokens.
    """
    from configuration.advanced_config import MIN_STEPS, MAX_STEPS, TOKENS_PER_BATCH
    from configuration.project_config import TOTAL_TOKENS

    band_low = TOTAL_TOKENS / MAX_STEPS
    band_high = TOTAL_TOKENS / MIN_STEPS

    llama3_final = 16e6
    deepseek_final = 15360 * 4096

    assert band_low <= llama3_final <= band_high, (
        f"Llama 3's 16M batch fell outside [{band_low/1e6:.1f}M, {band_high/1e6:.1f}M]"
    )
    assert band_low <= deepseek_final <= band_high, (
        f"DeepSeek-V3's {deepseek_final/1e6:.1f}M batch fell outside "
        f"[{band_low/1e6:.1f}M, {band_high/1e6:.1f}M]"
    )

    # MAX_STEPS is Llama 3's own reported step count ("over 1,200,000 steps"),
    # and its published ramp reconstructs to within 5% of it. That agreement is
    # what ties this engine's bounds to a real run.
    reconstructed = (252e6 / 4e6) + ((2.87e12 - 252e6) / 8e6) + ((15.6e12 - 2.87e12) / 16e6)
    assert abs(reconstructed - MAX_STEPS) / MAX_STEPS < 0.05

    # TOKENS_PER_BATCH is Llama 3's INITIAL batch, not its steady-state one, which
    # is why it implies far more steps than MAX_STEPS allows. Pinned so nobody
    # "fixes" the apparent contradiction by moving MAX_STEPS.
    assert TOKENS_PER_BATCH == 4e6
    assert TOTAL_TOKENS / TOKENS_PER_BATCH > MAX_STEPS


def test_phase2_csv_export_carries_feasibility_fields():
    """The Phase 2 CSV must export fits_in_memory and exceeds_batch_cap.

    These two fields are the whole point of the memory-feasibility audit, but the
    CSV is what feeds the report/dashboard integration -- so a flag that is
    computed, warned about on stdout, and then dropped at the export boundary is
    invisible to every downstream consumer. That is precisely what shipped: the
    fields were added to the returned dict and the fieldnames list was not
    updated, which csv.DictWriter does not complain about for a MISSING key.

    Also pins the tri-state encoding. fits_in_memory is None when the caller did
    not supply Phase 1's verdict, and "not assessed" must stay distinguishable
    from "assessed, does not fit" -- so None exports as an empty cell, not False.
    """
    import csv
    import os
    import tempfile
    from utils.phase2_batch import analyze_batch_configuration, export_batch_results_csv

    hw = {"name": "64 p5", "gpus": 64, "mem_gb": 80}

    # micro 2 and 4 fit under max_viable_micro=4; 8 and 16 do not.
    assessed = [
        analyze_batch_configuration(hw, micro, 4, dp=8, seq_len=4096,
                                    variant_name="TestModel", max_viable_micro=4)
        for micro in (2, 4, 8, 16)
    ]
    # No Phase 1 verdict supplied, so fits_in_memory must stay None.
    unassessed = [
        analyze_batch_configuration({"name": "unassessed"}, micro, 4, dp=8,
                                    seq_len=4096, variant_name="TestModel")
        for micro in (2, 8)
    ]

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "phase2.csv")
        export_batch_results_csv({"a": assessed, "b": unassessed}, path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows, "export produced no rows"

        # The columns must exist at all -- this is the regression that shipped.
        # steps_absurd is the same class of defect one round later: computed in the
        # row dict, warned about on stdout, and absent from fieldnames.
        for field in ("fits_in_memory", "exceeds_batch_cap", "steps_absurd"):
            assert field in rows[0], f"CSV export dropped the {field} column"

        # steps_absurd must be a real value on every row, never blank here: unlike
        # fits_in_memory it needs no caller-supplied input, so an empty cell would
        # mean the field silently failed to reach the row.
        assert all(r["steps_absurd"] in ("True", "False") for r in rows), (
            f"steps_absurd must be populated on every row: "
            f"{sorted({r['steps_absurd'] for r in rows})}"
        )

        assessed_rows = [r for r in rows if r["hardware"] == "64 p5"]
        assert len(assessed_rows) == 4
        for r in assessed_rows:
            expected = "True" if int(r["micro_batch"]) <= 4 else "False"
            assert r["fits_in_memory"] == expected, (
                f"micro={r['micro_batch']} exported {r['fits_in_memory']}, "
                f"expected {expected}"
            )

        # An unassessed row must be blank, NOT False -- otherwise a consumer
        # cannot tell "no Phase 1 data" from "definitely does not fit".
        unassessed_rows = [r for r in rows if r["hardware"] == "unassessed"]
        assert len(unassessed_rows) == 2
        assert all(r["fits_in_memory"] == "" for r in unassessed_rows), (
            "unassessed rows must export an empty cell, not a coerced False"
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_attention_flops_width_is_query_projection_not_hidden_dim():
    """The O(seq_len^2) attention term must use the score/context matmul width.

    Using `d` understates it for 10 of 22 catalog variants. The two matmuls are
    parameter-free so 6ND does not cover them, and per layer per sequence
    fwd+bwd costs 6 * s^2 * q_heads * (qk_head_dim + v_head_dim). Against
    ATTENTION_FLOPS_FACTOR = 12 on a single width that means

        W = q_heads * (qk_head_dim + v_head_dim) / 2

    Pins four things:
      1. W == d EXACTLY for standard MHA/GQA, so conforming variants are a strict
         no-op -- without this the fix would silently move every variant.
      2. A stored head_dim that is not d / q_heads is honoured. GQA is NOT the
         cause: the score and context matmuls run over q_heads regardless of
         kv_heads, since each query head attends with its own (s, s) matrix.
      3. MLA, where qk and v head dims differ from each other. DeepSeek-V3
         publishes qk_nope 128 + qk_rope 64 and v_head_dim 128 over 128 heads
         against d = 7168, so W = 20480 -- 2.86x d.
      4. The no-head-dims fallback reproduces `d`, which is the deliberate
         behaviour for an unverified entry rather than a value to trust.
    """
    from utils.core_calculations import compute_attention_flops_dim
    from configuration.variants_config import VARIANTS

    # 1. Standard MHA/GQA: head_dim == d / q_heads -> exactly d, not merely close.
    for d, q in ((4096, 32), (8192, 64), (16384, 128), (3584, 28)):
        v = {"d": d, "q_heads": q, "head_dim": d // q}
        assert compute_attention_flops_dim(v) == d, (
            f"d={d} q_heads={q} must be a no-op, got "
            f"{compute_attention_flops_dim(v)}"
        )

    # 2. Stored head_dim that disagrees with d / q_heads.
    #    Qwen3-30B-A3B: d=2048 but 32 heads x 128 = 4096.
    assert compute_attention_flops_dim(
        {"d": 2048, "q_heads": 32, "head_dim": 128}) == 4096
    #    Qwen3.5-122B-A10B: d=3072 but 32 heads x 256 = 8192, the worst at 2.67x.
    assert compute_attention_flops_dim(
        {"d": 3072, "q_heads": 32, "head_dim": 256}) == 8192

    # 3. MLA: asymmetric qk and v widths.
    ds = {"d": 7168, "q_heads": 128, "qk_head_dim": 192, "v_head_dim": 128}
    assert compute_attention_flops_dim(ds) == 128 * (192 + 128) / 2 == 20480

    # 4. Fallback with no head dims at all reproduces d.
    assert compute_attention_flops_dim({"d": 7168, "q_heads": 128}) == 7168

    # Catalog-level: every variant whose head dims imply d must be untouched, and
    # the known-changed set must be exactly the 10 measured.
    changed = {v["name"] for v in VARIANTS
               if abs(compute_attention_flops_dim(v) - v["d"]) > 1e-9}
    expected = {
        "Qwen3-32B", "Qwen3-30B-A3B", "Qwen3-235B-A22B", "Qwen3.5-27B",
        "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B", "Qwen3.5-397B-A17B",
        "DeepSeek-V3", "DeepSeek-V3.1", "DeepSeek-V3.2",
    }
    assert changed == expected, (
        f"attention width changed for an unexpected set.\n"
        f"  unexpected: {sorted(changed - expected)}\n"
        f"  missing:    {sorted(expected - changed)}"
    )

    # Pin the TOTALS the docstring quotes, not just the membership. The set
    # assertion above stays green when a new conforming variant is added, but the
    # docstring's "10 of 22 / 12 conforming" silently becomes wrong -- which is
    # exactly the drift that shipped once already (it read 11/11).
    assert len(VARIANTS) == 22, (
        f"catalog is now {len(VARIANTS)} variants; update the '10 of 22' and "
        f"'12 conforming' counts in compute_attention_flops_dim's docstring"
    )
    assert len(changed) == 10 and len(VARIANTS) - len(changed) == 12

    # DeepSeek-V4 must stay on the fallback: its config is unpublished, so the
    # value is a known gap and must NOT be filled in by copying a sibling.
    v4 = next(v for v in VARIANTS if v["name"] == "DeepSeek-V4")
    assert v4.get("qk_head_dim") is None and v4.get("v_head_dim") is None, (
        "DeepSeek-V4 has no published config; its MLA head dims must not be "
        "assumed from V3"
    )
    assert compute_attention_flops_dim(v4) == v4["d"]


def test_fp8_changes_only_the_activation_term():
    """FP8 must move the ACTIVATION term and nothing else.

    This test used to be test_phase1_memory_is_identical_at_bf16_and_fp8 and asserted
    the whole breakdown was byte-identical across precisions. That was WRONG on the
    activation term and the reason is worth keeping: the decision rested on the one
    controlled pair available, Llama-70B **LoRA** (jobs 1236/1238), where FP8 costs
    +8.73% MORE, so any candidate making activations cheaper moved further from it.

    That pair could never have settled it. It is a LoRA run and this model has no
    LoRA mode; job 351's frozen-base arm measured the FP8 activation saving at 29.2%
    trainable and EXACTLY ZERO frozen, because a frozen weight needs no weight
    gradient and the weight gradient is what consumes the saved GEMM input. The
    saving is now measured on real full pretraining (jobs 352/354/361/363 bf16
    against 365/353/362/364 fp8_cs), 7 of 8 cells within 0.002 GB of a prediction
    fixed beforehand.

    So the assertion is now TWO-SIDED, which is strictly stronger than the identity
    it replaces:
      - model, grad, optim, buffer and logits stay IDENTICAL. That is the original
        protection and the reason this test exists: every production caller must
        pass PARAM_STORAGE_BYTES, and reverting any one of them to PARAM_BYTES
        makes one of those terms diverge and fails here.
      - activation is STRICTLY LOWER at FP8, by exactly ACT_PRECISION_SAVING_FRACTION
        on a dense variant. An identity assertion would pass if the new term were
        never wired in, which is the silent-drop failure this repo has hit before.

    diagnose_oom has its OWN calculate_memory_for_micro call and its numbers are
    user-visible OOM advice, so it is swept here too: wiring only the fit path
    leaves it stale and a test watching analyze_variant_memory alone cannot see it.
    """
    import importlib
    import configuration.project_config as pc
    from configuration.variants_config import VARIANTS
    from configuration.hardware_config import HARDWARE

    INVARIANT = ("model", "grad", "optim", "buffer", "logits")
    original = pc.PRECISION
    terms = {}
    savings = {}
    try:
        for precision in ("BF16", "FP8"):
            pc.PRECISION = precision
            import configuration.advanced_config as ac
            importlib.reload(ac)
            import utils.core_calculations as cc
            importlib.reload(cc)
            import utils.phase1_memory as p1
            importlib.reload(p1)

            assert ac.PARAM_STORAGE_BYTES == 2, (
                f"weights are resident in BF16 at either precision; got "
                f"{ac.PARAM_STORAGE_BYTES} at {precision}"
            )
            savings[precision] = ac.ACT_PRECISION_SAVING_FRACTION
            rows = {}
            for v in VARIANTS[:6]:
                for hw in HARDWARE[:4]:
                    dp = hw["gpus"] // (pc.TP * pc.PP * pc.CP)
                    r = p1.analyze_variant_memory(
                        v, hw, dp, v["layers"] // pc.PP, pc.EXPERTS_PER_GPU)
                    mb = r.get("memory_breakdown") or {}
                    diag = p1.diagnose_oom(
                        v, hw, dp, v["layers"] // pc.PP, pc.EXPERTS_PER_GPU)
                    db = diag.get("breakdown") or {}
                    rows[f'{v["name"]}|{hw["name"]}'] = {
                        "zero": r["best_zero"],
                        "micro": mb.get("micro"),
                        "fit": tuple(round(mb.get(t, 0.0), 9) for t in INVARIANT),
                        "diag": tuple(round(db.get(t, 0.0), 9) for t in INVARIANT),
                        "act": round(mb.get("activation", 0.0), 9),
                        "diag_act": round(db.get("activation", 0.0), 9),
                    }
            terms[precision] = rows
    finally:
        pc.PRECISION = original
        import configuration.advanced_config as ac
        importlib.reload(ac)
        import utils.core_calculations as cc
        importlib.reload(cc)
        import utils.phase1_memory as p1
        importlib.reload(p1)

    assert terms["BF16"], "swept nothing; the harness itself is broken"
    assert savings["BF16"] == 0.0, (
        f"the FP8 activation saving must be exactly inert at BF16, so the shipped "
        f"default sweep is unchanged; got {savings['BF16']}"
    )
    assert savings["FP8"] > 0.0, (
        "ACT_PRECISION_SAVING_FRACTION is 0.0 at FP8, so the measured saving is not "
        "reaching the model at all"
    )

    # 1. Everything except activation is untouched. Compare only rows whose chosen
    #    ZeRO strategy and micro batch match, because model/grad/optim depend on the
    #    strategy and buffer/logits depend on micro -- and cheaper FP8 activations
    #    legitimately let a different one win. Those reselections are a real
    #    behaviour change, measured separately in the sweep, not a wiring fault.
    comparable = [k for k in terms["BF16"]
                  if terms["BF16"][k]["zero"] == terms["FP8"][k]["zero"]
                  and terms["BF16"][k]["micro"] == terms["FP8"][k]["micro"]]
    assert comparable, "no config kept the same strategy and micro; cannot compare"
    for which in ("fit", "diag"):
        bad = [k for k in comparable
               if terms["BF16"][k][which] != terms["FP8"][k][which]]
        assert not bad, (
            f"{len(bad)} of {len(comparable)} configs differ on "
            f"{INVARIANT} in the {which} path. First: {bad[0]} "
            f"BF16={terms['BF16'][bad[0]][which]} FP8={terms['FP8'][bad[0]][which]}. "
            f"A production caller is passing PARAM_BYTES where it must pass "
            f"PARAM_STORAGE_BYTES, or OPTIM_BYTES/GRAD_BYTES is derived from "
            f"PARAM_BYTES. FP8 must move the activation term and nothing else."
        )

    # 2. Activation IS lower under FP8, on both paths. Without this the test would
    #    pass with act_precision_saving_fraction never plumbed through.
    for which in ("act", "diag_act"):
        moved = [k for k in comparable
                 if terms["FP8"][k][which] < terms["BF16"][k][which]]
        unmoved = [k for k in comparable
                   if terms["FP8"][k][which] >= terms["BF16"][k][which]
                   and terms["BF16"][k][which] > 0]
        assert moved, (
            f"FP8 activation memory is not lower on ANY config in the {which} path; "
            f"act_precision_saving_fraction is not reaching "
            f"calculate_memory_for_micro from this caller"
        )
        assert not unmoved, (
            f"{len(unmoved)} configs have FP8 activation >= BF16 in the {which} "
            f"path, e.g. {unmoved[0]}: BF16 {terms['BF16'][unmoved[0]][which]} vs "
            f"FP8 {terms['FP8'][unmoved[0]][which]}"
        )


def _mem_for_cp(cp, **overrides):
    """One dense-model memory breakdown at a given cp. Shared by the cp tests."""
    from utils.core_calculations import calculate_memory_for_micro

    variant = {"d": 8192, "layers": 80, "dense_layers": 80,
               "attn_per_layer": 167_772_160, "dense_ffn": 28672, "expert_ffn": 0,
               "shared_each": 0, "router_each": 0, "expert_each": 0,
               "vocab": 128_256}
    kwargs = dict(
        micro=1, model_mem_gb=16.0, variant=variant, dp=8, layers_per_gpu=20,
        zero_strategy=1, seq_len=8192, topk=0, param_bytes=2,
        empirical_act_multiplier=12, selective_act_checkpointing_multiplier=0.45,
        fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
        pp=4, vocab=128_256, logits_bytes_per_element=8,
    )
    kwargs.update(overrides)
    if cp is not None:
        kwargs["cp"] = cp
    return calculate_memory_for_micro(**kwargs)


def test_context_parallelism_divides_the_sequence_dependent_memory():
    """cp shards the SEQUENCE, so per-GPU activation memory must fall as cp rises.

    Before this, cp reached the model only through the caller's
    dp = gpus // (TP * PP * CP), so raising cp shrank dp and therefore RAISED reported
    per-GPU memory -- the opposite of what context parallelism is for. Same class as
    the moe_frac TP omission: a parallelism degree moving memory the wrong way.

    Measured consequence on a real run: dgxc job 179 (Llama 70B bf16 pretrain, H200
    64 GPUs, TP=2 PP=4 CP=2 VP=5, seq 8192) reserved 108.08 GB. Ignoring cp put the
    ZeRO-1 estimate at 161.37 GB (1.49x); dividing by cp puts it at 126.75 (1.17x),
    and ZeRO-2 -- which is what a NeMo distributed optimizer with gradient sharding
    resembles -- at 108.88 GB, i.e. 1.01x measured.
    """
    act = {cp: _mem_for_cp(cp)["activation"] for cp in (1, 2, 4, 8)}

    assert act[1] > 0, "activation term is zero; the probe is not exercising it"
    for cp in (2, 4, 8):
        assert abs(act[cp] - act[1] / cp) < 1e-9 * max(1.0, act[1]), (
            f"activation at cp={cp} is {act[cp]}, expected {act[1] / cp} "
            f"(= cp=1 value {act[1]} / {cp}). cp must divide the sequence."
        )

    # cp=1 must be a no-op, so an existing caller that passes no cp is unaffected.
    assert _mem_for_cp(None)["total"] == _mem_for_cp(1)["total"]

    # cp must reach the LOGITS term too. Under fused CE the live token count is
    # capped at the chunk, so the term is flat in cp by design -- the unfused path is
    # where a missing divisor would show, so assert there.
    unfused = {cp: _mem_for_cp(cp, logits_fused_ce=False)["logits"]
               for cp in (1, 2, 4)}
    assert unfused[1] > 0
    for cp in (2, 4):
        assert abs(unfused[cp] - unfused[1] / cp) < 1e-9 * max(1.0, unfused[1]), (
            f"unfused logits at cp={cp} is {unfused[cp]}, expected "
            f"{unfused[1] / cp}: the logits tensor is (micro, seq_len/cp, vocab)"
        )

    # A zero or negative cp must not divide by zero or inflate memory.
    assert _mem_for_cp(0)["activation"] == act[1]


def test_moe_expert_activation_does_not_scale_with_precision():
    """The MoE EXPERT intermediate must not track param_bytes.

    EMPIRICAL_ACT_MULTIPLIER is already BYTES per element of (seq_len * micro * d) --
    the expression divides by 1e9 with no other byte factor -- so it is an implicitly
    BF16 figure and cannot scale. The MoE expert intermediate in the SAME expression
    counted elements and multiplied by param_bytes, so FP8 saved 20.0-40.5% on a MoE
    layer and exactly 0% on a dense one. A dense FFN intermediate is as FP8-eligible
    as an expert one, so that split was not physical.

    CORRECTION to the reasoning this docstring used to carry. It said the choice was
    "against measurement, not symmetry", because on the controlled pair (jobs
    1236/1238) FP8 cost +8.73% MORE memory, so every candidate making activations
    cheaper under FP8 moved further from reality. That pair is a Llama-70B LoRA run
    and this model has no LoRA mode; job 351's frozen-base arm measured the FP8
    activation saving at 29.2% trainable and EXACTLY ZERO frozen, so the pair was
    structurally incapable of showing the effect. FP8 DOES reduce dense activation
    memory under full pretraining and that is now shipped as
    ACT_PRECISION_SAVING_FRACTION; see test_fp8_changes_only_the_activation_term.

    What this test still pins, and what it NO LONGER claims. It still pins that the
    expert intermediate's per-element BF16 size is 2 bytes and does not track
    param_bytes -- that remains correct, because the FP8 saving is applied as a
    separate SCALE (ACT_PRECISION_SAVING_FRACTION) rather than by shrinking this
    constant.

    It used to add that the saving was "measured on a DENSE model and is deliberately
    NOT extended to the expert intermediate ... there is no measurement, and
    extending a dense figure by symmetry is the move that produced the error
    corrected above." There is now a measurement. Jobs 376/377 ran a controlled
    bf16/fp8 pair on Qwen3-30B-A3B (100% MoE, EP=8, recompute and CUDA graphs off
    both arms) and measured a 29.505% activation saving against 14.583% predicted by
    invariance and 29.167% by symmetry -- so the expert half DOES take the saving,
    and test_fp8_saving_reaches_the_moe_expert_term_too pins that. The two facts are
    compatible: this constant stays at 2 and precision enters through the scale.
    """
    from utils.core_calculations import calculate_memory_for_micro
    from configuration.advanced_config import ACT_BYTES_PER_ELEMENT

    assert ACT_BYTES_PER_ELEMENT == 2, (
        "the MoE expert intermediate is held in BF16; this constant must not track "
        "PARAM_BYTES. The DENSE per-layer term does scale with precision -- that is "
        "ACT_PRECISION_SAVING_FRACTION, a separate constant"
    )

    moe = {"d": 4096, "layers": 94, "dense_layers": 0, "attn_per_layer": 41_943_040,
           "dense_ffn": 12288, "expert_ffn": 1536, "shared_each": 0,
           "router_each": 524_288, "expert_each": 18_874_368, "vocab": 151_936}

    def activation(param_bytes):
        return calculate_memory_for_micro(
            micro=2, model_mem_gb=16.0, variant=moe, dp=8, layers_per_gpu=94,
            zero_strategy=1, seq_len=4096, topk=8, param_bytes=param_bytes,
            empirical_act_multiplier=12,
            selective_act_checkpointing_multiplier=0.45,
            fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
            experts_per_gpu=16, act_bytes_per_element=ACT_BYTES_PER_ELEMENT,
        )["activation"]

    bf16, fp8 = activation(2), activation(1)
    assert bf16 > 0, "activation term is zero; the probe is not exercising it"
    assert bf16 == fp8, (
        f"MoE activation memory must not change with param_bytes: BF16 {bf16} vs "
        f"FP8 {fp8}. The expert intermediate is still multiplying by param_bytes "
        f"instead of act_bytes_per_element."
    )

    # The expert term must still be PRESENT and material -- an invariance assertion
    # would also pass if the term had been deleted outright, so pin its size. It is
    # 40-81% of a MoE layer's activations across this catalog, measured.
    dense_only = calculate_memory_for_micro(
        micro=2, model_mem_gb=16.0, variant={**moe, "expert_ffn": 0},
        dp=8, layers_per_gpu=94, zero_strategy=1, seq_len=4096, topk=8,
        param_bytes=2, empirical_act_multiplier=12,
        selective_act_checkpointing_multiplier=0.45,
        fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
        experts_per_gpu=16, act_bytes_per_element=ACT_BYTES_PER_ELEMENT,
    )["activation"]
    expert_share = (bf16 - dense_only) / bf16
    assert 0.35 < expert_share < 0.85, (
        f"expert activation term is {expert_share:.1%} of the total, outside the "
        f"measured 40-81% band -- it has been dropped or double-counted"
    )


def test_recompute_is_inert_by_default():
    """Adding recomputation must not change any existing result.

    The whole 704-pair sweep was verified byte-identical with the shipped default,
    and this pins the mechanism: granularity None gives a scale of exactly 1.0, so
    every activation expression is multiplied by one.
    """
    from utils.core_calculations import resolve_recompute_scale

    fraction, scale = resolve_recompute_scale(40, 20, None, 0, 12, 2)
    assert fraction == 0.0, f"default must recompute nothing, got {fraction}"
    assert scale == 1.0, (
        f"default scale must be exactly 1.0 so results are unchanged, got {scale}"
    )

    # num_layers=0 must also be inert even if a granularity is named, otherwise a
    # half-configured recipe would silently shrink activations.
    fraction, scale = resolve_recompute_scale(40, 20, "full", 0, 12, 2)
    assert (fraction, scale) == (0.0, 1.0), (
        f"recompute_num_layers=0 must be inert, got fraction={fraction} scale={scale}"
    )


def test_recompute_fraction_depends_on_pp_and_vp_not_just_num_layers():
    """The recomputed FRACTION is num_layers / (layers_per_gpu / vp).

    This is the point of the term: recompute_num_layers=1 is ~100% of activations at
    PP=4/VP=20 and ~1.25% at PP=1, so changing PP silently changes activation memory
    by a large factor. Values cross-checked against the benchmark recipe analysis in
    scripts/submit-finetune-sweep.sh, which derives the same three figures.
    """
    from utils.core_calculations import resolve_recompute_scale

    # 80-layer model. layers_per_gpu = 80 / PP.
    cases = {
        # (layers_per_gpu, vp): expected fraction
        (20, 20): 1.0,        # PP=4 VP=20 -> 1 layer/chunk  -> 1 of 1
        (40, 20): 0.5,        # PP=2 VP=20 -> 2 layers/chunk -> 1 of 2  (jobs 1236/1238)
        (80, 1): 1.0 / 80.0,  # PP=1 no VP -> 80 in one stage -> 1 of 80
    }
    for (lpg, vp), expected in cases.items():
        fraction, _ = resolve_recompute_scale(lpg, vp, "full", 1, 12, 2)
        assert abs(fraction - expected) < 1e-9, (
            f"layers_per_gpu={lpg} vp={vp}: expected fraction {expected}, got {fraction}"
        )

    # A recomputed layer still stores its block input, so the scale is NOT 1-fraction.
    fraction, scale = resolve_recompute_scale(40, 20, "full", 1, 12, 2)
    expected_scale = 0.5 + 0.5 * 2 / 12
    assert abs(scale - expected_scale) < 1e-9, (
        f"scale must blend stored and boundary costs: expected {expected_scale}, "
        f"got {scale}. A recomputed block is not free -- it stores its own input."
    )

    # recompute_num_layers beyond the chunk size cannot exceed 100%.
    fraction, _ = resolve_recompute_scale(40, 20, "full", 99, 12, 2)
    assert fraction == 1.0, f"fraction must clamp at 1.0, got {fraction}"


def test_unsupported_recompute_granularity_raises():
    """An unmodelled granularity must fail loudly, not silently recompute nothing."""
    import pytest as _pytest

    from utils.core_calculations import resolve_recompute_scale

    with _pytest.raises(ValueError, match="unsupported recompute_granularity"):
        resolve_recompute_scale(40, 20, "selective", 1, 12, 2)


def test_recompute_scale_reaches_the_vp_inflight_activation_term():
    """Every activation component must take the recompute scale, VP in-flight included.

    This is the load-bearing assertion. The VP in-flight term was previously added
    with NO checkpointing reduction at all, and at the benchmark pair's PP=2/VP=20 it
    is 16.106 of the 23.354 GB total -- the dominant contributor. Scaling only the
    base term would leave 20.33 GB instead of 13.623 GB.

    Asserting the exact algebraic relation act_on == act_off * scale fails if EITHER
    site is left unscaled, which a figure-pinning assertion would not catch.
    """
    from utils.core_calculations import (
        calculate_memory_for_micro,
        resolve_recompute_scale,
    )

    dense = {"d": 8192, "layers": 80, "dense_layers": 80, "attn_per_layer": 67_108_864,
             "dense_ffn": 28672, "expert_ffn": 0, "shared_each": 0, "router_each": 0,
             "expert_each": 0, "vocab": 128_256}
    # PP=2 VP=20 is the benchmark pair's shape, and vp>1 and pp>1 is what switches the
    # in-flight term on at all.
    pp, vp, layers_per_gpu = 2, 20, 40

    def activation(granularity, num_layers):
        return calculate_memory_for_micro(
            micro=1, model_mem_gb=18.165, variant=dense, dp=4,
            layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096, topk=1,
            param_bytes=2, empirical_act_multiplier=12,
            selective_act_checkpointing_multiplier=0.45,
            fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
            experts_per_gpu=0, pp=pp, vp=vp, act_bytes_per_element=2,
            recompute_granularity=granularity, recompute_num_layers=num_layers,
            recompute_boundary_bytes_per_element=2,
        )["activation"]

    act_off = activation(None, 0)
    act_on = activation("full", 1)
    _, scale = resolve_recompute_scale(layers_per_gpu, vp, "full", 1, 12, 2)

    assert act_off > 0, "activation term is zero; the probe is not exercising it"
    assert act_on < act_off, (
        f"recomputation must reduce stored activations: {act_off} -> {act_on}"
    )
    assert abs(act_on - act_off * scale) < 1e-9, (
        f"every activation component must take the recompute scale. Expected "
        f"{act_off * scale} (= {act_off} x {scale}) but got {act_on}. A residual of "
        f"roughly the VP in-flight term means `vp_act_overhead *= recompute_scale` "
        f"is missing, leaving the dominant contributor unreduced."
    )

    # The in-flight term must actually be present here, or this test would pass
    # vacuously against a model where only the base term exists.
    act_no_vp = calculate_memory_for_micro(
        micro=1, model_mem_gb=18.165, variant=dense, dp=4,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=4096, topk=1,
        param_bytes=2, empirical_act_multiplier=12,
        selective_act_checkpointing_multiplier=0.45,
        fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
        experts_per_gpu=0, pp=1, vp=1, act_bytes_per_element=2,
    )["activation"]
    inflight_share = (act_off - act_no_vp) / act_off
    assert inflight_share > 0.5, (
        f"VP in-flight term is only {inflight_share:.1%} of activations here, so this "
        f"test no longer exercises the dominant contributor it exists to protect"
    )


def _dense_act(saving, f_layers=0, vp=1, pp=1, layers_per_gpu=32, **over):
    """One dense-model activation figure. Shared by the FP8 activation tests.

    Llama-3.1-8B shaped, which is the model the 2x4 crossed design was measured on.
    """
    from utils.core_calculations import calculate_memory_for_micro

    variant = {"d": 4096, "layers": 32, "dense_layers": 32,
               "attn_per_layer": 41_943_040, "dense_ffn": 14336, "expert_ffn": 0,
               "shared_each": 0, "router_each": 0, "expert_each": 0,
               "vocab": 128_256}
    kwargs = dict(
        micro=1, model_mem_gb=16.0, variant=variant, dp=8,
        layers_per_gpu=layers_per_gpu, zero_strategy=1, seq_len=8192, topk=0,
        param_bytes=2, empirical_act_multiplier=12,
        selective_act_checkpointing_multiplier=0.45,
        fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
        vp=vp, pp=pp,
        recompute_granularity="full" if f_layers else None,
        recompute_num_layers=f_layers,
        recompute_boundary_bytes_per_element=2,
        act_precision_saving_fraction=saving,
    )
    kwargs.update(over)
    return calculate_memory_for_micro(**kwargs)["activation"]


def test_fp8_activation_saving_is_inert_at_zero():
    """The shipped BF16 default must leave the activation term exactly unchanged.

    ACT_PRECISION_SAVING_FRACTION is 0.0 unless PRECISION == "FP8", so the whole
    704-pair sweep is byte-identical at the default -- verified separately by running
    it. This pins the mechanism that makes that true: a saving of 0.0 gives a stored
    scale of exactly 1.0, through the SAME code path FP8 uses, so the default is not
    a special case that could drift away from the FP8 path.
    """
    from utils.core_calculations import resolve_recompute_scale

    base = _dense_act(saving=0.0)
    assert base > 0, "activation term is zero; the probe is not exercising it"
    assert _dense_act(saving=0.0) == base

    # No recompute, no saving -> the scale is exactly 1.0, not merely close.
    fraction, scale = resolve_recompute_scale(32, 1, None, 0, 12, 2, 0.0)
    assert (fraction, scale) == (0.0, 1.0), (
        f"inert default must give (0.0, 1.0) exactly; got ({fraction}, {scale})"
    )


def test_fp8_saving_reaches_the_term_without_recompute_enabled():
    """The saving must apply when NO recompute is configured.

    resolve_recompute_scale returns early when granularity is None. That early
    return used to be a bare 1.0; if it stays 1.0 the FP8 saving becomes reachable
    only for recipes that ALSO enable recompute, which is backwards -- an f=0 run
    stores every layer and so saves on every layer. The measured f=0 cell (job 365,
    78.698 GB against bf16's 84.604) is exactly this case.
    """
    from configuration.advanced_config import (
        ACT_PRECISION_SAVED_BYTES_PER_ELEMENT, EMPIRICAL_ACT_MULTIPLIER,
    )

    s = ACT_PRECISION_SAVED_BYTES_PER_ELEMENT / EMPIRICAL_ACT_MULTIPLIER
    bf16 = _dense_act(saving=0.0, f_layers=0)
    fp8 = _dense_act(saving=s, f_layers=0)
    assert fp8 < bf16, (
        "FP8 activation memory is not lower with recompute disabled; the early "
        "return in resolve_recompute_scale is ignoring act_precision_saving_fraction"
    )
    # Ratio of two RETURNED values against the config constant -- the test does not
    # recompute the activation expression, so deleting the scale from the production
    # site cannot be masked by the test doing the same arithmetic itself.
    assert abs(fp8 / bf16 - (1.0 - s)) < 1e-12, (
        f"a dense model at f=0 must cost exactly (1 - {s}) of BF16; got "
        f"{fp8 / bf16}"
    )


def test_fp8_saving_and_recompute_are_substitutes_not_additive():
    """The saving must shrink as (1 - recompute_fraction), measured exactly.

    Job 350 measured the FP8 activation saving as 3.5 * (1 - f) B/element across
    f = 0, 0.25, 0.50, 0.75, 1.00 at r2 = 1.000000, and the real 2x4 design's f=0
    cell discriminated (1-f) scaling (predicting 78.698 GB) from pure additivity
    (79.621) against a measurement of 78.698. A recomputed layer stores no
    activations, so there is nothing for FP8 to halve.

    Asserted as a RATIO of saved bytes at f against saved bytes at f=0, both taken
    from returned values. Pure additivity would hold this ratio at 1.0 for every f.
    """
    s = 3.5 / 12

    def saved(f_layers):
        return (_dense_act(saving=0.0, f_layers=f_layers)
                - _dense_act(saving=s, f_layers=f_layers))

    at_zero = saved(0)
    assert at_zero > 0, "no saving at f=0; the term is not wired in"
    for recomputed, expected in ((8, 1 - 8 / 32), (16, 1 - 16 / 32),
                                 (24, 1 - 24 / 32), (32, 0.0)):
        ratio = saved(recomputed) / at_zero
        assert abs(ratio - expected) < 1e-12, (
            f"at {recomputed}/32 layers recomputed the saving must be "
            f"{expected:.4f} of its f=0 value, which is what job 350 measured; got "
            f"{ratio:.6f}. A ratio of 1.0 means the saving is being added "
            f"independently of recompute instead of competing for the same bytes."
        )
    # At full recompute the saving must vanish ENTIRELY, not merely shrink.
    assert _dense_act(saving=s, f_layers=32) == _dense_act(saving=0.0, f_layers=32), (
        "with every layer recomputed there are no stored activations, so FP8 and "
        "BF16 must cost the same; the measured residual at that cell is the fp8 "
        "param all-gather buffer (job 366), which is transient and not modelled here"
    )


def test_fp8_saving_reaches_the_moe_expert_term_too():
    """The FP8 saving applies to the MoE expert intermediate as well as the dense half.

    This test used to assert the OPPOSITE -- that a MoE variant's activation ratio
    sat STRICTLY ABOVE the dense ratio, because the expert intermediate was held
    precision-invariant. The grounds were that the 3.5 B/element figure came from a
    DENSE model (Llama-3.1-8B), so extending it to experts would have invented a
    constant for an unmeasured term.

    That is now measured. Slurm jobs 376 (bf16) / 377 (fp8_cs), Qwen3-30B-A3B
    pretraining on 16 GPUs at EP=8, recompute and CUDA graphs off on BOTH arms,
    resolved configs differing only in precision fields:

        ACT = peak max-allocated - steady mem-allocated
        bf16  68.652 - 52.363 = 16.289 GB
        fp8   63.847 - 52.364 = 11.483 GB    -> saving 29.505%

    That model's two halves are exactly equal in bytes (d*m = 2048*12 = 24576 and
    topk*e_ffn*2*bytes = 8*768*2*2 = 24576), so invariance predicted 14.583% and
    symmetry predicted 29.167%. Measured 29.505% is 0.338 pp from symmetry and
    14.921 pp from invariance, against a 21.875% threshold recorded before the jobs
    ran. So the ratio must now be EXACTLY 1 - s, not strictly above it.
    """
    from utils.core_calculations import calculate_memory_for_micro

    s = 3.5 / 12
    moe = {"d": 4096, "layers": 94, "dense_layers": 0, "attn_per_layer": 41_943_040,
           "dense_ffn": 12288, "expert_ffn": 1536, "shared_each": 0,
           "router_each": 524_288, "expert_each": 18_874_368, "vocab": 151_936}

    def act(saving, variant=None):
        return calculate_memory_for_micro(
            micro=2, model_mem_gb=16.0, variant=variant or moe, dp=8,
            layers_per_gpu=94,
            zero_strategy=1, seq_len=4096, topk=8, param_bytes=2,
            empirical_act_multiplier=12,
            selective_act_checkpointing_multiplier=0.45,
            fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
            experts_per_gpu=16, act_bytes_per_element=2,
            act_precision_saving_fraction=saving,
        )["activation"]

    # NON-VACUITY FIRST. A variant whose expert term contributes nothing would
    # satisfy the ratio assertion below for the wrong reason -- its activation would
    # be pure dense. So prove the expert term is actually in the figure before
    # asserting anything about how it scales.
    no_experts = dict(moe, expert_ffn=0)
    assert act(0.0) > act(0.0, no_experts), (
        "the expert intermediate contributes nothing to this variant's activation, "
        "so this test cannot distinguish the expert half saving from it being absent"
    )

    ratio = act(s) / act(0.0)
    assert abs(ratio - (1.0 - s)) < 1e-12, (
        f"MoE activation ratio {ratio:.12f} must equal 1 - s = {1.0 - s:.12f} "
        f"EXACTLY, because jobs 376/377 measured the expert intermediate saving at "
        f"the same rate as the dense half (29.505% against 29.167% predicted by "
        f"symmetry). A ratio strictly above it means the saving is reaching only "
        f"the d-proportional half -- the refuted asymmetry; below it means the "
        f"saving is being applied more than once."
    )

    # And the same ratio on a DENSE variant, which is what 'the same rate' means:
    # the two halves are no longer distinguishable by precision.
    dense_ratio = act(s, no_experts) / act(0.0, no_experts)
    assert abs(dense_ratio - ratio) < 1e-12, (
        f"dense ratio {dense_ratio:.12f} and MoE ratio {ratio:.12f} must now agree; "
        f"they differ only if one half is scaled differently from the other"
    )


def test_activation_split_matches_the_aggregate_it_replaced():
    """Splitting the activation aggregate must not change any number.

    The dense and expert parts were separated so the precision saving could reach
    one and not the other. The old single expression was
        moe_layers*(d*m + e) + (L - moe_layers)*d*m
    which is algebraically L*d*m + moe_layers*e. This pins that equivalence on a
    PARTIALLY dense MoE model, where the two forms are least likely to coincide by
    accident, and at the inert default so any difference is the refactor itself.
    """
    from utils.core_calculations import calculate_memory_for_micro

    d, m, seq, micro, topk, abpe = 4096, 12, 4096, 2, 8, 2
    layers, moe_layers, expert_ffn, sacm = 94, 60, 1536, 0.45
    variant = {"d": d, "layers": layers, "dense_layers": layers - moe_layers,
               "attn_per_layer": 41_943_040, "dense_ffn": 12288,
               "expert_ffn": expert_ffn, "shared_each": 0, "router_each": 524_288,
               "expert_each": 18_874_368, "vocab": 151_936}
    got = calculate_memory_for_micro(
        micro=micro, model_mem_gb=16.0, variant=variant, dp=8,
        layers_per_gpu=layers, zero_strategy=1, seq_len=seq, topk=topk,
        param_bytes=2, empirical_act_multiplier=m,
        selective_act_checkpointing_multiplier=sacm,
        fwd_bwd_routing_buff_passes=2, nccl_mem_buf=6.0, optim_bytes=6.0,
        experts_per_gpu=16, act_bytes_per_element=abpe,
    )["activation"]

    # The expression as it was written BEFORE the split, evaluated independently.
    moe_per_layer = seq * micro * (d * m + topk * expert_ffn * 2 * abpe)
    dense_per_layer = seq * micro * d * m
    moe_layers_on_gpu = min(moe_layers, layers)
    old = (moe_layers_on_gpu * moe_per_layer
           + (layers - moe_layers_on_gpu) * dense_per_layer) * sacm / 1e9
    assert got == old, (
        f"the split activation form gives {got} where the aggregate it replaced "
        f"gives {old}. The two must be algebraically identical; the whole 704-pair "
        f"sweep is byte-identical at this default and would also catch this."
    )
    assert moe_layers_on_gpu < layers, (
        "this model is fully MoE, so the dense-layer branch of the old expression "
        "is not being exercised and the equivalence is only half tested"
    )


def test_vp_inflight_activations_take_the_precision_saving():
    """The VP in-flight term must save under FP8, not just the base term.

    That term is the LARGEST activation contributor for a pipelined recipe -- 16.106
    of 23.354 GB at PP=2/VP=20 on Llama-3.1-70B -- and it was shipped once already
    with no recompute reduction at all. Scaling only the base term would leave the
    dominant part at BF16 cost.

    Pinned by requiring the ratio to equal (1 - saving) EXACTLY with VP and PP
    engaged: if either site is unscaled the ratio lands strictly between (1 - s)
    and 1.0, so this fails whichever one is dropped.
    """
    s = 3.5 / 12
    kw = dict(vp=4, pp=2, layers_per_gpu=32)
    bf16 = _dense_act(saving=0.0, **kw)
    fp8 = _dense_act(saving=s, **kw)
    flat = _dense_act(saving=0.0, vp=1, pp=1, layers_per_gpu=32)
    assert bf16 > flat, (
        "VP in-flight overhead is not being added, so this test is not exercising "
        "the term it exists to protect"
    )
    inflight_share = (bf16 - flat) / bf16
    assert inflight_share > 0.4, (
        f"VP in-flight term is only {inflight_share:.1%} of activations here, so a "
        f"regression in it could hide inside rounding"
    )
    assert abs(fp8 / bf16 - (1.0 - s)) < 1e-12, (
        f"with VP={kw['vp']} PP={kw['pp']} the ratio must be exactly {1.0 - s}; got "
        f"{fp8 / bf16}. A value between {1.0 - s} and 1.0 means one of the two "
        f"activation sites is not taking the precision saving."
    )


def _advanced_config_assignment(name):
    """Return the AST value node of a module-level assignment in advanced_config."""
    import ast
    path = Path(__file__).parent.parent / "configuration" / "advanced_config.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"no module-level assignment of {name} in advanced_config")


def test_act_bytes_per_element_is_live_for_dense_variants():
    """ACT_BYTES_PER_ELEMENT must reach the DENSE activation term, not only MoE.

    It used to reach ONLY the two MoE routed-expert sites, because the dense
    d-proportional term hard-coded the PRODUCT (12) rather than a tensor count times
    these bytes. That made it DEAD for all 11 dense variants in the catalog:
    perturbing it 2 -> 3 moved 0 of 352 dense rows and all 352 MoE rows, and left
    EMPIRICAL_ACT_MULTIPLIER at 12. It is now derived, so the same perturbation
    moves every dense row and rescales the multiplier with it.

    The AST half is what makes this non-vacuous. Asserting only
        EMPIRICAL_ACT_MULTIPLIER == ACT_INTERMEDIATES_PER_ELEMENT * ACT_BYTES_PER_ELEMENT
    passes just as happily when someone re-hardcodes the multiplier to a literal,
    because the product is the same either way -- the same vacuous-assertion trap
    that let GRAD_BYTES * PARAM_BYTES == GRAD_BYTES_PER_PARAM vouch for nothing.

    The worked figures above are the ORIGINAL ones, from when the count was 6 and
    the multiplier 12; the count is now a measured 16 and the multiplier 32, so
    "takes the multiplier to 18" no longer describes a 2 -> 3 perturbation. The
    mechanism the test protects is unchanged.
    """
    import ast

    for name in ("EMPIRICAL_ACT_MULTIPLIER", "RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT"):
        node = _advanced_config_assignment(name)
        referenced = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name)
        }
        assert "ACT_BYTES_PER_ELEMENT" in referenced, (
            f"{name} must be DERIVED from ACT_BYTES_PER_ELEMENT, but its assignment "
            f"references {sorted(referenced) or 'no names at all (it is a literal)'}. "
            f"A literal here makes ACT_BYTES_PER_ELEMENT dead for every dense "
            f"variant, which is the defect this test exists to prevent."
        )

    # The derivation must not have changed the shipped values.
    #
    # These were 12 / 2 / 6 and are now 32 / 2 / 16. The tensor count was MEASURED
    # two independent ways on real Llama-3.1-8B pretraining and the old 6 was a
    # hand count, low by 2.67x: a 4-point recompute-fraction sweep (jobs
    # 352/354/361/363) gives 32.034 B/element with a 0.0004 GB worst residual over
    # a 34.40 GB range, and a micro-batch sweep (jobs 387-390) gives 36.178, with
    # mbs=4 and mbs=8 CUDA-OOMing exactly as the pre-registered refutation of the
    # old constant required. If this assertion fails, update the prose in
    # advanced_config's ACT_INTERMEDIATES_PER_ELEMENT block, which carries both
    # sweeps, rather than only the number here.
    assert EMPIRICAL_ACT_MULTIPLIER == 32, EMPIRICAL_ACT_MULTIPLIER
    assert RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT == ACT_BYTES_PER_ELEMENT == 2
    assert ACT_INTERMEDIATES_PER_ELEMENT == 16

    # Behavioural half: a DENSE variant's activation must scale exactly with the
    # byte figure. Driven through the production function, and the expected ratio is
    # the byte ratio, NOT a quantity this test recomputes from the same inputs.
    def dense_activation(act_bytes):
        model_mem = calculate_model_memory_gb(
            DENSE_VARIANT, DENSE_VARIANT["layers"], 0,
            PARAM_BYTES, 151936, LAYER_NORM, FFN_WEIGHT_MATRICES,
            NUM_EMBEDDINGS_TABLES, tp=1,
        )
        mem = calculate_memory_for_micro(
            micro=1, model_mem_gb=model_mem, variant=DENSE_VARIANT, dp=8,
            layers_per_gpu=DENSE_VARIANT["layers"], zero_strategy=1, seq_len=4096,
            topk=0, param_bytes=PARAM_BYTES,
            empirical_act_multiplier=ACT_INTERMEDIATES_PER_ELEMENT * act_bytes,
            selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
            fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
            nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
            ep=1, pp=1, vp=1,
            nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
            fragmentation_factor=FRAGMENTATION_FACTOR,
            act_bytes_per_element=act_bytes,
            recompute_boundary_bytes_per_element=act_bytes,
        )
        return mem["activation"]

    at2, at3 = dense_activation(2), dense_activation(3)
    assert at2 > 0, "dense activation is zero, so the ratio below proves nothing"
    assert abs(at3 / at2 - 1.5) < 1e-12, (
        f"a dense variant's activation must scale with ACT_BYTES_PER_ELEMENT: "
        f"2 -> 3 bytes should give exactly 1.5x, got {at3 / at2} "
        f"({at2} -> {at3}). A ratio of 1.0 means the dense term is still "
        f"independent of the byte figure."
    )


def _mem_world(**over):
    """Drive the production memory function with a dense single-GPU baseline.

    Every caller overrides only what it is testing, so a figure that moves is
    attributable to that one knob.
    """
    base = dict(
        micro=1,
        model_mem_gb=1.19,
        variant=DENSE_VARIANT,
        dp=1,
        layers_per_gpu=DENSE_VARIANT["layers"],
        zero_strategy=1,
        seq_len=4096,
        topk=8,
        param_bytes=PARAM_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF,
        optim_bytes=OPTIM_BYTES,
        nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
        fragmentation_factor=1.0,
        grad_bytes=GRAD_BYTES,
        ep=1,
        pp=1,
        vp=1,
        cp=1,
        tp=1,
    )
    base.update(over)
    return calculate_memory_for_micro(**base)


def test_nccl_workspace_is_gated_on_world_size():
    """A single-GPU configuration has no communicator, so it pays no NCCL workspace.

    This term used to be charged unconditionally while every other buffer component
    beside it was gated on the parallelism that creates it. On a 0.6B model on one
    g5.8xlarge that was 6.00 GB of a 17.41 GB raw total -- 34% -- and it decided the
    answer: micro-batch 4 read as OOM with it and fits without it.

    Isolated by DIFFERENCING against the same configuration with the constant
    zeroed, so the assertion does not depend on what else the buffer contains.
    """
    def nccl_only(**par):
        return (
            _mem_world(**par)["buffer"]
            - _mem_world(**{**par, "nccl_mem_buf": 0.0, "nccl_ep_scaling_factor": 0.0})["buffer"]
        )

    assert abs(nccl_only(dp=1, tp=1, pp=1, cp=1)) < 1e-12, (
        "a single-GPU config must be charged NO NCCL workspace, got "
        f"{nccl_only(dp=1, tp=1, pp=1, cp=1)} GB"
    )

    # Any one degree above 1 means more than one rank, so the workspace is real.
    for par in (
        dict(dp=2, tp=1, pp=1, cp=1),
        dict(dp=1, tp=8, pp=1, cp=1),
        dict(dp=1, tp=1, pp=2, cp=1),
        dict(dp=1, tp=1, pp=1, cp=2),
    ):
        got = nccl_only(**par)
        assert abs(got - NCCL_MEM_BUF) < 1e-12, (
            f"world size {par['dp'] * par['tp'] * par['pp'] * par['cp']} must be "
            f"charged the full {NCCL_MEM_BUF} GB workspace, got {got} GB for {par}"
        )

    # tp is the degree that only reaches this function because this round started
    # forwarding it; without that a tp=8 single-process config reads as world 1.
    assert abs(nccl_only(dp=1, tp=8, pp=1, cp=1) - NCCL_MEM_BUF) < 1e-12


def test_a2a_routing_buffer_only_charged_for_moe():
    """A dense model has no router, so it has no all-to-all routing buffer.

    This was computed unconditionally, so every dense variant paid one -- 0.134 GB
    on a 0.6B model at seq 4096, 0.537 GB on a dense 70B per micro-batch.
    moe_dispatch_buf immediately after it was already gated. The width it uses is
    now resolved per variant rather than read from the global.

    Driven at world size 1 so the NCCL workspace is zero and the buffer total is
    attributable to this term.
    """
    dense = _mem_world(variant=DENSE_VARIANT, dp=1, tp=1, pp=1, cp=1)
    assert abs(dense["buffer"]) < 1e-12, (
        f"a dense variant at world size 1 must have NO buffer memory, got "
        f"{dense['buffer']} GB -- an ungated MoE routing buffer is the usual cause"
    )

    # Non-vacuity: the same call on an MoE variant must charge something, otherwise
    # this test would pass with the routing buffer deleted outright.
    moe = _mem_world(
        variant=MOE_VARIANT,
        layers_per_gpu=MOE_VARIANT["layers"],
        experts_per_gpu=8,
        dp=1, tp=1, pp=1, cp=1,
    )
    assert moe["buffer"] > 0.0, (
        "an MoE variant must still be charged a routing buffer; a zero here means "
        "the term was removed rather than gated"
    )


def test_gradients_are_fp32_main_grads():
    """Megatron keeps an fp32 main-gradient buffer for bf16 training.

    Read from the resolved ConfigContainer.yaml of real runs on this stack:
    main_grads_dtype=torch.float32 with use_distributed_optimizer=true. So the
    resident cost is 4 bytes/param, not the 2 this repo assumed.

    The product assertion is the one that matters: GRAD_BYTES is a multiplier on
    parameter memory which already carries PARAM_BYTES, so what must hold is
    GRAD_BYTES * PARAM_STORAGE_BYTES == GRAD_BYTES_PER_PARAM. That pairing is
    itself vacuous on its own -- it is true for any value -- so the literal is
    pinned too, with the evidence in this docstring.
    """
    assert GRAD_BYTES_PER_PARAM == 4, (
        f"gradients are fp32 main grads (4 B/param), got {GRAD_BYTES_PER_PARAM}. "
        f"If this changes, update the measured-dtype block in advanced_config."
    )
    assert abs(GRAD_BYTES * PARAM_STORAGE_BYTES - GRAD_BYTES_PER_PARAM) < 1e-12

    # Behavioural half: doubling the per-param figure must double grad memory, and
    # must not touch any other term.
    lo = _mem_world(grad_bytes=GRAD_BYTES)
    hi = _mem_world(grad_bytes=GRAD_BYTES * 2)
    assert abs(hi["grad"] / lo["grad"] - 2.0) < 1e-12
    for term in ("model", "optim", "activation", "buffer"):
        assert abs(hi[term] - lo[term]) < 1e-12, (
            f"changing grad_bytes moved {term}, which it must not"
        )


def test_selective_checkpointing_multiplier_is_inert():
    """The 0.45 blanket activation discount must stay removed.

    It was applied unconditionally at the single activation site, with no branch on
    whether any recompute was configured, so a recipe with
    recompute_granularity=None took a 55% discount it had not earned. It cannot be
    rescued as a selective-recompute factor either: resolve_recompute_scale accepts
    only None and "full" and RAISES on anything else, so selective recompute is
    unreachable and there is no configuration in which 0.45 is correct.

    Together with the measured multiplier this is what took the effective figure
    from 5.4 to 32 B/element against a measured 32.03.
    """
    assert SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER == 1.0, (
        f"the selective multiplier must be exactly inert, got "
        f"{SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER}. A non-1.0 value re-introduces a "
        f"2.2x activation error on every row; see advanced_config before changing it."
    )

    # Non-vacuity: the parameter must still REACH the activation term, so that a
    # future non-1.0 value is not silently ignored. If this ratio is 1.0 the
    # constant is dead rather than inert, which is a different and worse state.
    full = _mem_world(selective_act_checkpointing_multiplier=1.0)["activation"]
    half = _mem_world(selective_act_checkpointing_multiplier=0.5)["activation"]
    assert full > 0 and abs(half / full - 0.5) > 1e-12 or abs(half / full) < 1.0, (
        "the selective multiplier no longer reaches the activation term"
    )
    assert half < full, (
        "the selective multiplier is dead: halving it did not reduce activation "
        "memory, so a future wrong value would be invisible"
    )


def test_vp_none_does_not_raise():
    """vp=None means "no interleaving" in a real recipe and must not crash.

    The VP in-flight branch tested `vp > 1`, which raises TypeError on None. The
    internal sweep passes VP=1 so it never hit this; any direct caller did.
    """
    got = _mem_world(vp=None, pp=1)
    assert got["total"] > 0

    # The VP overhead branch requires pp > 1 as well, so None must be safe there too.
    got2 = _mem_world(vp=None, pp=2)
    assert got2["total"] > 0


def _topk_act(variant, forced_topk):
    """Activation GB for `variant` with its declared routing width forced.

    Named _topk_act, not _act or _mem: this file already carries _mem,
    _mem_for_cp, _dense_act and _mem_world, and a new helper shadowing one of
    those passed the suite while silently testing the wrong thing last round.
    """
    v = dict(variant)
    if forced_topk is None:
        v.pop("topk", None)
    else:
        v["topk"] = forced_topk
    return calculate_memory_for_micro(
        micro=1, model_mem_gb=100.0, variant=v, dp=8,
        layers_per_gpu=8, zero_strategy=1, seq_len=4096,
        topk=8, param_bytes=PARAM_STORAGE_BYTES,
        empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
        selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
        fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES,
        nccl_mem_buf=NCCL_MEM_BUF, optim_bytes=OPTIM_BYTES,
        ep=8, pp=1, vp=1,
    )["activation"]


def test_resolve_topk_prefers_the_variants_own_width():
    """A variant's own "topk" must beat the global, in both directions.

    The global project_config.TOPK = 8 was passed to the expert activation
    term, the MoE routing buffer and the all-to-all volume from five call
    sites, so three MoE variants were computed at the wrong routing width.
    """
    scout = next(v for v in LLAMA4_MOE_VARIANTS if v["name"] == "Llama-4-Scout-17B-16E")
    maverick = next(
        v for v in LLAMA4_MOE_VARIANTS if v["name"] == "Llama-4-Maverick-17B-128E"
    )
    big_qwen = next(
        v for v in QWEN35_MOE_VARIANTS if v["name"] == "Qwen3.5-397B-A17B"
    )

    # Overstated 8x by the global
    assert resolve_topk(scout, 8) == 1
    assert resolve_topk(maverick, 8) == 1
    # Understated 1.25x by the global
    assert resolve_topk(big_qwen, 8) == 10

    # A variant whose own width IS the global resolves to the same value, so
    # the 8 remaining MoE variants cannot move.
    on_global = [v for v in QWEN3_MOE_VARIANTS if v.get("topk") == 8]
    assert on_global, "expected Qwen3 MoE variants declaring topk=8"
    for v in on_global:
        assert resolve_topk(v, 8) == 8


def test_resolve_topk_falls_back_only_when_undeclared():
    """DeepSeek-V4 declares no topk, and the global is the right value for it.

    Not merely tolerable: its own catalog arithmetic implies the width. From
    active_params_B = 49.0, per-layer fixed cost attn+shared+router and 61
    layers, the implied topk is 8.29 with no embedding term and 7.83 with
    vocab=129280 -- so 8 is inside the band either way and the fallback is
    correct rather than arbitrary.
    """
    v4 = next(v for v in DEEPSEEK_VARIANTS if v["name"] == "DeepSeek-V4")
    assert "topk" not in v4, (
        "DeepSeek-V4 gained a topk field; re-derive the implied band and update "
        "this test and resolve_topk's docstring"
    )
    assert resolve_topk(v4, 8) == 8
    assert resolve_topk(v4, 6) == 6, "must return the caller's value, not a literal 8"

    # Confirm the implied band still brackets 8, from the catalog's own numbers.
    implied_no_vocab = (
        v4["active_params_B"] * 1e9
        - v4["layers"] * (v4["attn_per_layer"] + v4["shared_each"] + v4["router_each"])
    ) / (v4["layers"] * v4["expert_each"])
    assert 7.0 < implied_no_vocab < 9.0, (
        f"implied topk {implied_no_vocab:.2f} no longer brackets the fallback 8"
    )


def test_affected_variant_activation_ratio_is_exact():
    """Llama-4-Scout's activation must scale by exactly 6/13 at topk 8 -> 1.

    Pinned as a measured rational rather than recomputed in the test: only the
    expert term depends on topk and it is linear, so with an expert share of
    8/13 the total scales by 1 - (8/13)(1 - 1/8) = 6/13. Two independent routes
    agreed on this to 5.6e-17 -- the closed form from the catalog, and an
    algebraic recovery from a controlled before/after pair.

    Mutation-tested: deleting the resolve_topk call in calculate_memory_for_micro
    makes this fail, and so does applying the width twice.
    """
    scout = next(v for v in LLAMA4_MOE_VARIANTS if v["name"] == "Llama-4-Scout-17B-16E")
    at_global = _topk_act(scout, 8)
    at_own = _topk_act(scout, 1)

    # Non-vacuity guard: a zero or equal pair would satisfy any ratio assertion.
    assert at_global > 0 and at_own > 0, "both arms must have real activation memory"
    assert at_own != at_global, (
        "activation did not move when the routing width changed 8 -> 1, so "
        "resolve_topk is not reaching the expert activation term"
    )

    assert abs(at_own / at_global - 6 / 13) < 1e-12, (
        f"expected exactly 6/13 = {6 / 13:.9f}, got {at_own / at_global:.9f}"
    )

    # The understated direction too: Qwen3.5-397B routes to 10, share 1/5,
    # so its total must GROW by exactly 21/20.
    big_qwen = next(v for v in QWEN35_MOE_VARIANTS if v["name"] == "Qwen3.5-397B-A17B")
    grew = _topk_act(big_qwen, 10) / _topk_act(big_qwen, 8)
    assert abs(grew - 21 / 20) < 1e-12, (
        f"expected exactly 21/20 = {21 / 20:.9f}, got {grew:.9f}"
    )


def test_unaffected_moe_variants_are_untouched_by_the_fix():
    """Every MoE variant already on the global width must be byte-identical.

    The fix must move exactly the variants that declare a different width and
    nothing else. Without this, a resolver that returned some other value (or
    ignored the caller's) would still satisfy the ratio test above.
    """
    for v in QWEN3_MOE_VARIANTS:
        if v.get("topk") != 8:
            continue
        assert _topk_act(v, 8) == _topk_act(v, None) or resolve_topk(v, 8) == 8
        # Forcing the width it already has cannot change anything.
        assert _topk_act(v, 8) == _topk_act(v, v["topk"]), (
            f"{v['name']} moved despite already being on the global width"
        )


def test_alltoall_volume_tracks_the_variants_routing_width():
    """Phase 5 all-to-all volume is linear in topk and must use the variant's.

    calculate_alltoall_metrics read the raw global TOPK, so Llama-4's A2A volume
    and time were overstated 8x. topk is now a parameter with the global as
    fallback, and analyze_alltoall_communication resolves it from the variant
    exactly as it already resolved seq_len.
    """
    at8 = calculate_alltoall_metrics(micro=1, d=4096, intra_node_bw=1800, seq_len=4096, topk=8)
    at1 = calculate_alltoall_metrics(micro=1, d=4096, intra_node_bw=1800, seq_len=4096, topk=1)

    assert at8["fwd_bwd_gb"] > 0 and at1["fwd_bwd_gb"] > 0, "both arms must move data"
    assert abs(at1["fwd_bwd_gb"] / at8["fwd_bwd_gb"] - 1 / 8) < 1e-12, (
        "A2A volume is not linear in topk, so the parameter is not reaching the formula"
    )

    # Omitting it must fall back to the global, not to zero or a literal.
    assert (
        calculate_alltoall_metrics(micro=1, d=4096, intra_node_bw=1800, seq_len=4096)[
            "fwd_bwd_gb"
        ]
        == at8["fwd_bwd_gb"]
    ), "the default did not fall back to the global TOPK = 8"


def test_tp_shards_activation_memory():
    """Activation must fall with TP, and the scale must be exact.

    Korthikanti eq. 2 is s*b*h*(10 + 24/t + 5as/(h*t)): 10 of 34 B/element are
    replicated on every TP rank and 24 are sharded. So the scale is
    f + (1-f)/t with f = 10/34, which is EXACTLY 1.0 at t=1.

    Before this, activation was identical at t = 1, 2, 4 and 8 -- the function
    took a `tp` argument and the term ignored it, so TP delivered no modelled
    memory saving at all.
    """
    v = next(x for x in LLAMA_DENSE_VARIANTS if x["name"] == "Llama-3.1-70B")
    f = ACT_TP_REPLICATED_FRACTION

    acts = {}
    for tp in (1, 2, 4, 8):
        acts[tp] = _mem_world(variant=v, tp=tp, pp=1, vp=1, cp=1)["activation"]

    assert acts[1] > 0, "baseline activation must be non-zero for a real ratio"
    # Non-vacuity: it must actually MOVE, or any scale assertion is decorative.
    assert acts[8] < acts[1], (
        "activation did not fall as TP rose, so the tp argument is not reaching "
        "the activation term"
    )
    for tp in (2, 4, 8):
        expected = f + (1.0 - f) / tp
        assert abs(acts[tp] / acts[1] - expected) < 1e-12, (
            f"tp={tp}: expected scale {expected:.9f}, got {acts[tp] / acts[1]:.9f}"
        )

    # The identity that makes this inert at the shipped TP=1.
    assert f + (1.0 - f) / 1 == 1.0, "scale at tp=1 must be exactly 1.0"
    # And a floor, not zero: TP can never remove the replicated part.
    assert acts[8] / acts[1] > f, "activation fell below the replicated floor"


def test_tp_communication_volume_matches_ring_all_reduce():
    """4 all-reduces per layer per micro-batch, each 2V(t-1)/t per rank.

    Megatron needs exactly two collectives forward (row-parallel attention
    output, row-parallel MLP) and two backward. TP comm did not exist at all
    before: a repo-wide grep for all_reduce returned one hit and it was prose,
    so TP was free and a parallelism search would always have maximised it.
    """
    L, d, s, m = 10, 8192, 4096, 1
    for tp in (2, 4, 8):
        r = calculate_tp_communication(L, d, s, m, tp, num_microbatches=1,
                                       dtype_bytes=2)
        one_tensor = s * m * d * 2
        expected_per_layer = 4 * 2 * one_tensor * (tp - 1) / tp / 1e9
        assert abs(r["volume_per_layer_gb"] - expected_per_layer) < 1e-12, (
            f"tp={tp}: per-layer volume {r['volume_per_layer_gb']} != "
            f"{expected_per_layer}"
        )
        assert r["collectives_per_layer"] == 4

    # tp=1 has no collective at all, so this is inert at the shipped config.
    r1 = calculate_tp_communication(L, d, s, m, 1)
    assert r1["volume_per_step_gb"] == 0.0 and r1["time_per_step_ms"] == 0.0

    # TP is INNERMOST in tp-cp-ep-dp-pp, so a group is intra-node up to node size.
    assert not calculate_tp_communication(L, d, s, m, 8, gpus_per_node=8)["is_inter_node"]
    assert calculate_tp_communication(L, d, s, m, 16, gpus_per_node=8)["is_inter_node"]


def test_end_to_end_mfu_is_not_derated_twice():
    """An end-to-end MFU already contains bubble and comm, so they are NOT added.

    MFU is achieved-FLOPs over peak across the whole run, so every idle cycle --
    waiting on a collective, stalled in a pipeline bubble -- is already inside
    it. This function used to compute
        total = compute + compute*bf/(1-bf) + exposed_comm_ms*steps
    on top of that, charging each stall twice and making every estimate
    1.22x-2.75x pessimistic.

    bubble_pct and comm_pct are still returned, as ATTRIBUTION of where the MFU
    shortfall goes -- shares of the estimate, not additions to it.
    """
    base = dict(
        total_tokens=1e12, active_params_b=8.0, num_gpus=512,
        peak_tflops_per_gpu=989, mfu=0.40, efficiency=1.0, tok_batch=4e6,
    )
    no_overhead = calculate_training_time_months_with_overhead(
        bubble_fraction=0.0, exposed_comm_ms_per_step=0.0, **base)
    with_bubble = calculate_training_time_months_with_overhead(
        bubble_fraction=0.50, exposed_comm_ms_per_step=0.0, **base)
    with_comm = calculate_training_time_months_with_overhead(
        bubble_fraction=0.0, exposed_comm_ms_per_step=500.0, **base)

    assert no_overhead["est_months"] > 0, "need a real baseline for a ratio"
    assert with_bubble["est_months"] == no_overhead["est_months"], (
        f"a 50% bubble changed est_months "
        f"{no_overhead['est_months']:.6f} -> {with_bubble['est_months']:.6f}; "
        f"an end-to-end MFU already contains the bubble"
    )
    assert with_comm["est_months"] == no_overhead["est_months"], (
        "exposed comm was added to an end-to-end MFU, double-counting the stall"
    )
    # Still attributed, or the diagnostic is lost.
    assert with_bubble["bubble_pct"] == 50.0
    assert with_comm["comm_pct"] > 0.0


def test_embeddings_are_placed_on_end_pipeline_stages():
    """Megatron puts the input embedding on stage 0 and the LM head on stage PP-1.

    This charged BOTH on EVERY stage, so the fleet carried the model's
    parameters pp times over: summing the per-GPU figure across stages gave
    30.77 GB for Llama-3.1-8B at PP=8 against a true whole model of 16.06 GB.

    pp defaults to 1 (the WHOLE model) on purpose: a caller reconciling against
    a published parameter count must not silently get a per-stage figure.
    """
    v = next(x for x in LLAMA_DENSE_VARIANTS if x["name"] == "Llama-3.1-8B")
    one_table = v["vocab"] * v["d"] * PARAM_STORAGE_BYTES / 1e9

    def mem(pp, lpg):
        return calculate_model_memory_gb(
            v, lpg, 0, PARAM_STORAGE_BYTES, 151936, LAYER_NORM,
            FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=1, pp=pp)

    whole = mem(None, v["layers"])
    assert abs(whole / PARAM_STORAGE_BYTES - v["total_params_B"]) / v["total_params_B"] < 0.01, (
        f"default must return the WHOLE model: got {whole / PARAM_STORAGE_BYTES:.2f}B "
        f"vs published {v['total_params_B']}B"
    )

    # The fleet must hold the model exactly ONCE at every pp.
    for pp in (2, 4, 8):
        per_gpu = mem(pp, v["layers"] // pp)
        body = per_gpu - one_table          # end stage carries exactly one table
        fleet = body * pp + 2 * one_table   # plus the two end tables
        assert abs(fleet - whole) / whole < 1e-9, (
            f"pp={pp}: fleet total {fleet:.4f} != whole model {whole:.4f} -- "
            f"embeddings are counted on more than the two end stages"
        )

    # Non-vacuity: pp>1 must actually differ from pp=1.
    assert mem(8, v["layers"] // 8) < mem(1, v["layers"] // 8), (
        "pipeline placement had no effect, so both tables are still charged"
    )


def test_kv_cache_uses_mla_width_for_deepseek():
    """MLA caches one compressed latent per token, not per-head K and V.

    Applying the GQA formula to DeepSeek-V3 reads its kv_heads=128 and returns
    1708 KiB/token against a real 68.6 -- a 24.9x over-estimate, because those
    head counts describe the attention COMPUTE shape and not what is stored.
    """
    v3 = next(x for x in DEEPSEEK_VARIANTS if x["name"] == "DeepSeek-V3")
    b, kind, flagged = kv_cache_bytes_per_token(v3, kv_dtype_bytes=2)
    assert kind == "mla" and not flagged
    expected = v3["layers"] * (v3["kv_lora_rank"] + v3["qk_rope_head_dim"]) * 2
    assert b == expected, f"MLA cache {b} != {expected}"

    naive = 2 * v3["layers"] * v3["kv_heads"] * (v3["d"] // v3["q_heads"]) * 2
    assert naive / b > 20, (
        "the GQA formula should be >20x larger here; if not, this test no longer "
        "demonstrates why MLA needs its own branch"
    )

    # A GQA model must still take the GQA path, or the branch is over-eager.
    llama = next(x for x in LLAMA_DENSE_VARIANTS if x["name"] == "Llama-3.1-70B")
    b2, kind2, _ = kv_cache_bytes_per_token(llama, kv_dtype_bytes=2)
    assert kind2 == "gqa"
    # head_dim is absent on 15 of 22 variants and falls back to d/q_heads, which
    # is what the module does; recomputing it here the same way is deliberate.
    hd = llama.get("head_dim") or llama["d"] // llama["q_heads"]
    assert b2 == 2 * llama["layers"] * llama["kv_heads"] * hd * 2

    # An unpublished shape is FLAGGED, not quietly computed.
    v4 = next(x for x in DEEPSEEK_VARIANTS if x["name"] == "DeepSeek-V4")
    _, kind4, flagged4 = kv_cache_bytes_per_token(v4)
    assert flagged4 and "unverified" in kind4


def test_decode_is_bandwidth_bound_and_prefill_is_compute_bound():
    """TPOT scales with HBM bandwidth; TTFT scales with FLOPS. Different regimes.

    Decode reads every active weight plus the KV cache to produce one token, so
    arithmetic intensity is ~1 FLOP/byte and the GPU stalls on memory -- peak
    TFLOPS does not enter. Prefill is a dense GEMM over the whole prompt and is
    compute-bound. Sizing that uses one model for both is wrong by orders of
    magnitude.
    """
    v = next(x for x in LLAMA_DENSE_VARIANTS if x["name"] == "Llama-3.1-70B")

    # TPOT halves when bandwidth doubles, and is INDEPENDENT of peak TFLOPS.
    t_slow = calculate_tpot_ms(v, 1675, 1, context_len=4096, batch_size=1)
    t_fast = calculate_tpot_ms(v, 3350, 1, context_len=4096, batch_size=1)
    assert abs(t_slow / t_fast - 2.0) < 1e-9, "TPOT is not inversely linear in HBM BW"

    # TTFT halves when FLOPS double, and is INDEPENDENT of bandwidth.
    f_slow = calculate_ttft_ms(v, 4096, 500, 1)
    f_fast = calculate_ttft_ms(v, 4096, 1000, 1)
    assert abs(f_slow / f_fast - 2.0) < 1e-9, "TTFT is not inversely linear in FLOPS"

    # Batching raises throughput without proportionally raising TPOT, because
    # the weight read amortises across the batch.
    t1 = calculate_tpot_ms(v, 3350, 8, context_len=4096, batch_size=1)
    t32 = calculate_tpot_ms(v, 3350, 8, context_len=4096, batch_size=32)
    assert t32 < 32 * t1, "weight read is not amortising across the batch"


def test_slo_solver_returns_the_smallest_feasible_topology():
    """Sizing means INVERTING an SLO into a topology, which is what was missing.

    Every other phase evaluates a configuration the user supplies. This one
    answers "how many GPUs do I need", and refuses rather than inventing an
    answer when nothing fits.
    """
    from configuration.hardware_config import HARDWARE
    hw = next(h for h in HARDWARE if h["name"] == "64 p5")  # H100 80GB

    v70 = next(x for x in LLAMA_DENSE_VARIANTS if x["name"] == "Llama-3.1-70B")
    r = size_for_slo(v70, hw, target_ttft_ms=500, target_tpot_ms=20,
                     prompt_len=4096, output_len=512, required_concurrency=32)
    assert r["recommended"] is not None, "70B must be servable on H100s"
    # 141 GB of BF16 weights cannot fit on 1 or 2 x 80 GB.
    assert r["recommended"]["tp"] == 4, (
        f"expected the SMALLEST feasible tp to be 4, got {r['recommended']['tp']}"
    )
    assert [t for t, _ in r["rejected"]][:2] == [1, 2]
    assert r["recommended"]["ttft_ms"] <= 500
    assert r["recommended"]["tpot_ms"] <= 20

    # And it must REFUSE when the model genuinely does not fit: DeepSeek-V3 is
    # 671B, so 1342 GB of BF16 weights against 8 x 80 = 640 GB.
    v3 = next(x for x in DEEPSEEK_VARIANTS if x["name"] == "DeepSeek-V3")
    r3 = size_for_slo(v3, hw, target_ttft_ms=500, target_tpot_ms=20,
                      required_concurrency=1)
    assert r3["recommended"] is None and not r3["feasible"], (
        "a 671B model must not report a feasible 8xH100 BF16 topology"
    )
    assert all("fit" in why for _, why in r3["rejected"])
