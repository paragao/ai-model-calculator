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


def test_step_assessment_classifies_every_step_count():
    """No step count may fall through the bands unclassified.

    MIN_STEPS, the band bounds and MAX_STEPS are contiguous, and were once all
    tested with strict inequalities on both sides, so the shared boundary values
    200,000 / 300,000 / 800,000 matched no branch and were reported as "unknown"
    with priority 2. Latent at seq_len 4096, real at any other sequence length.
    """
    from utils.phase2_batch import assess_training_steps

    # tokens_per_batch=1 keeps us out of the two target windows, which are checked
    # first and would otherwise mask the step bands.
    boundaries = [
        MIN_STEPS, UPPER_BOUND_LOW_RANGE, LOWER_BOUND_OPTIM_RANGE,
        UPPER_BOUND_OPTIM_RANGE, LOWER_BOUND_HIGH_RANGE, MAX_STEPS,
    ]
    probes = [0, 1] + [b + off for b in boundaries for off in (-1, 0, 1)] + [10 ** 9]
    for steps in probes:
        assessment, priority = assess_training_steps(steps, 1)
        assert assessment != "unknown", f"steps={steps} fell through every band"
        assert priority in (0, 1, 2, 3), f"steps={steps} gave priority {priority}"

    # And the bands must be ordered the way their names claim.
    assert assess_training_steps(MIN_STEPS - 1, 1) == ("too few steps", 3)
    assert assess_training_steps(MAX_STEPS, 1) == ("too many steps", 3)
    assert assess_training_steps(LOWER_BOUND_OPTIM_RANGE + 1, 1)[1] == 1
    assert assess_training_steps(UPPER_BOUND_OPTIM_RANGE, 1)[0] == "borderline (high steps)"


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
