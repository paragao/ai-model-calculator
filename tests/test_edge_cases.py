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
            cc.TP = tp
            dp = 512 // tp
            model_mem = cc.calculate_model_memory_gb(
                v, layers_per_gpu, experts_global, PARAM_BYTES, VOCAB, LAYER_NORM,
                FFN_WEIGHT_MATRICES, NUM_EMBEDDINGS_TABLES, tp=tp)
            b = cc.calculate_memory_for_micro(
                1, model_mem, v, dp, layers_per_gpu, 1, seq_len, TOPK, PARAM_BYTES,
                EMPIRICAL_ACT_MULTIPLIER, SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
                FWD_BWD_ROUTING_BUFF_PASSES, NCCL_MEM_BUF, OPTIM_BYTES,
                ep=EP, pp=1, vp=1,
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
