"""
Tests for the config.json loader and the zero-micro-batch pipeline bubble.

Fixtures are the published configs of
https://huggingface.co/bharatgenai/Param2-17B-A2.4B-Thinking and
https://huggingface.co/zai-org/GLM-5.3-Flash (tests/fixtures).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.architecture_loader import (
    architecture_from_config,
    attention_params,
    variant_from_architecture,
    load_variant_from_hf_config,
    _num,
)
from utils.core_calculations import calculate_model_memory_gb
from utils.phase6_pp_comm import calculate_pp_communication, microbatches_per_step

FIXTURES = Path(__file__).parent / "fixtures"
PARAM2 = FIXTURES / "param2_17b_a2_4b_config.json"
GLM = FIXTURES / "glm_5_3_flash_config.json"
GLM_TOTAL_B = 321.32  # safetensors total, as the portal passes it


def _glm():
    return load_variant_from_hf_config(GLM, "GLM-5.3-Flash", params_total_b=GLM_TOTAL_B)


def _param2():
    return load_variant_from_hf_config(PARAM2, "Param2-17B-A2.4B")


# ---- shared experts --------------------------------------------------------

def test_param2_shared_experts_are_sized_and_active():
    """2 shared experts of width 2048 each = 4096 (moe_shared_expert_intermediate_size)."""
    v = _param2()
    assert v["shared_each"] == 3 * 2048 * 4096
    # 20 MoE layers (first_k_dense_replace=1) run the shared experts for every token.
    no_shared = dict(json.load(open(PARAM2)), num_shared_experts=0,
                     moe_shared_expert_intermediate_size=0)
    v0 = variant_from_architecture(no_shared, "no-shared")
    shared_b = 20 * v["shared_each"] / 1e9
    assert abs((v["active_params_B"] - v0["active_params_B"]) - shared_b) < 0.02
    assert 2.4 <= v["active_params_B"] <= 2.6
    assert v["total_params_B"] > v0["total_params_B"]


def test_glm_shared_expert_uses_moe_width():
    """n_shared_experts=1 x moe_intermediate_size 2048, not the dense 12288."""
    assert _glm()["shared_each"] == 3 * 4096 * 2048


# ---- MLA and linear attention ----------------------------------------------

def test_glm_active_params_match_the_model_card():
    """Model card: 320B total, 18B active. Dense-GQA sizing with head_dim=0 gave 14.8B."""
    v = _glm()
    assert v["total_params_B"] == GLM_TOTAL_B
    assert 17.0 <= v["active_params_B"] <= 18.5
    assert v["attention"] == "MLA in 11 layers, linear attention in 34"
    assert v["attn_per_layer"] > 0


def test_mla_attention_formula():
    """Hand arithmetic for one MLA layer: q_lora path + compressed kv + output."""
    d, h = 4096, 64
    arch = {"kv_lora_rank": 512, "q_lora_rank": 1536, "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0, "v_head_dim": 256}
    total, kind = attention_params(_num(arch), d, 1, h, h, 0)
    q = d * 1536 + 1536 * h * 256
    kv = d * 512 + 512 * h * (256 + 256)
    o = h * 256 * d
    assert kind == "MLA"
    assert total == q + kv + o


def test_mla_with_zero_head_dim_is_not_zero():
    """A plain GQA formula on head_dim=0 returns 0 attention parameters."""
    arch = {"kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}
    total, _ = attention_params(_num(arch), 7168, 1, 128, 128, 0)
    assert total > 0


def test_linear_attention_layers_use_their_own_width():
    arch = {"linear_attn_layers": 3, "linear_attn_heads": 8, "linear_attn_head_dim": 64}
    total, kind = attention_params(_num(arch), 1024, 4, 16, 16, 64)
    full = 2 * 1024 * (16 + 16) * 64
    linear = 4 * 1024 * 8 * 64
    assert total == full + 3 * linear
    assert "linear attention in 3" in kind


def test_text_config_and_layer_types_are_flattened():
    arch = architecture_from_config(json.load(open(GLM)))
    assert arch["num_hidden_layers"] == 45  # from text_config
    assert arch["linear_attn_layers"] == 34
    assert arch["linear_attn_heads"] == 64 and arch["linear_attn_head_dim"] == 128


# ---- memory uses the variant and the TP actually chosen ---------------------

def test_loaded_variant_memory_shards_with_tp():
    v = _glm()
    layers = v["layers"]
    m1 = calculate_model_memory_gb(v, layers, v["num_experts"], 2, v["vocab"], 2, 3, 2, tp=1, ep=1)
    m4 = calculate_model_memory_gb(v, layers, v["num_experts"], 2, v["vocab"], 2, 3, 2, tp=4, ep=1)
    assert 3.9 < m1 / m4 <= 4.0  # layer norms and routers stay replicated
    # ep=1 holds every expert: the whole model. The shape misses the MTP layer and the
    # sparse-attention indexer, so it lands within 3% of the exact 321.32B x 2 bytes.
    assert abs(m1 - 2 * GLM_TOTAL_B) / (2 * GLM_TOTAL_B) < 0.03


# ---- pipeline bubble --------------------------------------------------------

def _pp(gbs, mbs, dp, pp=8, vp=4):
    return calculate_pp_communication(
        layers=45, hidden_dim=4096, seq_len=4096, mbs=mbs, gbs=gbs, pp=pp, vp=vp,
        dp=dp, ep=8, inter_node_bw_gb=400, intra_node_bw_gbps=900, gpus_per_node=8)


def test_batch_smaller_than_one_microbatch_per_rank_is_not_a_full_bubble():
    """2048 GPUs, PP=8 -> DP=256; micro batch 4 and 976 sequences gave 0 micro-batches
    and a 100% bubble with zero sends."""
    r = _pp(gbs=976, mbs=4, dp=256)
    assert r["num_microbatches"] >= 1
    assert r["pipeline_bubble_pct"] < 100
    assert r["total_sends_per_step"] > 0


def test_microbatches_per_step():
    assert microbatches_per_step(976, 4, 256) == 1
    assert microbatches_per_step(4096, 4, 64) == 16
    assert microbatches_per_step(10, 0, 4) == 1


def test_bubble_shrinks_as_microbatches_grow():
    few = _pp(gbs=256, mbs=1, dp=32)["pipeline_bubble_pct"]
    many = _pp(gbs=4096, mbs=1, dp=32)["pipeline_bubble_pct"]
    assert many < few < 100


# ---- command line -----------------------------------------------------------

def test_cli_hf_config_runs_end_to_end(capsys):
    import ai_model_calculator
    from configuration.hardware_config import HARDWARE
    ai_model_calculator.main(["--hf-config", str(PARAM2), "--model", "Param2-17B",
                              "--hw-name", HARDWARE[0]["name"], "--report"])
    out = capsys.readouterr().out
    assert "Param2-17B" in out and "Phase 1" in out
