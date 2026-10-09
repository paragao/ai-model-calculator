"""
Model variant configurations.

Pre-defined model architectures for the AI Model Calculator.
Includes dense and MoE models from Llama, Qwen, and DeepSeek families.

For dense models: expert_ffn=0, dense_layers=layers (all layers are dense).
For MoE models: expert_ffn>0, dense_layers=number of non-MoE layers.

Derived fields are computed using:
  head_dim = explicit (typically 128 or 256), or d / q_heads if not specified
  attn_per_layer = 2 * d * (q_heads + kv_heads) * head_dim
  expert_each = d * expert_ffn * 3
  shared_each = d * shared_expert_ffn * 3 (for shared experts) or 0
  router_each = d * num_experts (or 0 for dense)
  moe_layer_params = (num_experts * expert_each) + shared_each + router_each
  dense_layer_params = attn_per_layer + (d * dense_ffn * 3)
"""

# ============================================================================
# Llama 3.1 / 3.2 / 3.3 Dense Models
# ============================================================================

LLAMA_DENSE_VARIANTS = [
    {
        "name": "Llama-3.1-8B",
        "vocab": 128_256,
        "layers": 32,
        "d": 4096,
        "expert_ffn": 0,
        "dense_ffn": 14336,
        "q_heads": 32,
        "kv_heads": 8,
        "dense_layers": 32,
        "total_params_B": 8.03,
        "active_params_B": 8.03,
        "attn_per_layer": 41_943_040,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 213_909_504,
        "valid_pp": [1, 2, 4, 8, 16, 32],
        "seq_len": 4096,
    },
    {
        "name": "Llama-3.1-70B",
        "vocab": 128_256,
        "layers": 80,
        "d": 8192,
        "expert_ffn": 0,
        "dense_ffn": 28672,
        "q_heads": 64,
        "kv_heads": 8,
        "dense_layers": 80,
        "total_params_B": 70.6,
        "active_params_B": 70.6,
        "attn_per_layer": 150_994_944,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 855_638_016,
        "valid_pp": [1, 2, 4, 5, 8, 10, 16, 20, 40, 80],
        "seq_len": 4096,
    },
    {
        "name": "Llama-3.1-405B",
        "vocab": 128_256,
        "layers": 126,
        "d": 16384,
        "expert_ffn": 0,
        "dense_ffn": 53248,
        "q_heads": 128,
        "kv_heads": 8,
        "dense_layers": 126,
        "total_params_B": 405.0,
        "active_params_B": 405.0,
        "attn_per_layer": 570_425_344,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 3_220_439_040,
        "valid_pp": [1, 2, 3, 6, 7, 9, 14, 18, 21, 42, 63, 126],
        "seq_len": 4096,
    },
    {
        "name": "Llama-3.2-3B",
        "vocab": 128_256,
        "tied_embeddings": True,
        "layers": 28,
        "d": 3072,
        "expert_ffn": 0,
        "dense_ffn": 8192,
        "q_heads": 24,
        "kv_heads": 8,
        "dense_layers": 28,
        "total_params_B": 3.21,
        "active_params_B": 3.21,
        "attn_per_layer": 25_165_824,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 100_663_296,
        "valid_pp": [1, 2, 4, 7, 14, 28],
        "seq_len": 4096,
    },
    {
        "name": "Llama-3.3-70B",
        "vocab": 128_256,
        "layers": 80,
        "d": 8192,
        "expert_ffn": 0,
        "dense_ffn": 28672,
        "q_heads": 64,
        "kv_heads": 8,
        "dense_layers": 80,
        "total_params_B": 70.6,
        "active_params_B": 70.6,
        "attn_per_layer": 150_994_944,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 855_638_016,
        "valid_pp": [1, 2, 4, 5, 8, 10, 16, 20, 40, 80],
        "seq_len": 4096,
    },
]

# ============================================================================
# Llama 4 MoE Models
# ============================================================================

LLAMA4_MOE_VARIANTS = [
    # Both Llama-4 variants are multimodal (Llama4ForConditionalGeneration with a
    # vision_config: 34 layers, hidden 1408, intermediate 5632, plus a 4096-dim
    # projector). That tower is ~0.87 B parameters and this text-only engine does
    # NOT model it, so published totals sit slightly ABOVE what is computed here:
    #   Scout    107.77 B computed + 0.87 B vision = 108.64 B vs 109.0 published
    #   Maverick 400.71 B computed + 0.87 B vision = 401.58 B vs 400.0 published
    # Both are within rounding of the published figures, so the text accounting
    # below is considered reconciled. Do not "close" the residual by inflating
    # the text parameters.
    #
    # Also not modelled: non-uniform attention (no_rope_layers alternates
    # 1,1,1,0 and attention_chunk_size is 8192 with cache_implementation hybrid).
    {
        "name": "Llama-4-Scout-17B-16E",
        "vocab": 202_048,
        "num_experts": 16,
        "topk": 1,
        "layers": 48,
        "d": 5120,
        "expert_ffn": 8192,
        "dense_ffn": 16384,
        "q_heads": 40,
        "kv_heads": 8,
        "dense_layers": 0,
        "total_params_B": 109.0,
        "active_params_B": 17.0,
        "attn_per_layer": 62_914_560,
        "expert_each": 125_829_120,
        # Every Llama4TextMoe layer carries a shared_expert alongside the routed
        # ones, sized at intermediate_size (8192), so 3 * 5120 * 8192. Was 0.
        # NOTE: intermediate_size_mlp (16384) is VESTIGIAL for Scout --
        # interleave_moe_layer_step=1 and moe_layers lists all 48 layers, so no
        # dense MLP layer exists. Do not attribute the shortfall to it.
        # https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json
        "shared_each": 125_829_120,
        "router_each": 81_920,
        # 16 * 125_829_120 + 125_829_120 (shared) + 81_920 (router)
        "moe_layer_params": 2_139_176_960,
        "dense_layer_params": 314_572_800,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
        "seq_len": 4096,
    },
    {
        "name": "Llama-4-Maverick-17B-128E",
        "vocab": 202_048,
        "num_experts": 128,
        "topk": 1,
        "layers": 48,
        "d": 5120,
        "expert_ffn": 8192,
        "dense_ffn": 16384,
        "q_heads": 40,
        "kv_heads": 8,
        "dense_layers": 24,
        "total_params_B": 400.0,
        "active_params_B": 17.0,
        "attn_per_layer": 62_914_560,
        "expert_each": 125_829_120,
        # Shared expert per MoE layer, sized at intermediate_size (8192). Was 0.
        # Maverick has interleave_moe_layer_step=2, so 24 MoE layers carry this
        # and the other 24 are dense layers using intermediate_size_mlp (16384).
        "shared_each": 125_829_120,
        "router_each": 655_360,
        # 128 * 125_829_120 + 125_829_120 (shared) + 655_360 (router)
        "moe_layer_params": 16_232_611_840,
        "dense_layer_params": 314_572_800,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
        "seq_len": 4096,
    },
]

# ============================================================================
# Qwen2.5 Dense Models
# ============================================================================

QWEN25_DENSE_VARIANTS = [
    {
        "name": "Qwen2.5-7B",
        "vocab": 152_064,
        "layers": 28,
        "d": 3584,
        "expert_ffn": 0,
        "dense_ffn": 18944,
        "q_heads": 28,
        "kv_heads": 4,
        "dense_layers": 28,
        "total_params_B": 7.62,
        "active_params_B": 7.62,
        "attn_per_layer": 29_360_128,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 233_046_016,
        "valid_pp": [1, 2, 4, 7, 14, 28],
        "seq_len": 4096,
    },
    {
        "name": "Qwen2.5-32B",
        "vocab": 152_064,
        "layers": 64,
        "d": 5120,
        "expert_ffn": 0,
        "dense_ffn": 27648,
        "q_heads": 40,
        "kv_heads": 8,
        "dense_layers": 64,
        "total_params_B": 32.76,
        "active_params_B": 32.76,
        "attn_per_layer": 62_914_560,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 487_587_840,
        "valid_pp": [1, 2, 4, 8, 16, 32, 64],
        "seq_len": 4096,
    },
    {
        "name": "Qwen2.5-72B",
        "vocab": 152_064,
        "layers": 80,
        "d": 8192,
        "expert_ffn": 0,
        "dense_ffn": 29568,
        "q_heads": 64,
        "kv_heads": 8,
        "dense_layers": 80,
        "total_params_B": 72.7,
        "active_params_B": 72.7,
        "attn_per_layer": 150_994_944,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 877_658_112,
        "valid_pp": [1, 2, 4, 5, 8, 10, 16, 20, 40, 80],
        "seq_len": 4096,
    },
]

# ============================================================================
# Qwen3 Dense Models
# ============================================================================

QWEN3_DENSE_VARIANTS = [
    {
        "name": "Qwen3-8B",
        "vocab": 151_936,
        "layers": 36,
        "d": 4096,
        "expert_ffn": 0,
        "dense_ffn": 12288,  # config.json intermediate_size=12288 (was 11008)
                             # https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
        "q_heads": 32,
        "kv_heads": 8,
        "dense_layers": 36,
        "total_params_B": 8.04,
        "active_params_B": 8.04,
        "attn_per_layer": 41_943_040,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 192_937_984,  # 41_943_040 + 3 * 4096 * 12288
        "valid_pp": [1, 2, 3, 4, 6, 9, 12, 18, 36],
        "seq_len": 4096,
    },
    {
        "name": "Qwen3-32B",
        "vocab": 151_936,
        "layers": 64,
        "d": 5120,
        "expert_ffn": 0,
        "dense_ffn": 25600,
        "q_heads": 64,
        "kv_heads": 8,
        # Qwen3-32B sets head_dim explicitly to 128 in its config.json; it is
        # NOT d // q_heads (which would give 80). Verified against
        # https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json
        "head_dim": 128,
        "dense_layers": 64,
        "total_params_B": 30.5,
        "active_params_B": 30.5,
        "attn_per_layer": 94_371_840,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 452_165_632,
        "valid_pp": [1, 2, 4, 8, 16, 32, 64],
        "seq_len": 4096,
    },
]

# ============================================================================
# Qwen3 MoE Models
# ============================================================================

QWEN3_MOE_VARIANTS = [
    {
        "name": "Qwen3-30B-A3B",
        "vocab": 151_936,
        "num_experts": 128,
        "topk": 8,
        "layers": 48,
        "d": 2048,
        "expert_ffn": 768,
        "dense_ffn": 6144,
        "q_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "dense_layers": 0,
        "total_params_B": 30.53,
        "active_params_B": 3.35,
        "attn_per_layer": 18_874_368,
        "expert_each": 4_718_592,
        "shared_each": 0,
        "router_each": 262_144,
        "moe_layer_params": 604_241_920,
        "dense_layer_params": 56_623_104,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
        "seq_len": 4096,
    },
    {
        "name": "Qwen3-235B-A22B",
        "vocab": 151_936,
        "num_experts": 128,
        "topk": 8,
        "layers": 94,
        "d": 4096,
        "expert_ffn": 1536,
        "dense_ffn": 12288,
        "q_heads": 64,
        "kv_heads": 4,
        "head_dim": 128,
        "dense_layers": 0,
        "total_params_B": 235.09,
        "active_params_B": 22.19,
        "attn_per_layer": 71_303_168,
        "expert_each": 18_874_368,
        "shared_each": 0,
        "router_each": 524_288,
        "moe_layer_params": 2_416_443_392,
        "dense_layer_params": 222_298_112,
        "valid_pp": [1, 2, 47, 94],
        "seq_len": 4096,
    },
]

# ============================================================================
# Qwen3.5 Dense Models
#
# LIMITATION now fixed -- these entries once modelled Qwen3.5 as uniform full
# attention, which it is not:
#   - FIXED: the Phase 3 O(seq_len^2) term counts only full-attention layers,
#     via "full_attention_interval": 4 below. It previously applied to every
#     layer, overstating long-context time by 54% at 32K and 141% at 128K.
#
# RESOLVED: Qwen3.5 is a HYBRID attention family and is now modelled as one.
# full_attention_interval = 4 gives a 3:1 layer_types pattern -- three Gated
# DeltaNet "linear_attention" layers per one gated "full_attention" layer -- so
# of Qwen3.5-27B's 64 layers, 16 are full attention and 48 are linear.
#
# Both halves are now counted from the published configs:
#   - "attn_per_layer" is the FULL-attention layer.
#   - "linear_attn_per_layer" is the Gated DeltaNet layer, applied to the
#     (layers - layers//interval) linear layers by calculate_model_memory_gb.
#
# A GDN layer is NOT smaller than a full-attention layer. Measured:
#     variant              full_attn      GDN     GDN/full
#     Qwen3.5-27B        104,857,600  115,876,064   1.105x
#     Qwen3.5-35B-A3B     27,262,976   33,718,464   1.237x
#     Qwen3.5-122B-A10B   78,643,200   88,523,008   1.126x
#     Qwen3.5-397B-A17B  104,857,600  118,014,208   1.125x
# An earlier note in this file estimated 86% and called the old uniform treatment
# an OVERSTATEMENT. Both were wrong: it was an understatement.
#
# GDN weights, per transformers' Qwen3NextGatedDeltaNet, with
# key_dim = linear_num_key_heads * linear_key_head_dim and
# value_dim = linear_num_value_heads * linear_value_head_dim:
#     in_proj_qkvz  d * (key_dim*2 + value_dim*2)   q, k, v and the z gate
#     in_proj_ba    d * (linear_num_value_heads*2)  beta and alpha gates
#     conv1d        (key_dim*2 + value_dim) * linear_conv_kernel_dim  depthwise
#     out_proj      value_dim * d
#     plus RMSNormGated, A_log and dt_bias, which are negligible
#
# The FULL-attention layer was ALSO wrong before: "attn_output_gate": true
# DOUBLES q_proj (num_attention_heads * head_dim * 2), which the old values
# omitted -- they were 1.43x to 1.47x too small. Fixing both moved
# total_params_B from 24.35 to 26.90 for the 27B, and all four now reconcile to
# within 1% of their published sizes (-0.39%, -0.97%, +0.09%, -0.16%).
#
# Qwen3.5 is also multimodal (Qwen3_5ForConditionalGeneration with a
# vision_config: depth 27, hidden 1152), which this text-only engine does not
# model -- that is the residual under 1%. Do NOT close it by inflating text
# parameters. vocab_size 248320 and head_dim 256 verified against
# https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json
# ============================================================================

QWEN35_DENSE_VARIANTS = [
    {
        "name": "Qwen3.5-27B",
        "vocab": 248_320,
        "full_attention_interval": 4,
        "layers": 64,
        "d": 5120,
        "expert_ffn": 0,
        "dense_ffn": 17408,
        "q_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "dense_layers": 64,
        "total_params_B": 26.9,
        "active_params_B": 26.9,
        "attn_per_layer": 104_857_600,
        "linear_attn_per_layer": 115_876_064,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 340_787_200,
        "valid_pp": [1, 2, 4, 8, 16, 32, 64],
        "seq_len": 4096,
    },
]

# ============================================================================
# Qwen3.5 MoE Models
# ============================================================================

QWEN35_MOE_VARIANTS = [
    {
        "name": "Qwen3.5-35B-A3B",
        "vocab": 248_320,
        "num_experts": 256,
        "topk": 8,
        "full_attention_interval": 4,
        "layers": 40,
        "d": 2048,
        "expert_ffn": 512,
        "dense_ffn": 0,
        "q_heads": 16,
        "kv_heads": 2,
        "head_dim": 256,
        "dense_layers": 0,
        "total_params_B": 34.66,
        "active_params_B": 3.45,
        "attn_per_layer": 27_262_976,
        "linear_attn_per_layer": 33_718_464,
        "expert_each": 3_145_728,
        "shared_each": 3_145_728,
        "router_each": 524_288,
        "moe_layer_params": 808_976_384,
        "dense_layer_params": 18_874_368,
        "valid_pp": [1, 2, 4, 5, 8, 10, 20, 40],
        "seq_len": 4096,
    },
    {
        "name": "Qwen3.5-122B-A10B",
        "vocab": 248_320,
        "num_experts": 256,
        "topk": 8,
        "full_attention_interval": 4,
        "layers": 48,
        "d": 3072,
        "expert_ffn": 1024,
        "dense_ffn": 0,
        "q_heads": 32,
        "kv_heads": 2,
        "head_dim": 256,
        "dense_layers": 0,
        "total_params_B": 122.11,
        "active_params_B": 9.77,
        "attn_per_layer": 78_643_200,
        "linear_attn_per_layer": 88_523_008,
        "expert_each": 9_437_184,
        "shared_each": 9_437_184,
        "router_each": 786_432,
        "moe_layer_params": 2_426_142_720,
        "dense_layer_params": 53_477_376,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
        "seq_len": 4096,
    },
    {
        "name": "Qwen3.5-397B-A17B",
        "vocab": 248_320,
        "num_experts": 512,
        "topk": 10,
        "full_attention_interval": 4,
        "layers": 60,
        "d": 4096,
        "expert_ffn": 1024,
        "dense_ffn": 0,
        "q_heads": 32,
        "kv_heads": 2,
        "head_dim": 256,
        "dense_layers": 0,
        "total_params_B": 396.35,
        "active_params_B": 17.35,
        "attn_per_layer": 104_857_600,
        "linear_attn_per_layer": 118_014_208,
        "expert_each": 12_582_912,
        "shared_each": 12_582_912,
        "router_each": 2_097_152,
        "moe_layer_params": 6_457_131_008,
        "dense_layer_params": 71_303_168,
        "valid_pp": [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60],
        "seq_len": 4096,
    },
]

# ============================================================================
# DeepSeek MoE Models
#
# NOTE: attn_per_layer for these variants is NOT reproducible from the generic
# 2 * d * (q_heads + kv_heads) * head_dim formula in
# utils/calculate_variant_fields.py. DeepSeek uses Multi-head Latent Attention
# (MLA), which compresses KV into a low-rank latent projection, so the standard
# Q/K/V/O parameter count does not apply. Solving the generic formula against
# the stored value yields a non-integer head_dim (~51), which is the signature
# of that mismatch rather than a typo. Do not "correct" these to the formula
# output -- doing so would replace MLA accounting with dense-attention
# accounting and overstate DeepSeek V3/V3.1/V3.2 attention memory by ~10%.
# ============================================================================

DEEPSEEK_VARIANTS = [
    {
        "name": "DeepSeek-V3",
        "vocab": 129_280,
        "num_experts": 256,
        "topk": 8,
        "layers": 61,
        "d": 7168,
        "expert_ffn": 2048,
        "dense_ffn": 18432,
        "q_heads": 128,
        # MLA: qk_nope_head_dim 128 + qk_rope_head_dim 64; v_head_dim 128.
        # Verified against the published config. Used ONLY by
        # compute_attention_flops_dim, which needs the score/context matmul
        # widths and cannot derive them from d / q_heads for MLA.
        "qk_head_dim": 192,
        "v_head_dim": 128,
        # MLA KV-CACHE DIMENSIONS, for inference sizing. DeepSeek uses
        # Multi-head Latent Attention: the cache stores a COMPRESSED latent
        # per token, not per-head K and V. config.json gives
        # kv_lora_rank=512 and qk_rope_head_dim=64, so the cache is
        # (512 + 64) = 576 elements per layer per token.
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
        #
        # Without these, a naive 2*layers*kv_heads*head_dim cache formula
        # reads kv_heads=128 and overstates this model 1708 KiB/token against
        # a real ~70 KiB/token -- a >20x error, because those head counts
        # describe the attention COMPUTE shape, not what is cached.
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "kv_heads": 128,
        "dense_layers": 3,
        "total_params_B": 671.0,
        "active_params_B": 36.7,
        "attn_per_layer": 187_105_280,
        "expert_each": 44_040_192,
        # Shared expert is sized at moe_intermediate_size (2048), NOT intermediate_size
        # (18432, which applies only to the 3 dense layers): 3 * 7168 * 2048.
        # config.json: n_shared_experts=1, moe_intermediate_size=2048
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
        "shared_each": 44_040_192,
        "router_each": 1_835_008,
        # 256 * 44_040_192 + 44_040_192 (shared) + 1_835_008 (router)
        "moe_layer_params": 11_320_164_352,
        "dense_layer_params": 583_467_008,
        "valid_pp": [1, 61],
        "seq_len": 4096,
    },
    {
        "name": "DeepSeek-V3.1",
        "vocab": 129_280,
        "num_experts": 256,
        "topk": 8,
        "layers": 61,
        "d": 7168,
        "expert_ffn": 2048,
        "dense_ffn": 18432,
        "q_heads": 128,
        # MLA: qk_nope_head_dim 128 + qk_rope_head_dim 64; v_head_dim 128.
        # Verified against the published config. Used ONLY by
        # compute_attention_flops_dim, which needs the score/context matmul
        # widths and cannot derive them from d / q_heads for MLA.
        "qk_head_dim": 192,
        "v_head_dim": 128,
        # MLA KV-CACHE DIMENSIONS, for inference sizing. DeepSeek uses
        # Multi-head Latent Attention: the cache stores a COMPRESSED latent
        # per token, not per-head K and V. config.json gives
        # kv_lora_rank=512 and qk_rope_head_dim=64, so the cache is
        # (512 + 64) = 576 elements per layer per token.
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
        #
        # Without these, a naive 2*layers*kv_heads*head_dim cache formula
        # reads kv_heads=128 and overstates this model 1708 KiB/token against
        # a real ~70 KiB/token -- a >20x error, because those head counts
        # describe the attention COMPUTE shape, not what is cached.
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "kv_heads": 128,
        "dense_layers": 3,
        "total_params_B": 671.0,
        "active_params_B": 36.7,
        "attn_per_layer": 187_105_280,
        "expert_each": 44_040_192,
        # Shared expert is sized at moe_intermediate_size (2048), NOT intermediate_size
        # (18432, which applies only to the 3 dense layers): 3 * 7168 * 2048.
        # config.json: n_shared_experts=1, moe_intermediate_size=2048
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
        "shared_each": 44_040_192,
        "router_each": 1_835_008,
        # 256 * 44_040_192 + 44_040_192 (shared) + 1_835_008 (router)
        "moe_layer_params": 11_320_164_352,
        "dense_layer_params": 583_467_008,
        "valid_pp": [1, 61],
        "seq_len": 4096,
    },
    {
        "name": "DeepSeek-V3.2",
        "vocab": 129_280,
        "num_experts": 256,
        "topk": 8,
        "layers": 61,
        "d": 7168,
        "expert_ffn": 2048,
        "dense_ffn": 18432,
        "q_heads": 128,
        # MLA: qk_nope_head_dim 128 + qk_rope_head_dim 64; v_head_dim 128.
        # Verified against the published config. Used ONLY by
        # compute_attention_flops_dim, which needs the score/context matmul
        # widths and cannot derive them from d / q_heads for MLA.
        "qk_head_dim": 192,
        "v_head_dim": 128,
        # MLA KV-CACHE DIMENSIONS, for inference sizing. DeepSeek uses
        # Multi-head Latent Attention: the cache stores a COMPRESSED latent
        # per token, not per-head K and V. config.json gives
        # kv_lora_rank=512 and qk_rope_head_dim=64, so the cache is
        # (512 + 64) = 576 elements per layer per token.
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
        #
        # Without these, a naive 2*layers*kv_heads*head_dim cache formula
        # reads kv_heads=128 and overstates this model 1708 KiB/token against
        # a real ~70 KiB/token -- a >20x error, because those head counts
        # describe the attention COMPUTE shape, not what is cached.
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "kv_heads": 128,
        "dense_layers": 3,
        "total_params_B": 671.0,
        "active_params_B": 36.7,
        "attn_per_layer": 187_105_280,
        "expert_each": 44_040_192,
        # Shared expert is sized at moe_intermediate_size (2048), NOT intermediate_size
        # (18432, which applies only to the 3 dense layers): 3 * 7168 * 2048.
        # config.json: n_shared_experts=1, moe_intermediate_size=2048
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
        "shared_each": 44_040_192,
        "router_each": 1_835_008,
        # 256 * 44_040_192 + 44_040_192 (shared) + 1_835_008 (router)
        "moe_layer_params": 11_320_164_352,
        "dense_layer_params": 583_467_008,
        "valid_pp": [1, 61],
        "seq_len": 4096,
    },
    {
        "name": "DeepSeek-V4",
        # KV-CACHE SHAPE UNVERIFIED. This entry carries no kv_lora_rank and
        # no head_dim, and its q_heads=128 / kv_heads=1 is MQA-shaped where
        # every other DeepSeek uses MLA. So neither the MLA formula nor the
        # GQA one can be trusted here: the GQA fallback reads 13.3 KiB/token
        # against its siblings' real 68.6, an UNDER-estimate that would
        # overstate how many concurrent sequences fit. Flagged explicitly
        # rather than detected by heuristic, because the honest statement is
        # that the shape is unpublished, not that it resembles something.
        "kv_cache_unverified": True,
        # KNOWN CATALOG GAP, flagged not fixed: this entry carries no
        # "num_experts", no "topk" and no "vocab", so all three fall back to the
        # globals. Two are recoverable from the entry's own arithmetic and the
        # third is wrong:
        #
        #   num_experts = 384, EXACTLY and integrally, from this entry's own
        #     moe_layer_params: (25,435,963,392 - 66,060,288 - 2,752,512)
        #     / 66,060,288 = 384.0. The global fallback is 128, understating the
        #     expert count 3x. The Quick skill's SKILL.md independently
        #     documents 384, which agrees.
        #
        #   topk = 8 is what active_params_B = 49.0 implies, so the global
        #     fallback is already RIGHT here and nothing moves. Candidates:
        #     topk=6 gives 41.64 B (-15.0%), topk=8 gives 49.70 B (+1.4%), and 8
        #     is the closest of all twelve. (SKILL.md documented 6; corrected.)
        #
        #   vocab is MISSING and the fallback is wrong: it resolves to the Qwen3
        #     default 151,936 where DeepSeek uses 129,280, overstating the two
        #     embedding tables 17.5% (2.178 B vs 1.853 B params).
        #
        # Not fixed here because adding num_experts and vocab changes this
        # variant's memory in every phase and needs its own baselines, which is
        # separate work from the routing-width fix that found it.
        "layers": 61,
        "d": 7168,
        "expert_ffn": 3072,
        "dense_ffn": 3072,
        "q_heads": 128,
        "kv_heads": 1,
        "dense_layers": 0,
        "total_params_B": 1600.0,
        "active_params_B": 49.0,
        "attn_per_layer": 187_105_280,
        "expert_each": 66_060_288,
        "shared_each": 66_060_288,
        "router_each": 2_752_512,
        "moe_layer_params": 25_435_963_392,
        "dense_layer_params": 253_165_568,
        "valid_pp": [1, 61],
        "seq_len": 4096,
    },
]

# ============================================================================
# Combined list -- the calculator iterates over VARIANTS
# Include ONLY the variants the user selected for this analysis.
# ============================================================================

VARIANTS = (
    LLAMA_DENSE_VARIANTS
    + LLAMA4_MOE_VARIANTS
    + QWEN25_DENSE_VARIANTS
    + QWEN3_DENSE_VARIANTS
    + QWEN3_MOE_VARIANTS
    + QWEN35_DENSE_VARIANTS
    + QWEN35_MOE_VARIANTS
    + DEEPSEEK_VARIANTS
)
