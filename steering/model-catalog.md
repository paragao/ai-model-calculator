# Model Catalog

## Pre-defined models

The following models are pre-configured. The user can select one or more for analysis. Present them grouped by family.

## Dense Models

| Name | Layers | Hidden Dim | FFN Dim | Total Params | Vocab |
|------|--------|-----------|---------|-------------|-------|
| Llama-3.1-8B | 32 | 4096 | 14336 | 8.03B | 128,256 |
| Llama-3.1-70B | 80 | 8192 | 28672 | 70.6B | 128,256 |
| Llama-3.1-405B | 126 | 16384 | 53248 | 405B | 128,256 |
| Llama-3.2-3B | 28 | 3072 | 8192 | 3.21B | 128,256 |
| Llama-3.3-70B | 80 | 8192 | 28672 | 70.6B | 128,256 |
| Qwen2.5-7B | 28 | 3584 | 18944 | 7.6B | 152,064 |
| Qwen2.5-32B | 64 | 5120 | 27648 | 32.8B | 152,064 |
| Qwen2.5-72B | 80 | 8192 | 29568 | 72.7B | 152,064 |
| Qwen3-8B | 36 | 4096 | 11008 | 8.04B | 151,936 |
| Qwen3-32B | 64 | 5120 | 25600 | 30.5B | 151,936 |
| Qwen3.5-27B | 64 | 5120 | 17408 | 24.4B | 248,320 |

## MoE Models

| Name | Layers | Hidden Dim | Expert FFN | Experts | Top-K | Total Params | Active Params |
|------|--------|-----------|-----------|---------|-------|-------------|--------------|
| Qwen3-30B-A3B | 48 | 2048 | 768 | 128 | 8 | 30.5B | 3.3B |
| Qwen3-235B-A22B | 94 | 4096 | 1536 | 128 | 8 | 235.1B | 22.2B |
| Qwen3.5-35B-A3B | 40 | 2048 | 512 | 256 | 8 | 34.1B | 2.9B |
| Qwen3.5-122B-A10B | 48 | 3072 | 1024 | 256 | 8 | 120.6B | 8.2B |
| Qwen3.5-397B-A17B | 60 | 4096 | 1024 | 512 | 10 | 393.7B | 14.7B |
| Llama-4-Scout-17B-16E | 48 | 5120 | 8192 | 16 | 1 | 101.7B | 11.1B |
| Llama-4-Maverick-17B-128E | 48 | 5120 | 8192 | 128 | 1 | 397.7B | 14.2B |
| DeepSeek-V3 | 61 | 7168 | 2048 | 256 | 8 | 671B | 36.7B |
| DeepSeek-V3.1 | 61 | 7168 | 2048 | 256 | 8 | 671B | 36.7B |
| DeepSeek-V3.2 | 61 | 7168 | 2048 | 256 | 8 | 671B | 36.7B |
| DeepSeek-V4 | 61 | 7168 | 3072 | 384 | 6 | 1600B | 49.0B |

## Required inputs for custom models

If the user selects "Custom model", gather these parameters:

| Parameter | Description | Example (Llama-70B) | Required |
|-----------|-------------|---------------------|----------|
| `name` | Model name | `"Llama-70B"` | Yes |
| `layers` | Total transformer layers | `80` | Yes |
| `d` | Hidden dimension | `8192` | Yes |
| `q_heads` | Query attention heads | `64` | Yes |
| `kv_heads` | KV attention heads (GQA) | `8` | Yes |
| `head_dim` | Head dimension | `128` | Yes |
| `dense_ffn` | Dense FFN intermediate dim | `28672` | Yes |
| `total_params_B` | Total parameters (billions) | `70.6` | Yes |
| `active_params_B` | Active params per forward pass | `70.6` (same for dense) | Yes |
| `expert_ffn` | Expert FFN intermediate dim | `0` (dense) or e.g. `2048` | Yes |
| `dense_layers` | Number of non-MoE layers | `80` (all for dense) or `0-3` | Yes |

For MoE models, also ask:
- **N_EXPERTS**: Total number of experts (e.g., 128, 256)
- **TOPK**: Experts activated per token (e.g., 2, 8)
- **Has shared experts?**: If yes, `shared_each = d * shared_expert_ffn * 3`; if no, `shared_each = 0`

## Auto-derived fields

The following are computed automatically from the inputs above -- do NOT ask the user for these:

- `head_dim` = explicit from config (typically 128 or 256), or `d / q_heads` if not specified
- `attn_per_layer = 2 * d * (q_heads + kv_heads) * head_dim`
  - Note: the shorthand `d^2 * 2.25` only works when `head_dim = d / q_heads` (e.g., Llama 3.1)
- `expert_each = d * expert_ffn * 3`
- `shared_each = d * shared_expert_ffn * 3` (if shared experts) or `0`
- `router_each = d * N_EXPERTS` (or `0` for dense)
- `moe_layer_params = (N_EXPERTS * expert_each) + shared_each + router_each`
- `dense_layer_params = attn_per_layer + (d * dense_ffn * 3)`
- `valid_pp` = all divisors of `layers`

## The variants_config.py template

Save this file to `/tmp/ai-model-calculator/configuration/variants_config.py` before running the calculator. Include only the variants the user selected -- remove the others from the `VARIANTS` list at the bottom.

```python
"""
Model variant configurations.

Pre-defined model architectures for the AI Model Calculator.
Includes dense and MoE models from Llama, Qwen, and DeepSeek families.

For dense models: expert_ffn=0, dense_layers=layers (all layers are dense).
For MoE models: expert_ffn>0, dense_layers=number of non-MoE layers.

Derived fields (attn_per_layer, expert_each, shared_each, router_each,
moe_layer_params, dense_layer_params) are computed using:
  head_dim = explicit from config (typically 128 or 256)
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
        "layers": 32,
        "d": 4096,
        "expert_ffn": 0,
        "dense_ffn": 14336,
        "q_heads": 32,
        "kv_heads": 8,
        "dense_layers": 32,
        "total_params_B": 8.03,
        "active_params_B": 8.03,
        "attn_per_layer": 37_748_736,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 213_909_504,
        "valid_pp": [1, 2, 4, 8, 16, 32],
    },
    {
        "name": "Llama-3.1-70B",
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
    },
    {
        "name": "Llama-3.1-405B",
        "layers": 126,
        "d": 16384,
        "expert_ffn": 0,
        "dense_ffn": 53248,
        "q_heads": 128,
        "kv_heads": 8,
        "dense_layers": 126,
        "total_params_B": 405.0,
        "active_params_B": 405.0,
        "attn_per_layer": 603_979_776,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 3_220_439_040,
        "valid_pp": [1, 2, 3, 6, 7, 9, 14, 18, 21, 42, 63, 126],
    },
    {
        "name": "Llama-3.2-3B",
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
    },
    {
        "name": "Llama-3.3-70B",
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
    },
]

# ============================================================================
# Llama 4 MoE Models
# ============================================================================

LLAMA4_MOE_VARIANTS = [
    {
        "name": "Llama-4-Scout-17B-16E",
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
        "shared_each": 0,
        "router_each": 81_920,
        "moe_layer_params": 2_013_347_840,
        "dense_layer_params": 314_572_800,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
    },
    {
        "name": "Llama-4-Maverick-17B-128E",
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
        "shared_each": 0,
        "router_each": 655_360,
        "moe_layer_params": 16_106_782_720,
        "dense_layer_params": 314_572_800,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
    },
]

# ============================================================================
# Qwen2.5 Dense Models
# ============================================================================

QWEN25_DENSE_VARIANTS = [
    {
        "name": "Qwen2.5-7B",
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
    },
    {
        "name": "Qwen2.5-32B",
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
    },
    {
        "name": "Qwen2.5-72B",
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
    },
]

# ============================================================================
# Qwen3 Dense Models
# ============================================================================

QWEN3_DENSE_VARIANTS = [
    {
        "name": "Qwen3-8B",
        "layers": 36,
        "d": 4096,
        "expert_ffn": 0,
        "dense_ffn": 11008,
        "q_heads": 32,
        "kv_heads": 8,
        "dense_layers": 36,
        "total_params_B": 8.04,
        "active_params_B": 8.04,
        "attn_per_layer": 37_748_736,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 173_015_040,
        "valid_pp": [1, 2, 3, 4, 6, 9, 12, 18, 36],
    },
    {
        "name": "Qwen3-32B",
        "layers": 64,
        "d": 5120,
        "expert_ffn": 0,
        "dense_ffn": 25600,
        "q_heads": 64,
        "kv_heads": 8,
        "dense_layers": 64,
        "total_params_B": 30.5,
        "active_params_B": 30.5,
        "attn_per_layer": 58_982_400,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 452_165_632,
        "valid_pp": [1, 2, 4, 8, 16, 32, 64],
    },
]

# ============================================================================
# Qwen3 MoE Models
# ============================================================================

QWEN3_MOE_VARIANTS = [
    {
        "name": "Qwen3-30B-A3B",
        "layers": 48,
        "d": 2048,
        "expert_ffn": 768,
        "dense_ffn": 6144,
        "q_heads": 32,
        "kv_heads": 4,
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
    },
    {
        "name": "Qwen3-235B-A22B",
        "layers": 94,
        "d": 4096,
        "expert_ffn": 1536,
        "dense_ffn": 12288,
        "q_heads": 64,
        "kv_heads": 4,
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
    },
]

# ============================================================================
# Qwen3.5 Dense Models
# ============================================================================

QWEN35_DENSE_VARIANTS = [
    {
        "name": "Qwen3.5-27B",
        "layers": 64,
        "d": 5120,
        "expert_ffn": 0,
        "dense_ffn": 17408,
        "q_heads": 24,
        "kv_heads": 4,
        "dense_layers": 64,
        "total_params_B": 24.35,
        "active_params_B": 24.35,
        "attn_per_layer": 73_400_320,
        "expert_each": 0,
        "shared_each": 0,
        "router_each": 0,
        "moe_layer_params": 0,
        "dense_layer_params": 340_787_200,
        "valid_pp": [1, 2, 4, 8, 16, 32, 64],
    },
]

# ============================================================================
# Qwen3.5 MoE Models
# ============================================================================

QWEN35_MOE_VARIANTS = [
    {
        "name": "Qwen3.5-35B-A3B",
        "layers": 40,
        "d": 2048,
        "expert_ffn": 512,
        "dense_ffn": 0,
        "q_heads": 16,
        "kv_heads": 2,
        "dense_layers": 0,
        "total_params_B": 34.13,
        "active_params_B": 2.93,
        "attn_per_layer": 18_874_368,
        "expert_each": 3_145_728,
        "shared_each": 3_145_728,
        "router_each": 524_288,
        "moe_layer_params": 808_976_384,
        "dense_layer_params": 18_874_368,
        "valid_pp": [1, 2, 4, 5, 8, 10, 20, 40],
    },
    {
        "name": "Qwen3.5-122B-A10B",
        "layers": 48,
        "d": 3072,
        "expert_ffn": 1024,
        "dense_ffn": 0,
        "q_heads": 32,
        "kv_heads": 2,
        "dense_layers": 0,
        "total_params_B": 120.55,
        "active_params_B": 8.21,
        "attn_per_layer": 53_477_376,
        "expert_each": 9_437_184,
        "shared_each": 9_437_184,
        "router_each": 786_432,
        "moe_layer_params": 2_426_142_720,
        "dense_layer_params": 53_477_376,
        "valid_pp": [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
    },
    {
        "name": "Qwen3.5-397B-A17B",
        "layers": 60,
        "d": 4096,
        "expert_ffn": 1024,
        "dense_ffn": 0,
        "q_heads": 32,
        "kv_heads": 2,
        "dense_layers": 0,
        "total_params_B": 393.74,
        "active_params_B": 14.74,
        "attn_per_layer": 71_303_168,
        "expert_each": 12_582_912,
        "shared_each": 12_582_912,
        "router_each": 2_097_152,
        "moe_layer_params": 6_457_131_008,
        "dense_layer_params": 71_303_168,
        "valid_pp": [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60],
    },
]

# ============================================================================
# DeepSeek MoE Models
# ============================================================================

DEEPSEEK_VARIANTS = [
    {
        "name": "DeepSeek-V3",
        "layers": 61,
        "d": 7168,
        "expert_ffn": 2048,
        "dense_ffn": 18432,
        "q_heads": 128,
        "kv_heads": 128,
        "dense_layers": 3,
        "total_params_B": 671.0,
        "active_params_B": 36.7,
        "attn_per_layer": 187_105_280,
        "expert_each": 44_040_192,
        "shared_each": 396_361_728,
        "router_each": 1_835_008,
        "moe_layer_params": 11_672_485_888,
        "dense_layer_params": 583_467_008,
        "valid_pp": [1, 61],
    },
    {
        "name": "DeepSeek-V3.1",
        "layers": 61,
        "d": 7168,
        "expert_ffn": 2048,
        "dense_ffn": 18432,
        "q_heads": 128,
        "kv_heads": 128,
        "dense_layers": 3,
        "total_params_B": 671.0,
        "active_params_B": 36.7,
        "attn_per_layer": 187_105_280,
        "expert_each": 44_040_192,
        "shared_each": 396_361_728,
        "router_each": 1_835_008,
        "moe_layer_params": 11_672_485_888,
        "dense_layer_params": 583_467_008,
        "valid_pp": [1, 61],
    },
    {
        "name": "DeepSeek-V3.2",
        "layers": 61,
        "d": 7168,
        "expert_ffn": 2048,
        "dense_ffn": 18432,
        "q_heads": 128,
        "kv_heads": 128,
        "dense_layers": 3,
        "total_params_B": 671.0,
        "active_params_B": 36.7,
        "attn_per_layer": 187_105_280,
        "expert_each": 44_040_192,
        "shared_each": 396_361_728,
        "router_each": 1_835_008,
        "moe_layer_params": 11_672_485_888,
        "dense_layer_params": 583_467_008,
        "valid_pp": [1, 61],
    },
    {
        "name": "DeepSeek-V4",
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
```

**Usage:** Before running the calculator, edit this file to include only the selected model(s) in the final `VARIANTS = [...]` line. For example, if the user only wants Llama-3.1-70B:

```python
VARIANTS = LLAMA_DENSE_VARIANTS[1:2]  # Only Llama-3.1-70B
```

Or for a custom model, add a new dict to the appropriate list with all required fields computed using the auto-derived formulas above.
