"""
Build a variant from a Hugging Face ``config.json``.

``variant_loader`` needs ``total_params_B`` and ``active_params_B`` typed in by hand,
and ``calculate_variant_fields`` sizes attention as plain GQA. This module derives
both from the model's own config, so a model that is not in the built-in catalog can
be analyzed without hand arithmetic.

All counting is done by ``mfu_model.model_shape``, one generic parser shared with the
MFU model so the memory phases and the throughput estimate never disagree about the
model. It handles dense and MoE; MHA, GQA, MQA, MLA and linear / hybrid attention;
gated or plain FFN; shared and routed experts; MoE layer placement; tied embeddings;
multi-token-prediction layers (compute only). See docs/CALCULATOR-MFU.md section 2.

``attn_per_layer`` is the average over all layers, so the memory code, which multiplies
it by a layer count, stays correct for a mixed attention stack. Active parameters
include the shared experts (they run for every token). An exact total (safetensors
index) rescales the shape-derived counts.
"""

import json

from utils import mfu_model as mm


# Config keys that identify a hybrid layer stack (GLM-5.3-Flash layer_types).
_LINEAR_ATTN_HINT = "linear"


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def architecture_from_config(config):
    """Flat architecture dict from a config.json.

    Multimodal configs keep the language model under ``text_config``; its keys win.
    A ``layer_types`` list with linear-attention entries plus a ``linear_attn_config``
    block become ``linear_attn_layers`` / ``linear_attn_heads`` / ``linear_attn_head_dim``.
    """
    if not isinstance(config, dict):
        raise ValueError("config.json must be a JSON object")
    text = config.get("text_config")
    arch = {**config, **(text if isinstance(text, dict) else {})}

    layer_types = arch.get("layer_types")
    lin_cfg = arch.get("linear_attn_config")
    if isinstance(layer_types, list) and isinstance(lin_cfg, dict):
        linear = sum(1 for t in layer_types if _LINEAR_ATTN_HINT in str(t).lower())
        heads, dim = lin_cfg.get("num_heads"), lin_cfg.get("head_dim")
        if linear and isinstance(heads, int) and isinstance(dim, int):
            arch["linear_attn_layers"] = linear
            arch["linear_attn_heads"] = heads
            arch["linear_attn_head_dim"] = dim
    return arch


def variant_from_architecture(arch, name, params_total_b=None):
    """Variant dict (same fields as variants_config) from flat config.json fields.

    Args:
        arch: flat architecture dict, see ``architecture_from_config``.
        name: variant name.
        params_total_b: exact total parameter count in billions (e.g. from the
            safetensors index). Wins over the count derived from the shape.
    """
    shape = mm.model_shape(arch, params_total_b)
    a = shape["attention"]
    layers = shape["layers"]
    variant = {
        "name": name or "custom",
        "layers": layers,
        "d": shape["d"],
        "expert_ffn": shape["expert_ffn"],
        "dense_ffn": shape["dense_ffn"],
        "q_heads": a["heads"],
        "kv_heads": a["kv_heads"],
        "head_dim": a["d_qk"],
        "dense_layers": shape["dense_layers"],
        "total_params_B": shape["total_b"],
        "active_params_B": shape["active_b"] if shape["n_experts"] else shape["total_b"],
        "attn_per_layer": shape["attn_per_layer"],
        "expert_each": shape["expert_each"],
        "shared_each": shape["shared_each"],
        "router_each": shape["router_each"],
        "moe_layer_params": shape["moe_layer_params"],
        "dense_layer_params": shape["dense_layer_params"],
        "valid_pp": _divisors(layers),
        "attention": a["kind"],
        "_shape": shape,
    }
    if shape["vocab"]:
        variant["vocab"] = shape["vocab"]
    if shape["tied_embeddings"]:
        variant["tied_embeddings"] = True
    if shape["n_experts"]:
        variant["num_experts"] = shape["n_experts"]
        variant["topk"] = shape["topk"]
    if a["kind"].startswith("MLA"):
        # Widths of the score / context matmuls; see compute_attention_flops_dim.
        variant["qk_head_dim"] = a["d_qk"]
        variant["v_head_dim"] = a["d_v"]
    return variant


def load_variant_from_hf_config(path, name=None, params_total_b=None):
    """Read a config.json from disk and return one completed variant."""
    with open(path) as f:
        config = json.load(f)
    arch = architecture_from_config(config)
    return variant_from_architecture(arch, name or arch.get("model_type") or "custom",
                                     params_total_b=params_total_b)
