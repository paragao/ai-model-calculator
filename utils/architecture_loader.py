"""
Build a variant from a Hugging Face ``config.json``.

``variant_loader`` needs ``total_params_B`` and ``active_params_B`` typed in by hand,
and ``calculate_variant_fields`` sizes attention as plain GQA. This module derives
both from the model's own config, so a model that is not in the built-in catalog can
be analyzed without hand arithmetic.

What it counts, and why each term is here:

* Attention, per layer, from the config fields:
    - GQA / MHA:  2 * d * (q_heads + kv_heads) * head_dim
    - MLA (``kv_lora_rank`` set, DeepSeek style):
        q  = d * q_lora + q_lora * H * qk_head_dim   (d * H * qk_head_dim without q_lora)
        kv = d * (kv_lora + rope) + kv_lora * H * (nope + v_head_dim)
        o  = H * v_head_dim * d
      Sizing an MLA model as GQA with ``head_dim = d / heads`` is wrong, and
      ``head_dim`` is 0 in some MLA configs, which makes the GQA formula return 0.
    - Hybrid linear attention (``linear_attn_layers`` of the layers): q, k, v and
      output projections, 4 * d * linear_attn_heads * linear_attn_head_dim.
  ``attn_per_layer`` is the average over all layers, so the memory and parameter
  code, which multiplies it by a layer count, stays correct for a mixed stack.
* Shared experts. They run for every token, so they belong to the ACTIVE parameter
  count as well as the total. Their width is the first of:
  ``shared_expert_intermediate_size`` (Qwen-MoE),
  ``n_shared_experts * moe_intermediate_size`` (DeepSeek, GLM, Param2),
  ``moe_shared_expert_intermediate_size``.
* Active parameters = dense layers + MoE layers with attention, ``topk`` routed
  experts, the shared experts, the router, and the embeddings. When an exact total
  is given (from the safetensors index) the shape-derived active count is scaled by
  exact_total / shape_total, so the two stay on the same footing.
"""

import json


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


def _num(arch):
    def num(*keys, default=0):
        for k in keys:
            v = arch.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return int(v)
        return default
    return num


def attention_params(num, d, layers, q_heads, kv_heads, head_dim):
    """(attention parameters summed over all layers, attention kind)."""
    kv_lora = num("kv_lora_rank")
    if kv_lora:
        rope = num("qk_rope_head_dim")
        qk = (num("qk_nope_head_dim") + rope) or num("qk_head_dim") or head_dim
        nope = max(qk - rope, 0)
        v = num("v_head_dim", default=qk)
        q_lora = num("q_lora_rank")
        q = d * q_lora + q_lora * q_heads * qk if q_lora else d * q_heads * qk
        full = q + d * (kv_lora + rope) + kv_lora * q_heads * (nope + v) + q_heads * v * d
        kind = "MLA"
    else:
        full = 2 * d * (q_heads + kv_heads) * head_dim
        kind = "GQA" if kv_heads < q_heads else "MHA"
    linear = min(layers, num("linear_attn_layers"))
    if not linear:
        return full * layers, kind
    inner = num("linear_attn_heads", default=q_heads) * num("linear_attn_head_dim", default=head_dim)
    return (full * (layers - linear) + 4 * d * inner * linear,
            f"{kind} in {layers - linear} layers, linear attention in {linear}")


def variant_from_architecture(arch, name, params_total_b=None):
    """Variant dict (same fields as variants_config) from config.json fields.

    Args:
        arch: flat architecture dict, see ``architecture_from_config``.
        name: variant name.
        params_total_b: exact total parameter count in billions (e.g. from the
            safetensors index). Wins over the count derived from the shape.
    """
    num = _num(arch)

    layers = num("num_hidden_layers", "num_layers", "n_layer")
    d = num("hidden_size", "d_model", "n_embd")
    if not layers or not d:
        raise ValueError("architecture needs num_hidden_layers and hidden_size")
    q_heads = num("num_attention_heads", "n_head", default=max(1, d // 128))
    kv_heads = num("num_key_value_heads", default=q_heads)
    head_dim = num("head_dim", default=d // q_heads)
    dense_ffn = num("intermediate_size", "ffn_dim", default=4 * d)
    vocab = num("vocab_size")
    n_experts = num("n_routed_experts", "num_experts", "num_local_experts")
    topk = num("num_experts_per_tok", "moe_topk", "top_k") if n_experts else 0
    expert_ffn = num("moe_intermediate_size", default=dense_ffn) if n_experts else 0
    shared_ffn = 0
    if n_experts:
        shared_ffn = (num("shared_expert_intermediate_size")
                      or num("n_shared_experts", "num_shared_experts") * expert_ffn
                      or num("moe_shared_expert_intermediate_size"))
    dense_layers = layers if not n_experts else min(layers, num("first_k_dense_replace"))

    attn_total, attn_kind = attention_params(num, d, layers, q_heads, kv_heads, head_dim)
    attn = attn_total // layers
    dense_layer = attn + 3 * d * dense_ffn
    expert_each = 3 * d * expert_ffn
    shared_each = 3 * d * shared_ffn
    router_each = d * n_experts
    moe_layer = n_experts * expert_each + shared_each + router_each
    tied = arch.get("tie_word_embeddings", True) is not False
    embed = vocab * d * (1 if tied else 2)
    moe_layers = layers - dense_layers
    shape_total = dense_layers * dense_layer + moe_layers * (attn + moe_layer) + embed
    active = (dense_layers * dense_layer
              + moe_layers * (attn + topk * expert_each + shared_each + router_each)
              + embed)
    total_b = round(float(params_total_b), 2) if params_total_b else round(shape_total / 1e9, 2)
    active_b = total_b if not n_experts else round(active / 1e9 * (total_b / (shape_total / 1e9)), 2)

    variant = {
        "name": name or "custom",
        "layers": layers,
        "d": d,
        "expert_ffn": expert_ffn,
        "dense_ffn": dense_ffn,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dense_layers": dense_layers,
        "total_params_B": total_b,
        "active_params_B": active_b,
        "attn_per_layer": attn,
        "expert_each": expert_each,
        "shared_each": shared_each,
        "router_each": router_each,
        "moe_layer_params": moe_layer,
        "dense_layer_params": dense_layer,
        "valid_pp": _divisors(layers),
        "attention": attn_kind,
    }
    if vocab:
        variant["vocab"] = vocab
    if tied:
        variant["tied_embeddings"] = True
    if n_experts:
        variant["num_experts"] = n_experts
        variant["topk"] = topk
    if num("kv_lora_rank"):
        # Widths of the score / context matmuls; see compute_attention_flops_dim.
        rope = num("qk_rope_head_dim")
        qk = (num("qk_nope_head_dim") + rope) or num("qk_head_dim") or head_dim
        variant["qk_head_dim"] = qk
        variant["v_head_dim"] = num("v_head_dim", default=qk)
        variant["kv_lora_rank"] = num("kv_lora_rank")
        variant["qk_rope_head_dim"] = rope
    return variant


def load_variant_from_hf_config(path, name=None, params_total_b=None):
    """Read a config.json from disk and return one completed variant."""
    with open(path) as f:
        config = json.load(f)
    arch = architecture_from_config(config)
    return variant_from_architecture(arch, name or arch.get("model_type") or "custom",
                                     params_total_b=params_total_b)
