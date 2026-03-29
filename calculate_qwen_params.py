#!/usr/bin/env python3
"""
Helper script to calculate model parameters for Qwen3 models.

Fetches config.json from Hugging Face and calculates all parameters needed
for variants_config.py including component-wise parameter counts.

Usage:
    python3 calculate_qwen_params.py --model "Qwen/Qwen3-30B-A3B"
    python3 calculate_qwen_params.py --config path/to/config.json
"""

import json
import argparse
import urllib.request
import sys
from typing import Dict, Any, List, Optional


def fetch_hf_config(model_name: str) -> Dict[str, Any]:
    """
    Fetch config.json from Hugging Face model repository.

    Args:
        model_name: Hugging Face model name (e.g., "Qwen/Qwen3-30B-A3B")

    Returns:
        Dictionary containing model configuration
    """
    url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    print(f"Fetching config from: {url}")

    try:
        with urllib.request.urlopen(url) as response:
            config = json.loads(response.read().decode('utf-8'))
        print("✓ Config fetched successfully")
        return config
    except Exception as e:
        print(f"✗ Error fetching config: {e}")
        sys.exit(1)


def load_local_config(config_path: str) -> Dict[str, Any]:
    """Load config.json from local file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Config loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)


def detect_model_type(config: Dict[str, Any]) -> str:
    """
    Detect if model is dense or MoE based on config structure.

    Returns:
        "moe" or "dense"
    """
    if 'num_experts' in config and config.get('num_experts', 0) > 0:
        return "moe"
    return "dense"


def calculate_attention_params(config: Dict[str, Any]) -> int:
    """
    Calculate attention parameters per layer.

    Formula: attn_per_layer = d² × 2.25

    The 2.25 factor accounts for:
    - Q, K, V projection matrices (3 * d²)
    - Output projection (d²)
    - Additional attention mechanism parameters

    Simplified empirical formula: d² × 2.25
    """
    d = config['hidden_size']
    return int(d * d * 2.25)


def calculate_expert_params(config: Dict[str, Any]) -> int:
    """
    Calculate parameters per expert in MoE layer.

    Formula: expert_each = d × expert_ffn × 3

    The factor of 3 accounts for:
    - Gate projection (d × expert_ffn)
    - Up projection (d × expert_ffn)
    - Down projection (expert_ffn × d)
    """
    d = config['hidden_size']
    expert_ffn = config.get('moe_intermediate_size', 0)
    return d * expert_ffn * 3


def calculate_router_params(config: Dict[str, Any]) -> int:
    """
    Calculate router parameters per MoE layer.

    Formula: router_each = d × num_experts

    The router maps hidden states to expert selection logits.
    """
    d = config['hidden_size']
    num_experts = config.get('num_experts', 0)
    return d * num_experts


def calculate_dense_layer_params(config: Dict[str, Any], attn_per_layer: int) -> int:
    """
    Calculate parameters for a dense (non-MoE) layer.

    Formula: dense_layer_params = attn_per_layer + (d × dense_ffn × 3)

    Includes both attention and FFN components.
    """
    d = config['hidden_size']
    dense_ffn = config.get('intermediate_size', 0)
    ffn_params = d * dense_ffn * 3  # Gate, up, down projections
    return attn_per_layer + ffn_params


def calculate_moe_layer_params(
    config: Dict[str, Any],
    expert_each: int,
    router_each: int
) -> int:
    """
    Calculate total parameters in a MoE layer.

    Formula: moe_layer_params = (num_experts × expert_each) + router_each

    Note: Qwen3 has NO shared experts, so we don't add shared expert params.
    """
    num_experts = config.get('num_experts', 0)
    return (num_experts * expert_each) + router_each


def calculate_total_params(
    config: Dict[str, Any],
    model_type: str,
    attn_per_layer: int,
    moe_layer_params: int,
    dense_layer_params: int,
    dense_layers: int
) -> tuple:
    """
    Calculate total and active parameters.

    Returns:
        (total_params_B, active_params_B) in billions
    """
    d = config['hidden_size']
    layers = config['num_hidden_layers']
    vocab = config.get('vocab_size', 151936)

    # Embedding parameters
    embed_params = 2 * vocab * d  # Word + position embeddings

    # Layer norm parameters
    ln_params = layers * 2 * d  # Pre- and post-attention layer norms

    # Attention parameters
    attn_params = layers * attn_per_layer

    if model_type == "moe":
        # MoE model
        moe_layers = layers - dense_layers
        total_params = (
            embed_params +
            ln_params +
            attn_params +
            (moe_layers * moe_layer_params) +
            (dense_layers * (dense_layer_params - attn_per_layer))  # Dense FFN only
        )

        # Active params: only top-k experts are active per token
        num_experts_per_tok = config.get('num_experts_per_tok', 8)
        num_experts = config.get('num_experts', 128)
        expert_each = calculate_expert_params(config)
        router_each = calculate_router_params(config)

        active_expert_params = num_experts_per_tok * expert_each
        active_moe_layer = active_expert_params + router_each + attn_per_layer

        active_params = (
            embed_params +
            ln_params +
            (moe_layers * active_moe_layer) +
            (dense_layers * dense_layer_params)
        )
    else:
        # Dense model
        total_params = (
            embed_params +
            ln_params +
            attn_params +
            (layers * (dense_layer_params - attn_per_layer))  # Dense FFN only
        )
        active_params = total_params  # All params active in dense models

    return (total_params / 1e9, active_params / 1e9)


def calculate_valid_pp(num_layers: int, max_pp: int = 48) -> List[int]:
    """
    Calculate valid pipeline parallelism values.

    Valid PP values must divide num_layers evenly for balanced pipeline stages.
    """
    valid_pp = []
    for pp in range(1, min(num_layers + 1, max_pp + 1)):
        if num_layers % pp == 0:
            valid_pp.append(pp)
    return valid_pp


def format_variant_dict(
    config: Dict[str, Any],
    model_name: str,
    model_type: str
) -> Dict[str, Any]:
    """
    Calculate all parameters and format as variant dictionary.

    Returns:
        Dictionary ready for copy-paste into variants_config.py
    """
    # Extract architecture parameters
    d = config['hidden_size']
    layers = config['num_hidden_layers']
    q_heads = config['num_attention_heads']
    kv_heads = config.get('num_key_value_heads', q_heads)

    # Calculate component parameters
    attn_per_layer = calculate_attention_params(config)

    if model_type == "moe":
        expert_ffn = config.get('moe_intermediate_size', 0)
        dense_ffn = config.get('intermediate_size', 0)
        expert_each = calculate_expert_params(config)
        router_each = calculate_router_params(config)
        shared_each = 0  # Qwen3 has no shared experts

        # Determine dense layers (usually 0 for full MoE)
        mlp_only_layers = config.get('mlp_only_layers', [])
        dense_layers = len(mlp_only_layers) if mlp_only_layers else 0

        moe_layer_params = calculate_moe_layer_params(config, expert_each, router_each)
        dense_layer_params = calculate_dense_layer_params(config, attn_per_layer) if dense_layers > 0 else 0
    else:
        # Dense model
        expert_ffn = 0
        dense_ffn = config.get('intermediate_size', 0)
        expert_each = 0
        router_each = 0
        shared_each = 0
        dense_layers = layers
        moe_layer_params = 0
        dense_layer_params = calculate_dense_layer_params(config, attn_per_layer)

    # Calculate total parameters
    total_params_b, active_params_b = calculate_total_params(
        config, model_type, attn_per_layer, moe_layer_params,
        dense_layer_params, dense_layers
    )

    # Calculate valid PP values
    valid_pp = calculate_valid_pp(layers)

    return {
        "name": model_name,
        "layers": layers,
        "d": d,
        "expert_ffn": expert_ffn,
        "dense_ffn": dense_ffn,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "dense_layers": dense_layers,
        "total_params_B": round(total_params_b, 2),
        "active_params_B": round(active_params_b, 2),
        "moe_layer_params": moe_layer_params,
        "dense_layer_params": dense_layer_params,
        "attn_per_layer": attn_per_layer,
        "expert_each": expert_each,
        "shared_each": shared_each,
        "router_each": router_each,
        "valid_pp": valid_pp,
    }


def verify_calculations(variant: Dict[str, Any], published_params: Optional[float] = None):
    """
    Verify calculated parameters and print analysis.
    """
    print("\n" + "="*70)
    print(f"MODEL: {variant['name']}")
    print("="*70)

    print(f"\nArchitecture:")
    print(f"  Layers:           {variant['layers']}")
    print(f"  Hidden size (d):  {variant['d']}")
    print(f"  Query heads:      {variant['q_heads']}")
    print(f"  KV heads:         {variant['kv_heads']}")
    print(f"  Expert FFN:       {variant['expert_ffn']}")
    print(f"  Dense FFN:        {variant['dense_ffn']}")
    print(f"  Dense layers:     {variant['dense_layers']}")

    print(f"\nComponent Parameters:")
    print(f"  attn_per_layer:     {variant['attn_per_layer']:,}")
    print(f"  expert_each:        {variant['expert_each']:,}")
    print(f"  shared_each:        {variant['shared_each']:,} (Qwen3 has no shared experts)")
    print(f"  router_each:        {variant['router_each']:,}")
    print(f"  moe_layer_params:   {variant['moe_layer_params']:,}")
    print(f"  dense_layer_params: {variant['dense_layer_params']:,}")

    print(f"\nTotal Parameters:")
    print(f"  Calculated:  {variant['total_params_B']:.2f}B")
    print(f"  Active:      {variant['active_params_B']:.2f}B")

    if published_params is not None:
        diff = abs(variant['total_params_B'] - published_params)
        diff_pct = (diff / published_params) * 100
        print(f"  Published:   {published_params:.2f}B")
        print(f"  Difference:  {diff:.2f}B ({diff_pct:.2f}%)")

        if diff_pct < 1.0:
            print("  ✓ Match within 1% tolerance")
        elif diff_pct < 5.0:
            print("  ⚠ Match within 5% tolerance (acceptable)")
        else:
            print("  ✗ Difference exceeds 5% (investigate)")

    print(f"\nValid Pipeline Parallelism: {variant['valid_pp']}")


def format_python_dict(variant: Dict[str, Any]) -> str:
    """Format variant dictionary as Python code for copy-paste."""
    return f"""    {{
        "name": "{variant['name']}",
        "layers": {variant['layers']},
        "d": {variant['d']},
        "expert_ffn": {variant['expert_ffn']},
        "dense_ffn": {variant['dense_ffn']},
        "q_heads": {variant['q_heads']},
        "kv_heads": {variant['kv_heads']},
        "dense_layers": {variant['dense_layers']},
        "total_params_B": {variant['total_params_B']},
        "active_params_B": {variant['active_params_B']},
        "moe_layer_params": {variant['moe_layer_params']:_},
        "dense_layer_params": {variant['dense_layer_params']:_},
        "attn_per_layer": {variant['attn_per_layer']:_},
        "expert_each": {variant['expert_each']:_},
        "shared_each": {variant['shared_each']:_},
        "router_each": {variant['router_each']:_},
        "valid_pp": {variant['valid_pp']},
    }},"""


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Qwen3 model parameters for variants_config.py"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help='Hugging Face model name (e.g., "Qwen/Qwen3-30B-A3B")'
    )
    group.add_argument(
        "--config",
        help="Path to local config.json file"
    )
    parser.add_argument(
        "--published-params",
        type=float,
        help="Published parameter count in billions for verification"
    )
    parser.add_argument(
        "--output",
        help="Output file for variant dictionary (default: print to stdout)"
    )

    args = parser.parse_args()

    # Load config
    if args.model:
        config = fetch_hf_config(args.model)
        model_name = args.model.split('/')[-1]  # Extract model name
    else:
        config = load_local_config(args.config)
        model_name = input("Enter model name (e.g., 'Qwen3-30B-A3B'): ").strip()

    # Detect model type
    model_type = detect_model_type(config)
    print(f"✓ Detected model type: {model_type.upper()}")

    # Calculate all parameters
    variant = format_variant_dict(config, model_name, model_type)

    # Verify calculations
    verify_calculations(variant, args.published_params)

    # Output Python dictionary
    print("\n" + "="*70)
    print("PYTHON DICTIONARY FOR variants_config.py")
    print("="*70)
    python_code = format_python_dict(variant)
    print(python_code)

    # Optionally save to file
    if args.output:
        with open(args.output, 'w') as f:
            f.write(python_code)
        print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
