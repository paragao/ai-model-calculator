"""
Helper script to calculate missing variant configuration fields.

This script automatically calculates derived fields for model variants based on
their basic architecture parameters. It ensures all variants have the complete
set of fields required for memory and training analysis.

Formulas used:
- attn_per_layer = 2 * d * (q_heads + kv_heads) * head_dim
  (falls back to d² × 2.25 when head_dim = d / q_heads)
- expert_each = d × expert_ffn × 3
- shared_each = d × dense_ffn × 3 (shared expert uses dense FFN dimension)
- router_each = d × num_experts
- moe_layer_params = (num_experts × expert_each) + shared_each + router_each
- dense_layer_params = attn_per_layer + (d × dense_ffn × 3)
"""

import os
import sys
from pathlib import Path

# Default number of experts (can be overridden)
DEFAULT_NUM_EXPERTS = 128


def calculate_attn_per_layer(d, q_heads=None, kv_heads=None, head_dim=None):
    """
    Calculate attention parameters per layer.

    Formula: 2 * d * (q_heads + kv_heads) * head_dim
    This accounts for Q, K, V projections and output projection.

    If q_heads/kv_heads/head_dim are not provided, falls back to d^2 * 2.25
    (which assumes head_dim = d / q_heads and kv_heads = q_heads / 8).

    Args:
        d: Hidden dimension
        q_heads: Number of query attention heads (optional)
        kv_heads: Number of key/value attention heads (optional)
        head_dim: Dimension per attention head (optional, defaults to d // q_heads)

    Returns:
        int: Number of attention parameters per layer
    """
    if q_heads is not None and kv_heads is not None:
        if head_dim is None:
            head_dim = d // q_heads
        return int(2 * d * (q_heads + kv_heads) * head_dim)
    else:
        # Legacy fallback for models where head_dim = d / q_heads
        return int(d * d * 2.25)


def calculate_expert_each(d, expert_ffn):
    """
    Calculate parameters per expert.

    Formula: d × expert_ffn × 3
    This accounts for: up projection, gate projection, down projection.

    Args:
        d: Hidden dimension
        expert_ffn: Expert FFN intermediate dimension

    Returns:
        int: Number of parameters per expert
    """
    return d * expert_ffn * 3


def calculate_shared_each(d, dense_ffn):
    """
    Calculate parameters for shared expert (if present).

    Formula: d × dense_ffn × 3
    The shared expert uses the dense FFN dimension (not expert FFN dimension),
    matching the Qwen3 architecture where the shared expert has the same
    intermediate size as the dense layers.

    Args:
        d: Hidden dimension
        dense_ffn: Dense FFN intermediate dimension (used by shared expert)

    Returns:
        int: Number of parameters for shared expert
    """
    return d * dense_ffn * 3


def calculate_router_each(d, num_experts=DEFAULT_NUM_EXPERTS):
    """
    Calculate router parameters per MoE layer.

    Formula: d × num_experts
    The router maps hidden states to expert selection logits.

    Args:
        d: Hidden dimension
        num_experts: Total number of experts

    Returns:
        int: Number of router parameters per layer
    """
    return d * num_experts


def calculate_moe_layer_params(d, expert_ffn, dense_ffn, num_experts=DEFAULT_NUM_EXPERTS, has_shared=True):
    """
    Calculate total parameters for a MoE layer.

    Formula: (num_experts × expert_each) + shared_each + router_each

    Args:
        d: Hidden dimension
        expert_ffn: Expert FFN intermediate dimension
        dense_ffn: Dense FFN intermediate dimension (used by shared expert)
        num_experts: Total number of experts
        has_shared: Whether the layer has shared experts

    Returns:
        int: Total parameters for MoE layer
    """
    expert_each = calculate_expert_each(d, expert_ffn)
    shared_each = calculate_shared_each(d, dense_ffn) if has_shared else 0
    router_each = calculate_router_each(d, num_experts)

    return (num_experts * expert_each) + shared_each + router_each


def calculate_dense_layer_params(d, dense_ffn):
    """
    Calculate total parameters for a dense (standard FFN) layer.

    Formula: attn_per_layer + (d × dense_ffn × 3)
    This includes attention params plus FFN (up, gate, down projections).

    Args:
        d: Hidden dimension
        dense_ffn: Dense FFN intermediate dimension

    Returns:
        int: Total parameters for dense layer
    """
    attn_per_layer = calculate_attn_per_layer(d)
    ffn_params = d * dense_ffn * 3

    return attn_per_layer + ffn_params


def check_variant_completeness(variant):
    """
    Check if a variant has all required calculated fields.

    Args:
        variant: Variant dictionary

    Returns:
        tuple: (is_complete: bool, missing_fields: list)
    """
    required_fields = [
        'attn_per_layer',
        'expert_each',
        'shared_each',
        'router_each',
        'moe_layer_params',
        'dense_layer_params'
    ]

    missing_fields = []
    for field in required_fields:
        if field not in variant or variant[field] is None:
            missing_fields.append(field)

    return (len(missing_fields) == 0, missing_fields)


def calculate_missing_fields(variant, num_experts=DEFAULT_NUM_EXPERTS):
    """
    Calculate all missing fields for a variant.

    Args:
        variant: Variant dictionary with at least d, expert_ffn, dense_ffn
        num_experts: Number of experts (default 128)

    Returns:
        dict: Dictionary with calculated fields
    """
    d = variant['d']
    expert_ffn = variant.get('expert_ffn', 0)
    dense_ffn = variant.get('dense_ffn', 0)

    # Determine if this variant has shared experts
    # If expert_ffn > 0, assume it has shared experts unless explicitly set to 0
    has_shared = expert_ffn > 0 and variant.get('shared_each', 0) != 0

    calculated = {}

    # Calculate attention parameters
    q_heads = variant.get('q_heads', None)
    kv_heads = variant.get('kv_heads', None)
    head_dim = variant.get('head_dim', None)
    calculated['attn_per_layer'] = calculate_attn_per_layer(d, q_heads, kv_heads, head_dim)

    # Calculate expert-related parameters
    if expert_ffn > 0:
        calculated['expert_each'] = calculate_expert_each(d, expert_ffn)
        calculated['shared_each'] = calculate_shared_each(d, dense_ffn) if has_shared else 0
        calculated['router_each'] = calculate_router_each(d, num_experts)
        calculated['moe_layer_params'] = calculate_moe_layer_params(d, expert_ffn, dense_ffn, num_experts, has_shared)
    else:
        # Dense model (no experts)
        calculated['expert_each'] = 0
        calculated['shared_each'] = 0
        calculated['router_each'] = 0
        calculated['moe_layer_params'] = 0

    # Calculate dense layer parameters
    if dense_ffn > 0:
        calculated['dense_layer_params'] = calculate_dense_layer_params(d, dense_ffn)
    else:
        calculated['dense_layer_params'] = 0

    return calculated


def validate_and_complete_variants(variants, num_experts=DEFAULT_NUM_EXPERTS, verbose=True):
    """
    Check all variants and calculate missing fields.

    Args:
        variants: List of variant dictionaries
        num_experts: Number of experts (default 128)
        verbose: Print information about missing fields

    Returns:
        tuple: (all_complete: bool, updated_variants: list)
    """
    all_complete = True
    updated_variants = []

    for variant in variants:
        is_complete, missing_fields = check_variant_completeness(variant)

        if not is_complete:
            all_complete = False
            if verbose:
                print(f"⚠️  Variant '{variant['name']}' missing fields: {missing_fields}")
                print(f"   Calculating missing fields...")

            # Calculate missing fields
            calculated = calculate_missing_fields(variant, num_experts)

            # Merge calculated fields into variant
            updated_variant = variant.copy()
            for field, value in calculated.items():
                if field not in updated_variant or updated_variant[field] is None:
                    updated_variant[field] = value
                    if verbose:
                        print(f"   {field} = {value:,}")

            updated_variants.append(updated_variant)
        else:
            updated_variants.append(variant)

    return (all_complete, updated_variants)


def update_variants_config_file(variants, config_file_path):
    """
    Update the variants_config.py file with completed variants.

    This function rewrites the VARIANTS list in the config file with
    properly formatted entries including all calculated fields.

    Args:
        variants: List of complete variant dictionaries
        config_file_path: Path to variants_config.py file
    """
    # Read the current file to preserve header comments
    with open(config_file_path, 'r') as f:
        lines = f.readlines()

    # Find where VARIANTS starts
    variants_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('VARIANTS = ['):
            variants_start = i
            break

    if variants_start is None:
        raise ValueError("Could not find 'VARIANTS = [' in config file")

    # Keep everything before VARIANTS definition
    header = ''.join(lines[:variants_start])

    # Generate new VARIANTS definition
    variants_content = "VARIANTS = [\n"

    for variant in variants:
        variants_content += "    {\n"

        # Write fields in a consistent order
        field_order = [
            'name', 'layers', 'd', 'expert_ffn', 'dense_ffn',
            'q_heads', 'kv_heads', 'dense_layers',
            'total_params_B', 'active_params_B',
            'moe_layer_params', 'dense_layer_params',
            'attn_per_layer', 'expert_each', 'shared_each', 'router_each',
            'valid_pp'
        ]

        for field in field_order:
            if field in variant:
                value = variant[field]

                # Format based on field type
                if isinstance(value, str):
                    variants_content += f'        "{field}": "{value}",\n'
                elif isinstance(value, list):
                    variants_content += f'        "{field}": {value},\n'
                elif isinstance(value, (int, float)) and field in [
                    'moe_layer_params', 'dense_layer_params',
                    'attn_per_layer', 'expert_each', 'shared_each', 'router_each'
                ]:
                    # Format large numbers with underscores
                    if isinstance(value, int) and value >= 1000:
                        formatted = f"{value:_}"
                        variants_content += f'        "{field}": {formatted},\n'
                    else:
                        variants_content += f'        "{field}": {value},\n'
                else:
                    variants_content += f'        "{field}": {value},\n'

        variants_content += "    },\n"

    variants_content += "]\n"

    # Write updated file
    with open(config_file_path, 'w') as f:
        f.write(header)
        f.write(variants_content)

    print(f"✅ Updated {config_file_path} with calculated fields")


def main():
    """
    Main function for standalone usage.

    Usage:
        python3 calculate_variant_fields.py [--update]

    Options:
        --update: Automatically update variants_config.py file
    """
    # Add parent directory to path to import config
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    sys.path.insert(0, str(project_dir))

    from configuration.variants_config import VARIANTS
    from configuration.project_config import N_EXPERTS

    print("="*70)
    print("VARIANT FIELD CALCULATOR")
    print("="*70)
    print()

    # Check and calculate missing fields
    all_complete, updated_variants = validate_and_complete_variants(
        VARIANTS,
        num_experts=N_EXPERTS,
        verbose=True
    )

    if all_complete:
        print("\n✅ All variants are complete!")
    else:
        print(f"\n⚠️  Found variants with missing fields")

        # Check if --update flag is provided
        if '--update' in sys.argv:
            config_file = project_dir / 'configuration' / 'variants_config.py'
            print(f"\nUpdating {config_file}...")
            update_variants_config_file(updated_variants, config_file)
        else:
            print("\nTo update variants_config.py, run:")
            print(f"  python3 {Path(__file__).name} --update")


if __name__ == "__main__":
    main()
