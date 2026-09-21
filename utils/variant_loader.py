"""
Variant loading and selection helpers for the AI Model Calculator CLI.

Provides:
  - load_variants_from_json: parse a JSON architecture file into a list of
    complete variant dicts (derived fields auto-computed).
  - select_variant_by_name: case-insensitive exact match of a variant name.

A variant's expert count for derived-field math is taken from an explicit
per-variant field (``num_experts``/``n_experts``) when present, otherwise from
the project-level ``N_EXPERTS`` passed in by the caller.
"""

import json

from utils.calculate_variant_fields import validate_and_complete_variants


class VariantNotFoundError(ValueError):
    """Raised when a requested model name does not match any variant."""

    def __init__(self, name, available):
        self.name = name
        self.available = list(available)
        super().__init__(name)


# Base architecture fields the analysis phases read directly and which cannot
# be derived by calculate_variant_fields. A JSON variant must supply these.
REQUIRED_BASE_FIELDS = (
    "name",
    "layers",
    "d",
    "dense_layers",
    "total_params_B",
    "active_params_B",
    "valid_pp",
)


def _validate_base_fields(variant, index):
    """Ensure a JSON variant carries the base fields the phases require."""
    missing = [f for f in REQUIRED_BASE_FIELDS if f not in variant]
    if missing:
        raise ValueError(
            f"Variant '{variant.get('name', f'#{index}')}' is missing required "
            f"base field(s): {', '.join(missing)}. Derived fields "
            f"(attn_per_layer, expert_each, moe_layer_params, etc.) are computed "
            f"automatically, but base architecture fields must be provided."
        )


def _extract_variant_list(data):
    """Normalize parsed JSON into a list of variant dicts.

    Accepts three shapes:
      - a single variant object: {"name": ..., ...}
      - a bare list of variant objects: [ {...}, {...} ]
      - a wrapper object: {"variants": [ {...}, ... ]}
    """
    if isinstance(data, dict):
        if "variants" in data:
            variants = data["variants"]
        else:
            variants = [data]
    elif isinstance(data, list):
        variants = data
    else:
        raise ValueError(
            f"Unsupported JSON structure: expected object or list, got {type(data).__name__}"
        )

    if not isinstance(variants, list):
        raise ValueError('"variants" must be a list of variant objects')
    if not variants:
        raise ValueError("JSON file contains no variants")
    for i, v in enumerate(variants):
        if not isinstance(v, dict):
            raise ValueError(f"Variant at index {i} is not an object")
        if "name" not in v:
            raise ValueError(f"Variant at index {i} is missing required field 'name'")
        _validate_base_fields(v, i)
    return variants


def load_variants_from_json(path, default_num_experts):
    """Load and complete variants from a JSON file.

    Args:
        path: Path to the JSON architecture file.
        default_num_experts: Fallback expert count (project-level N_EXPERTS)
            used for a variant that does not explicitly declare its own.

    Returns:
        list: Variant dicts with all derived fields populated.
    """
    with open(path) as f:
        data = json.load(f)

    variants = _extract_variant_list(data)

    completed = []
    for v in variants:
        # Honor an explicit per-variant expert count when present; otherwise
        # fall back to the project-level default.
        num_experts = v.get("num_experts", v.get("n_experts", default_num_experts))
        _, one = validate_and_complete_variants(
            [v], num_experts=num_experts, verbose=False
        )
        completed.append(one[0])

    return completed


def select_variant_by_name(variants, name):
    """Return the single variant whose name matches (case-insensitive).

    Args:
        variants: List of variant dicts.
        name: Requested model name (matched case-insensitively, exact).

    Returns:
        The matching variant dict.

    Raises:
        VariantNotFoundError: If no variant name matches. Carries the list of
            available names so the caller can present them.
    """
    target = name.strip().lower()
    for v in variants:
        if v["name"].lower() == target:
            return v
    raise VariantNotFoundError(name, [v["name"] for v in variants])
