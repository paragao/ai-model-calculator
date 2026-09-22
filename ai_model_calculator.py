#!/usr/bin/python3
"""Command-line entry point for the AI Model Calculator.

Parses arguments, resolves which model variant(s) to analyze, and delegates the
actual six-phase computation to ``model_calculations.run``.

Usage:
    python3 ai_model_calculator.py [--model NAME] [--json FILE]
                                   [--hardware FILE] [--hw-name NAME] [--report]

Resolution:
    --json FILE     Compute ONLY the variant(s) defined in FILE (replaces the
                    project's built-in variants). Derived fields are auto-computed.
    --model NAME    Restrict the run to a single variant matched by name
                    (case-insensitive, exact). Applies to either the built-in
                    list or the --json list.
    --hardware FILE Analyze against ONLY the hardware defined in FILE (replaces
                    the project's built-in hardware list).
    --hw-name NAME  Restrict the run to a single hardware entry matched by name
                    (case-insensitive, exact). Applies to the built-in list or
                    the --hardware list.
    (neither)       Compute all built-in variants / hardware from configuration/.
    --report        Emit the concise markdown summary instead of full output.
"""

import argparse
import sys


def build_parser():
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai_model_calculator.py",
        description="AI Model Calculator — GPU memory, batch, training time, "
                    "and communication analysis for LLM training.",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Analyze only the variant with this name (case-insensitive, exact "
             "match). Without --json, matches the built-in variants.",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        dest="json_file",
        help="Path to a JSON file defining one or more model variants. When "
             "provided, ONLY these variants are analyzed (replaces built-ins).",
    )
    parser.add_argument(
        "--hardware",
        metavar="FILE",
        dest="hardware_file",
        help="Path to a JSON file defining one or more hardware configurations. "
             "When provided, ONLY these are analyzed (replaces built-ins).",
    )
    parser.add_argument(
        "--hw-name",
        metavar="NAME",
        dest="hw_name",
        help="Analyze only the hardware entry with this name (case-insensitive, "
             "exact match). Without --hardware, matches the built-in hardware.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Output a concise markdown-style summary report.",
    )
    return parser


def resolve_variants(args):
    """Resolve the list of variants to analyze from parsed args.

    Returns the list of variant dicts to feed into model_calculations.run.
    Exits the process with a non-zero code on user-input errors (bad JSON,
    unknown model name).
    """
    from utils.variant_loader import (
        load_variants_from_json,
        select_variant_by_name,
        VariantNotFoundError,
    )

    if args.json_file:
        from configuration.project_config import N_EXPERTS
        try:
            variants = load_variants_from_json(args.json_file, N_EXPERTS)
        except FileNotFoundError:
            print(f"Error: JSON file not found: {args.json_file}", file=sys.stderr)
            sys.exit(1)
        except (ValueError, KeyError) as e:
            print(f"Error: invalid model JSON ({args.json_file}): {e}", file=sys.stderr)
            sys.exit(1)
    else:
        from configuration.variants_config import VARIANTS
        variants = VARIANTS

    if args.model:
        try:
            variants = [select_variant_by_name(variants, args.model)]
        except VariantNotFoundError as e:
            print(f"Error: no variant named '{e.name}'.", file=sys.stderr)
            print("Available variants:", file=sys.stderr)
            for n in e.available:
                print(f"  - {n}", file=sys.stderr)
            sys.exit(1)

    return variants


def resolve_hardware(args):
    """Resolve the list of hardware configs to analyze from parsed args.

    Returns the list of hardware dicts to feed into model_calculations.run.
    Exits the process with a non-zero code on user-input errors (bad JSON,
    unknown hardware name).
    """
    from utils.hardware_loader import (
        load_hardware_from_json,
        select_hardware_by_name,
        HardwareNotFoundError,
    )

    if args.hardware_file:
        try:
            hardware = load_hardware_from_json(args.hardware_file)
        except FileNotFoundError:
            print(f"Error: hardware file not found: {args.hardware_file}", file=sys.stderr)
            sys.exit(1)
        except (ValueError, KeyError) as e:
            print(f"Error: invalid hardware JSON ({args.hardware_file}): {e}", file=sys.stderr)
            sys.exit(1)
    else:
        from configuration.hardware_config import HARDWARE
        hardware = HARDWARE

    if args.hw_name:
        try:
            hardware = [select_hardware_by_name(hardware, args.hw_name)]
        except HardwareNotFoundError as e:
            print(f"Error: no hardware named '{e.name}'.", file=sys.stderr)
            print("Available hardware:", file=sys.stderr)
            for n in e.available:
                print(f"  - {n}", file=sys.stderr)
            sys.exit(1)

    return hardware


def main(argv=None):
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    variants = resolve_variants(args)
    hardware = resolve_hardware(args)

    import model_calculations
    model_calculations.run(variants, hardware, report=args.report)


if __name__ == "__main__":
    main()
