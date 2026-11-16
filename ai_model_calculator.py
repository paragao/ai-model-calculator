#!/usr/bin/python3
"""Command-line entry point for the AI Model Calculator.

Parses arguments, resolves which model variant(s) to analyze, and delegates the
actual six-phase computation to ``model_calculations.run``.

Usage:
    python3 ai_model_calculator.py [--model NAME] [--json FILE]
                                   [--hf-config FILE [--params-total-b B]]
                                   [--hardware FILE] [--hw-name NAME] [--report]

Resolution:
    --json FILE     Compute ONLY the variant(s) defined in FILE (replaces the
                    project's built-in variants). Derived fields are auto-computed.
    --hf-config FILE
                    Compute ONLY the model described by a Hugging Face config.json.
                    Parameters, active parameters, shared experts, MLA and
                    linear-attention layers are derived from the config.
                    --params-total-b B gives the exact total (safetensors index).
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
    --estimate      Phase 9: plan search, estimated MFU, throughput, days and cost
                    (see --instance, --num-gpus, --tokens-b, --seq-len, --precision,
                    --tp/--pp/--ep, --vp, --gbs, --mfu, --target-days, --format).
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
        "--hf-config",
        metavar="FILE",
        dest="hf_config",
        help="Path to a Hugging Face config.json. The variant (total and active "
             "parameters, shared experts, MLA / linear attention) is derived from "
             "it, and ONLY this variant is analyzed. Not combinable with --json.",
    )
    parser.add_argument(
        "--params-total-b",
        metavar="B",
        type=float,
        dest="params_total_b",
        help="With --hf-config: exact total parameters in billions, e.g. from the "
             "safetensors index. Wins over the count derived from the shape.",
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
    est = parser.add_argument_group(
        "estimate mode (Phase 9)",
        "Plan search, estimated MFU, throughput, days and cost for one instance type "
        "and GPU count (docs/CALCULATOR-MFU.md). Replaces the fixed MFU constant.",
    )
    est.add_argument("--estimate", action="store_true",
                     help="Run Phase 9 instead of the six-phase analysis.")
    est.add_argument("--instance", default="p5en.48xlarge", metavar="NAME",
                     help="Instance type or GPU name: p5, p5e, p5en, p6-b200, p6-b300, p4d, p4de, "
                          "H100, H200, B200, B300, A100 (default p5en.48xlarge).")
    est.add_argument("--num-gpus", default="64", metavar="N|auto",
                     help="GPU count, or 'auto' for the smallest whole-node count that fits "
                          "(and meets --target-days). Default 64.")
    est.add_argument("--tokens-b", type=float, default=1000.0, metavar="B",
                     help="Training tokens in billions (default 1000).")
    est.add_argument("--seq-len", type=int, default=4096, help="Sequence length (default 4096).")
    est.add_argument("--precision", default="bf16", choices=["bf16", "fp8"], help="Default bf16.")
    est.add_argument("--tp", type=int, default=1, help="Fix tensor parallel (1 = search).")
    est.add_argument("--pp", type=int, default=1, help="Fix pipeline parallel (1 = search).")
    est.add_argument("--ep", type=int, default=1, help="Fix expert parallel (1 = search).")
    est.add_argument("--vp", type=int, default=1, help="Virtual pipeline stages (default 1).")
    est.add_argument("--gbs", type=int, default=None, metavar="SEQS",
                     help="Global batch in sequences (default: 16M tokens per step).")
    est.add_argument("--mfu", type=float, default=None,
                     help="Override the estimated MFU (fraction, e.g. 0.4).")
    est.add_argument("--target-days", type=float, default=None,
                     help="With --num-gpus auto: stop at the first GPU count that trains in this many days.")
    est.add_argument("--format", default="text", choices=["text", "json"],
                     help="Estimate output format (default text).")
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

    if args.hf_config and args.json_file:
        print("Error: --hf-config and --json cannot be combined.", file=sys.stderr)
        sys.exit(1)

    if args.hf_config:
        import os
        from utils.architecture_loader import load_variant_from_hf_config
        name = os.path.splitext(os.path.basename(args.hf_config))[0]
        try:
            variants = [load_variant_from_hf_config(
                args.hf_config, name=args.model or name,
                params_total_b=args.params_total_b)]
        except FileNotFoundError:
            print(f"Error: config file not found: {args.hf_config}", file=sys.stderr)
            sys.exit(1)
        except (ValueError, KeyError) as e:
            print(f"Error: invalid config.json ({args.hf_config}): {e}", file=sys.stderr)
            sys.exit(1)
        return variants

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


def run_estimate_mode(args, variants):
    """Phase 9 for each selected variant: plan search, estimated MFU, days and cost."""
    import json
    from utils.phase9_mfu_estimate import run_estimate, format_estimate

    num_gpus = "auto" if str(args.num_gpus).lower() == "auto" else int(args.num_gpus)
    config = {
        "hardware": args.instance, "num_gpus": num_gpus, "precision": args.precision,
        "sequence_length": args.seq_len, "total_tokens_b": args.tokens_b,
        "tensor_parallel": args.tp, "pipeline_parallel": args.pp, "expert_parallel": args.ep,
        "virtual_pipeline_parallel": args.vp, "global_batch_size": args.gbs,
        "mfu": args.mfu, "target_days": args.target_days,
    }
    results = []
    for v in variants:
        try:
            results.append(run_estimate(config, v))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    if args.format == "json":
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        print("\n\n".join(format_estimate(r) for r in results))
    if any("error" in r for r in results):
        sys.exit(1)


def main(argv=None):
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    variants = resolve_variants(args)

    if args.estimate:
        run_estimate_mode(args, variants)
        return

    hardware = resolve_hardware(args)

    import model_calculations
    model_calculations.run(variants, hardware, report=args.report)


if __name__ == "__main__":
    main()
