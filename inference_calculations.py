#!/usr/bin/python3
"""
LLM Inference Calculator — Main Orchestrator

Analyzes GPU memory, throughput, latency, communication overhead, and cost
efficiency for LLM inference serving workloads on AWS hardware.

Phases:
  7. KV Cache & Model Memory
  8. Prefill (TTFT) Analysis
  9. Decode (Throughput/ITL) Analysis
 10. TP Communication Overhead
 11. EP Communication (MoE only)
 12. Scheduler & Batching
 13. Cost Efficiency

Usage:
  python3 inference_calculations.py [--report] [--engine vllm|sglang|trtllm|all]
  python3 inference_calculations.py --variant "Gemma-4-31B-IT" --hw "1 g7e-12xl"
"""

import argparse
import sys
import os

parser = argparse.ArgumentParser(description='LLM Inference Calculator')
parser.add_argument('--report', action='store_true',
                    help='Output a concise markdown-style summary report')
parser.add_argument('--engine', type=str, default=None,
                    choices=['vllm', 'sglang', 'trtllm', 'all'],
                    help='Override inference engine (default: from config)')
parser.add_argument('--variant', type=str, default=None,
                    help='Filter to a single model variant by name')
parser.add_argument('--hw', type=str, default=None,
                    help='Filter to a single hardware config by name')
parser.add_argument('--tp', type=int, default=None,
                    help='Override tensor parallelism degree (default: from config)')
parser.add_argument('--ep', type=int, default=None,
                    help='Override expert parallelism degree (default: from config)')
parser.add_argument('--gpu-util', type=float, default=None,
                    help='Override GPU memory utilization (0.0-1.0, default: from config)')
args = parser.parse_args()

# ============================================================================
# Imports
# ============================================================================

from configuration.variants_config import VARIANTS
from configuration.hardware_config import INFERENCE_HARDWARE
from configuration.inference_config import (
    INPUT_SEQ_LENS, OUTPUT_SEQ_LEN, CONCURRENCY_LEVELS,
    ENGINE, QUANTIZATION, KV_CACHE_DTYPE,
    MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION, ENGINE_OVERHEAD_GB,
    INFERENCE_TP, INFERENCE_PP, INFERENCE_EP,
    ENGINE_MFU, QUANT_BYTES, KV_DTYPE_BYTES,
    INSTANCE_PRICING,
)

from utils.phase7_kv_cache import (
    analyze_kv_cache, generate_kv_recommendations,
    export_kv_results_csv, export_kv_results_json,
)
from utils.phase8_prefill import (
    analyze_prefill, generate_prefill_recommendations,
    export_prefill_results_csv,
)
from utils.phase9_decode import (
    analyze_decode, generate_decode_recommendations,
    export_decode_results_csv,
)
from utils.phase10_tp_comm import (
    analyze_tp_comm, generate_tp_recommendations,
    export_tp_results_csv,
)
from utils.phase11_ep_comm import (
    analyze_ep_comm, generate_ep_recommendations,
    export_ep_results_csv,
)
from utils.phase12_scheduler import (
    analyze_scheduler, generate_scheduler_recommendations,
    export_scheduler_results_csv,
)
from utils.phase13_cost import (
    analyze_cost, generate_cost_recommendations,
    export_cost_results_csv,
)

from utils.formatting_utils import USE_COLOR, color_text


# ============================================================================
# Resolve configuration
# ============================================================================

engine = args.engine or ENGINE
tp = args.tp if args.tp is not None else INFERENCE_TP
ep = args.ep if args.ep is not None else INFERENCE_EP
gpu_util = args.gpu_util if args.gpu_util is not None else GPU_MEMORY_UTILIZATION
quant_bytes = QUANT_BYTES[QUANTIZATION]
kv_bytes = KV_DTYPE_BYTES[KV_CACHE_DTYPE]
if kv_bytes is None:
    kv_bytes = quant_bytes  # auto = same as quantization

# Engine MFU parameters
if engine == "all":
    engines_to_run = ["vllm", "sglang", "trtllm"]
else:
    engines_to_run = [engine]

# Filter variants if specified
if args.variant:
    variants = [v for v in VARIANTS if v["name"] == args.variant]
    if not variants:
        print(f"Error: variant '{args.variant}' not found. Available:")
        for v in VARIANTS:
            print(f"  {v['name']}")
        sys.exit(1)
else:
    variants = list(VARIANTS)

# Filter hardware if specified
if args.hw:
    hw_list = [h for h in INFERENCE_HARDWARE if h["name"] == args.hw]
    if not hw_list:
        print(f"Error: hardware '{args.hw}' not found. Available:")
        for h in INFERENCE_HARDWARE:
            print(f"  {h['name']}")
        sys.exit(1)
else:
    hw_list = list(INFERENCE_HARDWARE)


# ============================================================================
# Report generation
# ============================================================================

def generate_report():
    """Generate a concise markdown-style inference analysis report."""
    lines = []
    lines.append("# LLM Inference Calculator -- Summary Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Variants**: {', '.join(v['name'] for v in variants)}")
    lines.append(f"- **Hardware**: {', '.join(h['name'] for h in hw_list)}")
    lines.append(f"- **Engine(s)**: {', '.join(engines_to_run)}")
    lines.append(f"- **Quantization**: {QUANTIZATION} ({quant_bytes} bytes/param)")
    lines.append(f"- **KV Cache**: {KV_CACHE_DTYPE} ({kv_bytes} bytes/element)")
    lines.append(f"- **TP**: {tp}, **EP**: {ep}")
    lines.append(f"- **ISL**: {INPUT_SEQ_LENS}")
    lines.append(f"- **OSL**: {OUTPUT_SEQ_LEN}")
    lines.append(f"- **Concurrency**: {CONCURRENCY_LEVELS}")
    lines.append("")

    for eng in engines_to_run:
        emfu = ENGINE_MFU[eng]
        lines.append(f"## Engine: {eng}")
        lines.append("")

        for hw in hw_list:
            lines.append(f"### {hw['name']} ({hw['instance_type']}, {hw['mem_gb']} GB/GPU)")
            lines.append("")

            # Results table header
            lines.append("| Variant | ISL | Max Conc | TTFT (ms) | Bottleneck | Decode tok/s @C=32 | ITL (ms) | Cost $/M tok |")
            lines.append("|---------|-----|----------|-----------|------------|---------------------|----------|--------------|")

            for v in variants:
                # KV cache analysis
                kv = analyze_kv_cache(v, hw, tp, quant_bytes, kv_bytes,
                                      INPUT_SEQ_LENS, OUTPUT_SEQ_LEN,
                                      gpu_util, ENGINE_OVERHEAD_GB)

                # Prefill analysis
                pf = analyze_prefill(v, hw, tp, quant_bytes, INPUT_SEQ_LENS, emfu)

                # TP comm overhead for decode
                rpt_tp_comm_ms = 0.0
                if tp > 1:
                    from utils.phase10_tp_comm import estimate_tp_comm_time_ms
                    rpt_tp_info = estimate_tp_comm_time_ms(v, hw, tp, 1, quant_bytes)
                    rpt_tp_comm_ms = rpt_tp_info["comm_time_ms"] * (1.0 - emfu.get("tp_comm_overlap", 0.0))

                for i, isl in enumerate(INPUT_SEQ_LENS):
                    isl_mem = kv["isl_results"][i]
                    isl_pf = pf["isl_results"][i]

                    if not isl_mem["fits"]:
                        lines.append(
                            f"| {v['name']} | {isl//1000}K | OOM | -- | -- | -- | -- | -- |"
                        )
                        continue

                    max_conc = isl_mem["max_concurrent"]
                    ttft = isl_pf.get("ttft_adjusted_ms", 0)
                    bottleneck = isl_pf.get("bottleneck", "?")

                    # Decode at concurrency=32
                    eff_batch = min(32, max_conc, MAX_NUM_SEQS)
                    if eff_batch > 0:
                        from utils.phase9_decode import calculate_batched_throughput
                        dec = calculate_batched_throughput(
                            v, hw, tp, quant_bytes, kv_bytes,
                            isl, OUTPUT_SEQ_LEN, eff_batch,
                            emfu["decode_bw_eff"], emfu["batching_eff"],
                            rpt_tp_comm_ms
                        )
                        tok_s = dec["total_tok_s"]
                        itl = dec["itl_ms"]
                    else:
                        tok_s = 0
                        itl = 0

                    # Cost
                    hourly = INSTANCE_PRICING.get(hw["instance_type"], 0)
                    cost_m = hourly / (tok_s * 3.6) if tok_s > 0 else float('inf')

                    lines.append(
                        f"| {v['name']} | {isl//1000}K | {max_conc} | "
                        f"{ttft:,.0f} | {bottleneck} | {tok_s:,.1f} | "
                        f"{itl:.1f} | ${cost_m:,.2f} |"
                    )

            lines.append("")

    lines.append("## Disclaimer")
    lines.append("")
    lines.append(
        "These are theoretical estimates based on roofline analysis. "
        "Actual performance depends on engine implementation, attention backend, "
        "CUDA graph optimizations, memory allocator behavior, and workload patterns. "
        "Use as directional guidance for hardware selection and capacity planning."
    )
    lines.append("")

    return "\n".join(lines)


if args.report:
    print(generate_report())
    sys.exit(0)


# ============================================================================
# INTERACTIVE MODE: Run all phases with detailed output
# ============================================================================

for eng in engines_to_run:
    emfu = ENGINE_MFU[eng]

    print(f"\n{'=' * 140}")
    print(f"LLM INFERENCE CALCULATOR -- Engine: {eng.upper()}")
    print(f"Quantization: {QUANTIZATION} | KV Cache: {KV_CACHE_DTYPE} | TP={tp} | EP={ep}")
    print(f"ISL: {INPUT_SEQ_LENS} | OSL: {OUTPUT_SEQ_LEN} | Concurrency: {CONCURRENCY_LEVELS}")
    print(f"{'=' * 140}")

    # Collect all results for export
    all_kv = {}
    all_prefill = {}
    all_decode = {}
    all_tp = {}
    all_ep = {}
    all_scheduler = {}
    all_cost = {}

    for hw in hw_list:
        print(f"\n{'#' * 140}")
        print(f"# {hw['name']} ({hw['instance_type']}, {hw['gpus']} GPUs, "
              f"{hw['mem_gb']} GB/GPU, HBM BW={hw['hbm_bw_gbps']} GB/s)")
        print(f"{'#' * 140}")

        hw_kv_results = []
        hw_prefill_results = []

        for v in variants:
            # ==================================================================
            # PHASE 7: KV Cache & Memory Analysis
            # ==================================================================
            kv = analyze_kv_cache(v, hw, tp, quant_bytes, kv_bytes,
                                  INPUT_SEQ_LENS, OUTPUT_SEQ_LEN,
                                  gpu_util, ENGINE_OVERHEAD_GB)
            hw_kv_results.append(kv)

            print(f"\n  {v['name']} (TP={tp}, model={kv['model_mem_gb']:.1f} GB/GPU, "
                  f"KV/token={kv['kv_per_token_kb']:.1f} KB)")

            # Phase 7 table
            print(f"    {'ISL':>6} {'KV/req':>10} {'Avail KV':>10} {'Max Conc':>10} {'Status':>12}")
            print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

            for isl_r in kv["isl_results"]:
                if isl_r["fits"]:
                    status = "OK" if isl_r["max_concurrent"] >= 8 else "LOW"
                    print(f"    {isl_r['isl']:>6} {isl_r['kv_per_request_gb']:>9.2f}G "
                          f"{isl_r['available_for_kv_gb']:>9.1f}G {isl_r['max_concurrent']:>10} "
                          f"{status:>12}")
                else:
                    print(f"    {isl_r['isl']:>6} {'--':>10} {'--':>10} {'OOM':>10} "
                          f"{color_text('OOM', 'red'):>12}")

            # ==================================================================
            # PHASE 8: Prefill (TTFT) Analysis
            # ==================================================================
            pf = analyze_prefill(v, hw, tp, quant_bytes, INPUT_SEQ_LENS, emfu)
            hw_prefill_results.append(pf)

            print(f"\n    Prefill (TTFT):")
            print(f"    {'ISL':>6} {'FLOPs':>12} {'TTFT (ms)':>12} {'Bottleneck':>12}")
            print(f"    {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

            for isl_pf in pf["isl_results"]:
                flops_t = isl_pf["prefill_flops"] / 1e12
                ttft = isl_pf.get("ttft_adjusted_ms", 0)
                bn = isl_pf.get("bottleneck", "?")
                print(f"    {isl_pf['isl']:>6} {flops_t:>10.1f}TF {ttft:>10,.0f}ms {bn:>12}")

            # ==================================================================
            # PHASE 9: Decode (Throughput / ITL) Analysis
            # ==================================================================
            # Calculate TP communication overhead per decode step (batch=1)
            # This is added to each decode step's latency to account for
            # inter-GPU communication (significant on PCIe, small on NVLink)
            tp_comm_ms = 0.0
            if tp > 1:
                from utils.phase10_tp_comm import estimate_tp_comm_time_ms
                tp_comm_info = estimate_tp_comm_time_ms(v, hw, tp, 1, quant_bytes)
                tp_comm_ms = tp_comm_info["comm_time_ms"]
                # Apply overlap fraction from engine config
                overlap = emfu.get("tp_comm_overlap", 0.0)
                tp_comm_ms = tp_comm_ms * (1.0 - overlap)

            def max_conc_fn(isl, osl):
                for r in kv["isl_results"]:
                    if r["isl"] == isl:
                        return r["max_concurrent"] if r["fits"] else 0
                return 0

            dec = analyze_decode(v, hw, tp, quant_bytes, kv_bytes,
                                 INPUT_SEQ_LENS, OUTPUT_SEQ_LEN,
                                 CONCURRENCY_LEVELS, emfu, max_conc_fn,
                                 tp_comm_ms)

            print(f"\n    Decode (Throughput / ITL):")
            print(f"    {'ISL':>6} {'Conc':>6} {'Eff Batch':>10} {'tok/s':>10} {'ITL (ms)':>10} {'Bottleneck':>12}")
            print(f"    {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

            for isl_dec in dec.get("isl_results", []):
                for cr in isl_dec.get("concurrency_results", []):
                    print(f"    {isl_dec['isl']:>6} {cr['requested_concurrency']:>6} "
                          f"{cr['effective_batch']:>10} {cr['total_tok_s']:>10.1f} "
                          f"{cr['itl_ms']:>10.1f} {cr.get('bottleneck', '?'):>12}")

            # ==================================================================
            # PHASE 10: TP Communication (if TP > 1)
            # ==================================================================
            if tp > 1:
                tp_analysis = analyze_tp_comm(v, [hw], [tp], [1, 32, 64],
                                              quant_bytes, quant_bytes, emfu)

                print(f"\n    TP={tp} Communication:")
                for entry in tp_analysis.get("results", {}).get(hw["name"], []):
                    eff = entry.get("efficiency", 0)
                    print(f"      batch={entry['batch_size']:>3}: "
                          f"comm={entry['comm_time_ms']:.2f}ms, "
                          f"efficiency={eff:.1%}, "
                          f"interconnect={entry.get('interconnect_type', '?')}")

            # ==================================================================
            # PHASE 11: EP Communication (MoE only)
            # ==================================================================
            if v.get("expert_ffn", 0) > 0 and ep > 1:
                ep_analysis = analyze_ep_comm(v, [hw], [ep], [1, 32, 64],
                                              quant_bytes)

                if not ep_analysis.get("skipped", False):
                    print(f"\n    EP={ep} Communication (MoE):")
                    for entry in ep_analysis.get("results", {}).get(hw["name"], []):
                        print(f"      batch={entry['batch_size']:>3}: "
                              f"dispatch+combine={entry['ep_latency_ms']:.2f}ms")

            # ==================================================================
            # PHASE 12: Scheduler Analysis
            # ==================================================================
            # Use ISL=14K (first ISL) for scheduler analysis
            first_isl = INPUT_SEQ_LENS[0]
            first_kv = kv["isl_results"][0]
            first_pf = pf["isl_results"][0]

            if first_kv["fits"]:
                max_mem_conc = first_kv["max_concurrent"]
                ttft_ms = first_pf.get("ttft_adjusted_ms", 1000)

                # Get batched throughputs for each concurrency
                conc_throughputs = []
                for conc in CONCURRENCY_LEVELS:
                    eff = min(conc, max_mem_conc, MAX_NUM_SEQS)
                    if eff > 0:
                        from utils.phase9_decode import calculate_batched_throughput
                        bt = calculate_batched_throughput(
                            v, hw, tp, quant_bytes, kv_bytes,
                            first_isl, OUTPUT_SEQ_LEN, eff,
                            emfu["decode_bw_eff"], emfu["batching_eff"],
                            tp_comm_ms
                        )
                        conc_throughputs.append((conc, bt["total_tok_s"]))
                    else:
                        conc_throughputs.append((conc, 0))

                # Single request throughput
                from utils.phase9_decode import calculate_single_request_throughput
                single = calculate_single_request_throughput(
                    v, hw, tp, quant_bytes, kv_bytes,
                    first_isl, OUTPUT_SEQ_LEN, emfu["decode_bw_eff"],
                    tp_comm_ms
                )
                single_tok_s = single["avg_tok_s"]
                itl_ms = single["itl_ms"]

                sched = analyze_scheduler(
                    CONCURRENCY_LEVELS, MAX_NUM_SEQS, max_mem_conc,
                    ttft_ms, itl_ms, OUTPUT_SEQ_LEN,
                    single_tok_s,
                    dict(conc_throughputs),  # convert list of tuples to dict
                )

                print(f"\n    Scheduler (ISL={first_isl//1000}K):")
                print(f"    {'Conc':>6} {'Eff Batch':>10} {'Queue (ms)':>12} "
                      f"{'E2E (ms)':>12} {'tok/s':>10} {'Saturated':>10}")
                print(f"    {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

                for entry in sched.get("concurrency_results", []):
                    sat = "YES" if entry.get("is_saturated", False) else "no"
                    print(f"    {entry['concurrency']:>6} {entry['effective_batch']:>10} "
                          f"{entry.get('avg_queue_delay_ms', 0):>12.0f} "
                          f"{entry.get('effective_latency_ms', 0):>12.0f} "
                          f"{entry.get('total_tok_s', 0):>10.1f} {sat:>10}")

            # ==================================================================
            # PHASE 13: Cost Analysis
            # ==================================================================
            hourly = INSTANCE_PRICING.get(hw["instance_type"], 0)
            if hourly > 0 and first_kv["fits"]:
                cost = analyze_cost(
                    hw["name"], hw["instance_type"], tp, INSTANCE_PRICING,
                    conc_throughputs
                )

                print(f"\n    Cost ({hw['instance_type']} @ ${hourly:.2f}/hr):")
                print(f"    {'Conc':>6} {'tok/s':>10} {'tok/$':>12} {'$/M tok':>12}")
                print(f"    {'-'*6} {'-'*10} {'-'*12} {'-'*12}")

                for entry in cost.get("concurrency_results", []):
                    print(f"    {entry['concurrency']:>6} {entry['total_tok_s']:>10.1f} "
                          f"{entry['tokens_per_dollar']:>12,.0f} "
                          f"${entry['cost_per_million_tokens']:>11.2f}")

        # Store results for export
        all_kv[hw["name"]] = hw_kv_results
        all_prefill[hw["name"]] = hw_prefill_results

    # ======================================================================
    # Export
    # ======================================================================
    prefix = f"inference_{eng}_"
    export_kv_results_csv(all_kv, f"{prefix}phase7_kv_cache.csv")
    export_kv_results_json(all_kv, f"{prefix}phase7_kv_cache.json")
    export_prefill_results_csv(all_prefill, f"{prefix}phase8_prefill.csv")

    print(f"\n{'=' * 140}")
    print(f"EXPORT SUMMARY ({eng})")
    print(f"{'=' * 140}")
    print(f"  phase7_kv_cache:  {prefix}phase7_kv_cache.csv / .json")
    print(f"  phase8_prefill:   {prefix}phase8_prefill.csv")

print(f"\nDone.")
