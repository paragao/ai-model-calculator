"""
Phase 12: Scheduler and Batching Model for Inference.

Models how inference engines batch requests and manage concurrency.
Different engines have different batching strategies (continuous batching
in vLLM/SGLang, inflight batching in TRT-LLM).

Key concepts:
  1. Effective batch size: min(concurrency, max_num_seqs, max_concurrent_from_memory)
  2. Queue delay: when concurrency > effective_batch, requests wait
  3. Continuous batching: new requests can join mid-generation
  4. Prefix caching: shared prefix reduces prefill for similar prompts

Key formulas:
  effective_batch = min(concurrency, max_num_seqs, max_concurrent_mem)
  queue_depth = max(0, concurrency - effective_batch)
  avg_queue_delay_ms = queue_depth * avg_request_latency_ms / effective_batch
  request_latency_ms = ttft_ms + itl_ms * osl
  throughput = min(concurrency, effective_batch) * per_request_tok_s
"""

import math
import csv


def calculate_effective_batch(concurrency, max_num_seqs, max_concurrent_mem):
    """Effective batch size given engine and memory limits.

    The effective batch is the minimum of:
    - Requested concurrency (how many clients are sending requests)
    - Engine max_num_seqs (scheduler limit)
    - max_concurrent_mem (GPU memory limit from KV cache analysis)

    Formula:
        effective_batch = min(concurrency, max_num_seqs, max_concurrent_mem)

    Args:
        concurrency: Number of concurrent client requests.
        max_num_seqs: Engine scheduler limit on concurrent sequences.
        max_concurrent_mem: Maximum concurrent requests from GPU memory
                           (from Phase 7 KV cache analysis).

    Returns:
        Dict with:
        - effective_batch: The actual batch size that will be processed.
        - limited_by: Which constraint is binding ("concurrency",
          "engine_scheduler", or "gpu_memory").
        - concurrency: Original requested concurrency.
        - max_num_seqs: Engine scheduler limit.
        - max_concurrent_mem: GPU memory limit.
    """
    effective = min(concurrency, max_num_seqs, max_concurrent_mem)

    # Determine which constraint is binding
    if effective == max_concurrent_mem and max_concurrent_mem <= min(concurrency, max_num_seqs):
        limited_by = "gpu_memory"
    elif effective == max_num_seqs and max_num_seqs <= concurrency:
        limited_by = "engine_scheduler"
    else:
        limited_by = "concurrency"

    return {
        "effective_batch": max(0, effective),
        "limited_by": limited_by,
        "concurrency": concurrency,
        "max_num_seqs": max_num_seqs,
        "max_concurrent_mem": max_concurrent_mem,
    }


def estimate_queue_delay_ms(concurrency, effective_batch, avg_request_latency_ms):
    """Average queue wait time when the system is saturated.

    When concurrency exceeds effective_batch, excess requests queue up.
    Using Little's Law approximation: avg wait = queue_depth * service_time / servers.

    Formulas:
        queue_depth = max(0, concurrency - effective_batch)
        avg_queue_delay_ms = queue_depth * avg_request_latency_ms / effective_batch

    Args:
        concurrency: Number of concurrent client requests.
        effective_batch: Actual batch size being processed.
        avg_request_latency_ms: Average end-to-end request latency.

    Returns:
        Dict with:
        - queue_depth: Number of requests waiting in queue.
        - avg_queue_delay_ms: Average wait time in milliseconds.
        - is_saturated: True if concurrency > effective_batch.
    """
    queue_depth = max(0, concurrency - effective_batch)
    is_saturated = queue_depth > 0

    if is_saturated and effective_batch > 0:
        # Little's Law approximation:
        # avg wait = queue_depth * service_time / num_servers
        avg_queue_delay_ms = queue_depth * avg_request_latency_ms / effective_batch
    else:
        avg_queue_delay_ms = 0.0

    return {
        "queue_depth": queue_depth,
        "avg_queue_delay_ms": round(avg_queue_delay_ms, 2),
        "is_saturated": is_saturated,
    }


def estimate_request_latency_ms(ttft_ms, itl_ms, osl):
    """End-to-end latency for a single request.

    Formula:
        request_latency_ms = ttft_ms + itl_ms * osl

    TTFT is the time-to-first-token (prefill phase).
    ITL is the inter-token latency (decode phase).
    OSL is the output sequence length (tokens to generate).

    Args:
        ttft_ms: Time to first token in milliseconds.
        itl_ms: Inter-token latency in milliseconds.
        osl: Output sequence length (number of tokens to generate).

    Returns:
        Dict with:
        - request_latency_ms: Total end-to-end latency.
        - ttft_ms: Prefill component.
        - decode_latency_ms: Decode component (itl_ms * osl).
        - osl: Output sequence length.
        - ttft_fraction: Fraction of latency from prefill.
    """
    decode_latency_ms = itl_ms * osl
    request_latency_ms = ttft_ms + decode_latency_ms

    if request_latency_ms > 0:
        ttft_fraction = ttft_ms / request_latency_ms
    else:
        ttft_fraction = 0.0

    return {
        "request_latency_ms": round(request_latency_ms, 2),
        "ttft_ms": ttft_ms,
        "decode_latency_ms": round(decode_latency_ms, 2),
        "osl": osl,
        "ttft_fraction": round(ttft_fraction, 4),
    }


def calculate_throughput_at_concurrency(concurrency, effective_batch,
                                        single_req_tok_s, batched_tok_s):
    """Actual throughput at given concurrency considering batching.

    When concurrency <= effective_batch, all requests are processed
    in parallel with batched throughput. When concurrency > effective_batch,
    throughput is capped at the batched rate for effective_batch.

    Formulas:
        if concurrency <= 1:
            throughput = single_req_tok_s
        elif concurrency <= effective_batch:
            # Linear interpolation approximation
            throughput = batched_tok_s * (concurrency / effective_batch)
        else:
            throughput = batched_tok_s  # capped at batch capacity

    Args:
        concurrency: Number of concurrent client requests.
        effective_batch: Actual batch size that can be processed.
        single_req_tok_s: Throughput for a single request (tok/s).
        batched_tok_s: Throughput at full effective_batch (tok/s).

    Returns:
        Dict with:
        - total_tok_s: Aggregate throughput (tokens/sec).
        - per_request_tok_s: Per-request throughput.
        - is_batched: True if batch_size > 1.
        - utilization: Fraction of batch capacity used.
    """
    if effective_batch <= 0:
        return {
            "total_tok_s": 0.0,
            "per_request_tok_s": 0.0,
            "is_batched": False,
            "utilization": 0.0,
        }

    if concurrency <= 1:
        total_tok_s = single_req_tok_s
    elif concurrency <= effective_batch:
        # Throughput scales with concurrency up to effective_batch
        # Linear interpolation between single-request and full-batch
        total_tok_s = batched_tok_s * (concurrency / effective_batch)
    else:
        # Capped at effective_batch throughput
        total_tok_s = batched_tok_s

    active_batch = min(concurrency, effective_batch)
    per_request_tok_s = total_tok_s / active_batch if active_batch > 0 else 0.0
    utilization = active_batch / effective_batch if effective_batch > 0 else 0.0

    return {
        "total_tok_s": round(total_tok_s, 2),
        "per_request_tok_s": round(per_request_tok_s, 2),
        "is_batched": active_batch > 1,
        "utilization": round(utilization, 4),
    }


def analyze_scheduler(concurrency_levels, max_num_seqs, max_concurrent_mem,
                      ttft_ms, itl_ms, osl, single_req_tok_s,
                      batched_throughputs):
    """Full scheduler analysis sweeping concurrency levels.

    For each concurrency level, calculates effective batch, queue delay,
    request latency, and throughput.

    Args:
        concurrency_levels: List of concurrency values to sweep.
        max_num_seqs: Engine scheduler limit.
        max_concurrent_mem: GPU memory limit on concurrent requests.
        ttft_ms: Time to first token (ms).
        itl_ms: Inter-token latency (ms).
        osl: Output sequence length.
        single_req_tok_s: Single-request throughput (tok/s).
        batched_throughputs: Dict mapping batch_size -> total_tok_s,
                            or a single float for the max-batch throughput.

    Returns:
        Dict with:
        - max_num_seqs: Engine limit.
        - max_concurrent_mem: Memory limit.
        - osl: Output sequence length.
        - base_request_latency_ms: Base request latency (no queueing).
        - concurrency_results: List of per-concurrency result dicts.
    """
    # Base request latency (no queue delay)
    req_lat = estimate_request_latency_ms(ttft_ms, itl_ms, osl)
    base_latency_ms = req_lat["request_latency_ms"]

    # Resolve batched throughput: if dict, use max available batch;
    # if float/int, treat as the throughput at full effective_batch.
    if isinstance(batched_throughputs, dict):
        # Use the throughput at the highest batch size available
        max_batch_key = max(batched_throughputs.keys()) if batched_throughputs else 1
        full_batch_tok_s = batched_throughputs[max_batch_key]
    else:
        full_batch_tok_s = float(batched_throughputs)

    concurrency_results = []
    for conc in concurrency_levels:
        # Effective batch size
        eb = calculate_effective_batch(conc, max_num_seqs, max_concurrent_mem)
        effective_batch = eb["effective_batch"]

        # Throughput
        tp = calculate_throughput_at_concurrency(
            conc, effective_batch, single_req_tok_s, full_batch_tok_s
        )

        # Queue delay
        qd = estimate_queue_delay_ms(conc, effective_batch, base_latency_ms)

        # Effective per-request latency including queue delay
        effective_latency_ms = base_latency_ms + qd["avg_queue_delay_ms"]

        concurrency_results.append({
            "concurrency": conc,
            "effective_batch": effective_batch,
            "limited_by": eb["limited_by"],
            "total_tok_s": tp["total_tok_s"],
            "per_request_tok_s": tp["per_request_tok_s"],
            "utilization": tp["utilization"],
            "queue_depth": qd["queue_depth"],
            "avg_queue_delay_ms": qd["avg_queue_delay_ms"],
            "is_saturated": qd["is_saturated"],
            "effective_latency_ms": round(effective_latency_ms, 2),
            "base_latency_ms": base_latency_ms,
        })

    return {
        "max_num_seqs": max_num_seqs,
        "max_concurrent_mem": max_concurrent_mem,
        "osl": osl,
        "base_request_latency_ms": base_latency_ms,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "ttft_fraction": req_lat["ttft_fraction"],
        "concurrency_results": concurrency_results,
    }


def generate_scheduler_recommendations(all_results):
    """Generate recommendations for concurrency tuning.

    Args:
        all_results: Dict from analyze_scheduler.

    Returns:
        List of recommendation strings.
    """
    recs = []

    concurrency_results = all_results.get("concurrency_results", [])
    if not concurrency_results:
        return ["No scheduler analysis results available."]

    max_num_seqs = all_results.get("max_num_seqs", 256)
    max_concurrent_mem = all_results.get("max_concurrent_mem", 0)
    base_latency = all_results.get("base_request_latency_ms", 0)

    # Check if GPU memory is the bottleneck
    mem_limited = [r for r in concurrency_results if r["limited_by"] == "gpu_memory"]
    if mem_limited:
        first_mem = mem_limited[0]
        recs.append(
            f"Memory-limited: GPU memory caps effective batch at "
            f"{first_mem['effective_batch']} requests. "
            f"Consider FP8 KV cache, more GPU memory, or shorter sequences "
            f"to increase concurrency."
        )

    # Check if engine scheduler is the bottleneck
    sched_limited = [r for r in concurrency_results if r["limited_by"] == "engine_scheduler"]
    if sched_limited:
        recs.append(
            f"Engine-limited: Scheduler max_num_seqs={max_num_seqs} caps "
            f"effective batch. Consider increasing max_num_seqs if memory allows."
        )

    # Find saturation point
    saturated = [r for r in concurrency_results if r["is_saturated"]]
    unsaturated = [r for r in concurrency_results if not r["is_saturated"]]

    if saturated:
        first_sat = saturated[0]
        recs.append(
            f"Saturation: System saturates at concurrency={first_sat['concurrency']} "
            f"(effective_batch={first_sat['effective_batch']}). "
            f"Queue delay={first_sat['avg_queue_delay_ms']:.0f} ms."
        )

    # Queue delay warnings
    high_delay = [r for r in concurrency_results if r["avg_queue_delay_ms"] > base_latency]
    if high_delay:
        worst = max(high_delay, key=lambda x: x["avg_queue_delay_ms"])
        recs.append(
            f"High queue delay: At concurrency={worst['concurrency']}, "
            f"queue delay ({worst['avg_queue_delay_ms']:.0f} ms) exceeds "
            f"base request latency ({base_latency:.0f} ms). "
            f"Users experience > 2x expected latency."
        )

    # Throughput sweet spot
    if concurrency_results:
        best_throughput = max(concurrency_results, key=lambda x: x["total_tok_s"])
        if best_throughput["total_tok_s"] > 0:
            recs.append(
                f"Peak throughput: {best_throughput['total_tok_s']:.0f} tok/s "
                f"at concurrency={best_throughput['concurrency']} "
                f"(effective_batch={best_throughput['effective_batch']})."
            )

    # Latency-throughput tradeoff
    if unsaturated and len(unsaturated) >= 2:
        low_conc = unsaturated[0]
        high_conc = unsaturated[-1]
        if high_conc["total_tok_s"] > 2 * low_conc["total_tok_s"]:
            recs.append(
                f"Throughput scales: {low_conc['total_tok_s']:.0f} tok/s at "
                f"concurrency={low_conc['concurrency']} vs "
                f"{high_conc['total_tok_s']:.0f} tok/s at "
                f"concurrency={high_conc['concurrency']} — "
                f"increase concurrency for better GPU utilization."
            )

    return recs


def export_scheduler_results_csv(all_results, filename):
    """Export scheduler analysis results to CSV.

    Args:
        all_results: Dict from analyze_scheduler.
        filename: Output CSV file path.
    """
    rows = []
    base_info = {
        "max_num_seqs": all_results.get("max_num_seqs", ""),
        "max_concurrent_mem": all_results.get("max_concurrent_mem", ""),
        "osl": all_results.get("osl", ""),
        "ttft_ms": all_results.get("ttft_ms", ""),
        "itl_ms": all_results.get("itl_ms", ""),
    }

    for r in all_results.get("concurrency_results", []):
        row = dict(base_info)
        row.update({
            "concurrency": r["concurrency"],
            "effective_batch": r["effective_batch"],
            "limited_by": r["limited_by"],
            "total_tok_s": r["total_tok_s"],
            "per_request_tok_s": r["per_request_tok_s"],
            "utilization": r["utilization"],
            "queue_depth": r["queue_depth"],
            "avg_queue_delay_ms": r["avg_queue_delay_ms"],
            "is_saturated": r["is_saturated"],
            "effective_latency_ms": r["effective_latency_ms"],
            "base_latency_ms": r["base_latency_ms"],
        })
        rows.append(row)

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
