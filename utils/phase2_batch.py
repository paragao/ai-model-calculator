"""
Phase 2: Batch Configuration Analysis functions.

Analyzes batch size configurations (micro batch size and gradient accumulation steps)
to determine optimal training configurations that match target batch sizes.
"""

# Import configurations
from configuration.project_config import *
from configuration.advanced_config import *

# Import formatting utilities
from .formatting_utils import color_text


def assess_training_steps(steps, tokens_per_batch):
    """
    Assess whether the number of training steps is reasonable.

    Args:
        steps: Number of training steps
        tokens_per_batch: Tokens per batch

    Returns:
        tuple: (assessment_string, priority_level)
            priority_level: 0=optimal, 1=good, 2=borderline, 3=poor

    KNOWN INCONSISTENCY -- the two priority-0 target checks below disagree with the
    step bands beneath them, and with the achievable batch sizes:

      1. TOKENS_PER_BATCH (4e6) against TOTAL_TOKENS (15e12) implies 3,750,000
         steps, and the 2x target implies 1,875,000 -- both of which the bands in
         this same function call "too many steps" (MAX_STEPS is 1,200,000). The
         OPTIMAL band instead implies a batch of 18.8M to 50.0M tokens, i.e. 5x to
         12x the configured target.
      2. Neither window is reachable anyway. tokens_per_batch is
         dp * micro * accum * seq_len, so with powers-of-two MICROS and
         GRAD_ACCUM_VALUES it cannot land inside a +/-MARGIN (2%) window on an
         arbitrary absolute number. Swept over 22 variants x 32 presets x 5 micros
         x 4 accums = 14,080 configs: ZERO fall in either window, so priority 0 is
         never returned and generate_batch_recommendations' optimal_configs list is
         always empty.

    Which side is wrong depends on intent -- whether the user wants a 4M-token
    batch or wants 300k-800k steps -- so it is NOT guessed here. Resolving it means
    either raising TOKENS_PER_BATCH to the 18.8M-50M the bands imply, widening
    MARGIN enough to be hit, or replacing the absolute target with a range check.
    """
    # Check if matches target batch sizes
    if abs(tokens_per_batch - TOKENS_PER_BATCH) / TOKENS_PER_BATCH < MARGIN:
        return ("=== MATCHES TARGET ===", 0)
    elif abs(tokens_per_batch - TOKENS_PER_BATCH * 2) / (TOKENS_PER_BATCH * 2) < MARGIN:
        return ("=== MATCHES 2X TARGET ===", 0)

    # Check step ranges.
    #
    # The bounds are contiguous (MIN_STEPS == LOWER_BOUND_LOW_RANGE == 200k,
    # UPPER_BOUND_LOW_RANGE == LOWER_BOUND_OPTIM_RANGE == 300k,
    # UPPER_BOUND_OPTIM_RANGE == LOWER_BOUND_HIGH_RANGE == 800k,
    # UPPER_BOUND_HIGH_RANGE == MAX_STEPS == 1.2M), so they are tested here as
    # HALF-OPEN intervals. They were previously all strict on both sides, which
    # left the shared boundary values 200,000 / 300,000 / 800,000 matching no
    # branch at all and falling through to "unknown". No swept config hits them
    # exactly at seq_len 4096, so that was latent rather than live, but it is real
    # for any other sequence length or token budget.
    if steps < MIN_STEPS:
        return ("too few steps", 3)
    elif steps >= MAX_STEPS:
        return ("too many steps", 3)
    elif steps < UPPER_BOUND_LOW_RANGE:
        return ("borderline (low steps)", 2)
    elif steps < UPPER_BOUND_OPTIM_RANGE:
        return ("reasonable", 1)
    else:
        return ("borderline (high steps)", 2)


def analyze_batch_configuration(hw, micro, accum, dp, seq_len=None, variant_name=None,
                                max_viable_micro=None):
    """
    Analyze a single batch configuration.

    Args:
        hw: Hardware configuration dict
        micro: Micro batch size
        accum: Gradient accumulation steps
        dp: Data parallelism degree
        seq_len: Sequence length (defaults to the global SEQ_LEN when None)
        variant_name: Variant this configuration belongs to, for reporting
        max_viable_micro: Largest micro batch Phase 1 found fits in memory for this
            (variant, hardware) pair, or None when unknown. Phase 2 otherwise has
            NO memory awareness and will happily rate a configuration "reasonable"
            at a micro batch that cannot be allocated -- measured at 6.0% of
            displayed configs over the first 8 variants, e.g. Llama-3.1-8B on
            64 p5 rated micro=16 as priority 1 where Phase 1 found max viable 4.
            Passing it sets fits_in_memory; it does not change the priority, so a
            caller can report or filter as it chooses.

    Returns:
        dict with:
        - micro: Micro batch size
        - accum: Gradient accumulation steps
        - tokens_per_batch: Total tokens per batch
        - steps: Training steps needed
        - assessment: Assessment string
        - priority: Priority level (0-3, lower is better)
        - hardware: Hardware name
        - exceeds_batch_cap: True when tokens_per_batch is at or above the cap
          DERIVED for this configuration by compute_max_tokens_per_batch, i.e.
          max(TOTAL_TOKENS / MIN_STEPS, dp * seq_len) -- not the fixed
          MAX_TOKENS_PER_BATCH constant, which is 8x looser than MIN_STEPS implies
          and so never fired. This function does NOT drop such a configuration --
          the cap is applied by the caller in model_calculations, so it is reported
          here rather than enforced, and a caller that omits the filter would
          otherwise surface it silently.

          Consequence worth knowing when reading the CSV: because the default
          pipeline DOES filter, every exported row is already under the cap, so
          this column is uniformly False there (max exported batch 67.1M against a
          75.0M cap). It carries information only for a caller that skips the
          filter. Over the full sweep 352 of 14,080 configs exceed the derived cap;
          against the old fixed 600e6 constant, zero did.
        - fits_in_memory: True/False when max_viable_micro was supplied, else None
    """
    if seq_len is None:
        seq_len = SEQ_LEN
    tokens_per_batch = dp * micro * accum * seq_len
    steps = TOTAL_TOKENS / tokens_per_batch
    assessment, priority = assess_training_steps(steps, tokens_per_batch)

    return {
        "hardware": hw["name"],
        "variant": variant_name,
        "seq_len": seq_len,
        "micro": micro,
        "accum": accum,
        "tokens_per_batch": tokens_per_batch,
        "steps": steps,
        "assessment": assessment,
        "priority": priority,
        "exceeds_batch_cap": tokens_per_batch >= compute_max_tokens_per_batch(
            TOTAL_TOKENS, dp, seq_len
        ),
        "fits_in_memory": None if max_viable_micro is None else micro <= max_viable_micro,
    }


def generate_batch_recommendations(batch_results):
    """
    Generate recommendations for batch size configurations.

    Args:
        batch_results: Dict with hardware -> list of batch configs

    Returns:
        List of recommendation strings
    """
    recommendations = []

    for hw_name, configs in batch_results.items():
        # Find optimal configurations.
        #
        # NOTE optimal_configs is in practice ALWAYS EMPTY: priority 0 requires a
        # tokens_per_batch within +/-MARGIN of TOKENS_PER_BATCH, and no achievable
        # dp*micro*accum*seq_len lands in that window. See assess_training_steps.
        # So the good_configs path below is the one that actually runs.
        optimal_configs = [c for c in configs if c["priority"] == 0]
        good_configs = [c for c in configs if c["priority"] <= 1]

        # Of the configurations rated good or better, which cannot actually be
        # allocated? fits_in_memory is None when the caller did not supply Phase 1's
        # verdict, in which case nothing is claimed either way.
        infeasible = [c for c in good_configs if c.get("fits_in_memory") is False]

        if not optimal_configs and not good_configs:
            recommendations.append(
                f"⚠️  {hw_name}: No optimal batch configurations found - all result in poor step counts"
            )
        elif optimal_configs:
            # Show range of optimal configs
            micro_values = sorted(set(c["micro"] for c in optimal_configs))
            accum_values = sorted(set(c["accum"] for c in optimal_configs))

            if len(optimal_configs) == 1:
                c = optimal_configs[0]
                recommendations.append(
                    f"✅ {hw_name}: Optimal config is micro={c['micro']}, accum={c['accum']} "
                    f"({c['tokens_per_batch']/1e6:.1f}M tokens/batch)"
                )
            else:
                recommendations.append(
                    f"✅ {hw_name}: {len(optimal_configs)} optimal configs available "
                    f"(micro: {micro_values}, accum: {accum_values})"
                )

        if infeasible:
            over = sorted(set(c["micro"] for c in infeasible))
            feasible = [c for c in good_configs if c.get("fits_in_memory")]
            fit_note = (
                f"micro up to {max(c['micro'] for c in feasible)} does fit"
                if feasible else "NO rated config fits"
            )
            recommendations.append(
                f"⚠️  {hw_name}: {len(infeasible)} of {len(good_configs)} well-rated batch "
                f"configs need a micro batch that does NOT fit in memory "
                f"(micro {over}); {fit_note}. Batch rating ignores memory -- "
                f"cross-check against Phase 1 before using one."
            )

    return recommendations


def export_batch_results_csv(batch_results, filename):
    """
    Export batch analysis results to CSV.

    Args:
        batch_results: Dict with hardware -> list of batch configs
        filename: Output CSV filename
    """
    import csv

    with open(filename, 'w', newline='') as f:
        fieldnames = ['variant', 'hardware', 'seq_len', 'micro_batch', 'grad_accum',
                      'tokens_per_batch_M', 'training_steps', 'assessment', 'priority',
                      'fits_in_memory', 'exceeds_batch_cap']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for group_name, configs in batch_results.items():
            for config in configs:
                row = {
                    'variant': config.get('variant', ''),
                    'hardware': config.get('hardware', group_name),
                    'seq_len': config.get('seq_len', ''),
                    'micro_batch': config['micro'],
                    'grad_accum': config['accum'],
                    'tokens_per_batch_M': config['tokens_per_batch'] / 1e6,
                    'training_steps': int(config['steps']),
                    'assessment': config['assessment'],
                    'priority': config['priority'],
                    # Both may be absent or None: fits_in_memory is None when the
                    # caller did not supply Phase 1's max viable micro, and an
                    # empty cell must stay distinguishable from False. Writing ''
                    # rather than coercing to a bool keeps "not assessed" and
                    # "assessed, does not fit" apart in the export.
                    'fits_in_memory': ('' if config.get('fits_in_memory') is None
                                       else config['fits_in_memory']),
                    'exceeds_batch_cap': ('' if config.get('exceeds_batch_cap') is None
                                          else config['exceeds_batch_cap'])
                }
                writer.writerow(row)
    print(f"\n✅ Exported batch analysis to {filename}")
