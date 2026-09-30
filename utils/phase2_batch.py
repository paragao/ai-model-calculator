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


def classify_step_outliers(configs):
    """Classify one (variant, hardware) pair's configurations by step-count spread.

    Phase 2 does NOT recommend an absolute step count. It reports the step count
    each configuration implies -- steps = TOTAL_TOKENS / tokens_per_batch -- and
    then judges configurations only RELATIVE TO EACH OTHER: the smallest and the
    largest step values are outliers, and everything between them is reasonable.

    Why relative rather than absolute. The previous version scored each
    configuration against fixed bands (MIN_STEPS, an "optimal" 300k-800k window,
    MAX_STEPS) plus a priority-0 check for matching TOKENS_PER_BATCH. Both were
    unusable:
      - the target check was unreachable. tokens_per_batch = dp*micro*accum*seq_len
        cannot land within MARGIN (2%) of an arbitrary absolute number, so 0 of
        14,080 swept configurations hit it and priority 0 was never returned.
      - the bands disagreed with the configured target. TOKENS_PER_BATCH (4e6)
        against TOTAL_TOKENS (15e12) implies 3,750,000 steps, which the same
        function called "too many steps", and 81% of all configurations landed
        there -- a verdict that told the reader nothing.
    An absolute band also cannot be right for every token budget and topology at
    once, whereas the spread within a pair is always meaningful.

    Args:
        configs: List of dicts from analyze_batch_configuration, ALL belonging to
            the same (variant, hardware) pair. Order does not matter.

    Returns:
        The same list, with "assessment" and "priority" set on each config:
          priority 1 = "reasonable"
          priority 3 = "outlier (fewest steps)" or "outlier (most steps)"
        Only 1 and 3 are produced; the 0-3 range is kept so existing priority
        indexing stays valid.

    OUTLIERS ARE DROPPED BY VALUE, NOT BY POSITION, which matters here because the
    step counts are heavily degenerate: micro*accum takes repeated products (2x8,
    4x4, 8x2 and 16x1 all give 128), so a pair's 17-20 configurations collapse onto
    only 6-8 DISTINCT step values. Every configuration sharing the extreme value is
    equally extreme, so all of them are marked, rather than arbitrarily keeping a
    tied sibling. Measured over the full catalog: 968 low outliers, 704 high, and
    12,056 reasonable of 13,728 configurations.

    A set with FEWER THAN 3 DISTINCT step values is left entirely reasonable. With
    two values or one, trimming both ends would discard everything and report
    nothing, so there is nothing meaningful to trim. No pair in the default catalog
    hits this (the minimum is 6 distinct values), but a custom variant or a narrow
    MICROS list can.

    LIMITATION worth knowing before trusting "reasonable". The distinct step values
    form a geometric series, because micro and accum are powers of two -- so each
    value is 2x its neighbour and trimming one off each end removes only a factor
    of 2 per side. For Llama-3.1-8B on 64 p5 the eight values span 447,034 to
    57,220,458 steps (128x); after trimming, "reasonable" still spans 894,069 to
    28,610,229 (32x). This function tells you which configurations are extreme
    RELATIVE to their siblings. It does not assert that a 32x range is all equally
    sensible, and it deliberately makes no claim about absolute step counts.
    """
    if not configs:
        return configs

    distinct = sorted({c["steps"] for c in configs})

    # Fewer than three distinct values: trimming both ends leaves nothing, so
    # there is no meaningful outlier to name.
    if len(distinct) < 3:
        for c in configs:
            c["assessment"] = "reasonable"
            c["priority"] = 1
        return configs

    fewest, most = distinct[0], distinct[-1]
    for c in configs:
        if c["steps"] == fewest:
            c["assessment"] = "outlier (fewest steps)"
            c["priority"] = 3
        elif c["steps"] == most:
            c["assessment"] = "outlier (most steps)"
            c["priority"] = 3
        else:
            c["assessment"] = "reasonable"
            c["priority"] = 1
    return configs


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
        - steps: Training steps needed -- TOTAL_TOKENS / tokens_per_batch. This is
          the number Phase 2 REPORTS; it no longer scores it against an absolute
          target or band.
        - assessment: None until classify_step_outliers has run over the whole
          (variant, hardware) pair. The verdict is relative to the pair's other
          configurations, so it cannot be set here.
        - priority: None for the same reason. After classification it is 1
          (reasonable) or 3 (outlier).
        - steps_absurd: True when steps exceeds compute_absurd_step_ceiling, i.e.
          when the batch is smaller than Llama 3's ramp-start 4M and so smaller
          than any published frontier run used at any point. This is the ABSOLUTE
          sanity check and is INDEPENDENT of assessment/priority, which are purely
          relative: a configuration can be "reasonable" among its siblings and
          still be absurd, which is precisely the case on the six small-GPU presets
          where every configuration in the pair is absurd. Note there is no
          too-few-steps counterpart because the caller's cap already guarantees
          steps > MIN_STEPS -- see compute_absurd_step_ceiling.
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

    return {
        "hardware": hw["name"],
        "variant": variant_name,
        "seq_len": seq_len,
        "micro": micro,
        "accum": accum,
        "tokens_per_batch": tokens_per_batch,
        "steps": steps,
        # DELIBERATELY None. A configuration's assessment is RELATIVE to the other
        # configurations for the same (variant, hardware) pair -- the smallest and
        # largest step counts are the outliers -- so it cannot be decided while
        # looking at one configuration in isolation. Call classify_step_outliers on
        # the whole pair's list to fill these in. They are left None rather than
        # given a plausible provisional value on purpose: a provisional priority
        # would read as a real verdict if the classification step were ever missed,
        # and generate_batch_recommendations raises rather than accept one.
        "assessment": None,
        "priority": None,
        # The ABSOLUTE axis, and unlike assessment/priority it CAN be set here,
        # because it compares one configuration against a fixed published
        # reference rather than against its siblings. Kept as its own field
        # instead of folded into priority so the two axes stay readable: a
        # configuration can be the least extreme in its pair and still ask for a
        # batch no published run used.
        "steps_absurd": steps > compute_absurd_step_ceiling(TOTAL_TOKENS),
        "exceeds_batch_cap": tokens_per_batch >= compute_max_tokens_per_batch(
            TOTAL_TOKENS, dp, seq_len
        ),
        "fits_in_memory": None if max_viable_micro is None else micro <= max_viable_micro,
    }


def generate_batch_recommendations(batch_results):
    """
    Generate recommendations for batch size configurations.

    Reports the step-count SPREAD for each (variant, hardware) pair and names the
    outliers to drop, rather than asserting an absolute step target. See
    classify_step_outliers for why the absolute bands were removed.

    Emits on TWO INDEPENDENT AXES, which answer different questions and can
    disagree. The relative axis (📊) says which configurations are least extreme
    among their siblings. The absolute axis (🚨/⚠️) says whether the pair is inside
    published practice at all -- see compute_absurd_step_ceiling. A pair can be
    wholly absurd and still have "reasonable" siblings, because "reasonable" is
    only ever a comparison within the pair; that case is escalated to 🚨 precisely
    because the relative rule cannot express it.

    Args:
        batch_results: Dict with "variant | hardware" -> list of batch configs,
            already passed through classify_step_outliers.

    Returns:
        List of recommendation strings

    Raises:
        ValueError: if any configuration is still unclassified. The priority is
            assigned by a separate set-level pass, so a caller that forgets it
            would otherwise get silently empty recommendations -- which is exactly
            how the previous always-empty optimal_configs list went unnoticed.
    """
    recommendations = []

    for hw_name, configs in batch_results.items():
        if any(c.get("priority") is None for c in configs):
            raise ValueError(
                f"{hw_name}: {sum(1 for c in configs if c.get('priority') is None)} "
                f"configuration(s) have no priority. Call classify_step_outliers on "
                f"each (variant, hardware) pair's list before generating "
                f"recommendations."
            )

        if not configs:
            continue

        reasonable = [c for c in configs if c["priority"] == 1]
        outliers = [c for c in configs if c["priority"] == 3]

        if not reasonable:
            # Only reachable with fewer than 3 distinct step values, where nothing
            # is trimmed -- so an empty reasonable list means an empty pair.
            recommendations.append(
                f"⚠️  {hw_name}: no configurations to assess"
            )
            continue

        steps = sorted(c["steps"] for c in reasonable)
        lo_b = min(c["tokens_per_batch"] for c in reasonable)
        hi_b = max(c["tokens_per_batch"] for c in reasonable)

        if outliers:
            dropped = sorted({int(c["steps"]) for c in outliers})
            recommendations.append(
                f"📊 {hw_name}: {len(reasonable)} reasonable config(s) spanning "
                f"{steps[0]:,.0f} to {steps[-1]:,.0f} steps "
                f"({lo_b/1e6:.1f}M-{hi_b/1e6:.1f}M tokens/batch). "
                f"Dropped {len(outliers)} outlier(s) at the extremes: "
                f"{', '.join(f'{d:,}' for d in dropped)} steps."
            )
        else:
            recommendations.append(
                f"📊 {hw_name}: {len(reasonable)} config(s) spanning "
                f"{steps[0]:,.0f} to {steps[-1]:,.0f} steps "
                f"({lo_b/1e6:.1f}M-{hi_b/1e6:.1f}M tokens/batch). "
                f"Too few distinct step counts to trim outliers."
            )

        # Of the configurations rated reasonable, which cannot actually be
        # allocated? fits_in_memory is None when the caller did not supply Phase 1's
        # verdict, in which case nothing is claimed either way.
        infeasible = [c for c in reasonable if c.get("fits_in_memory") is False]
        if infeasible:
            over = sorted(set(c["micro"] for c in infeasible))
            feasible = [c for c in reasonable if c.get("fits_in_memory")]
            fit_note = (
                f"micro up to {max(c['micro'] for c in feasible)} does fit"
                if feasible else "NO reasonable config fits"
            )
            recommendations.append(
                f"⚠️  {hw_name}: {len(infeasible)} of {len(reasonable)} reasonable batch "
                f"configs need a micro batch that does NOT fit in memory "
                f"(micro {over}); {fit_note}. Batch rating ignores memory -- "
                f"cross-check against Phase 1 before using one."
            )

        # The absolute axis. This is reported SEPARATELY from the relative verdict
        # because the two answer different questions: "reasonable" means least
        # extreme among its siblings, which says nothing about whether the whole
        # pair is credible. The wholly-absurd case is the one the relative rule
        # structurally cannot see, so it is escalated rather than folded in.
        #
        # DELIBERATELY NOT "any absurd config". Every micro x accum grid contains
        # micro=1/accum=1, which is the smallest expressible batch and therefore
        # the most steps, so at least one absurd configuration is near-structural:
        # measured at 572 of 704 pairs (81%). Warning on that would reproduce the
        # exact failure of the step bands this replaced -- flagging four pairs in
        # five and so carrying no information. The threshold is instead whether the
        # pair's LARGEST-batch reasonable configuration is absurd, i.e. whether
        # following the relative recommendation still lands outside published
        # practice: 220 of 704 pairs (31%), of which 132 are wholly absurd. The
        # remaining per-configuration detail is in the CSV's steps_absurd column.
        absurd = [c for c in configs if c.get("steps_absurd")]
        best_reasonable = max(reasonable, key=lambda c: c["tokens_per_batch"])
        ceiling = compute_absurd_step_ceiling(TOTAL_TOKENS)

        if len(absurd) == len(configs):
            fewest = min(c["steps"] for c in configs)
            recommendations.append(
                f"🚨 {hw_name}: ALL {len(configs)} configurations imply more than "
                f"{ceiling:,.0f} steps -- the fewest is {fewest:,.0f}. Every batch "
                f"here is smaller than any published frontier run used at any "
                f"point in training, so the 'reasonable' ratings above are "
                f"relative to siblings that are themselves outside published "
                f"practice. This hardware is too small for this variant; no "
                f"configuration in this pair is credible."
            )
        elif best_reasonable.get("steps_absurd"):
            recommendations.append(
                f"⚠️  {hw_name}: even the largest reasonable batch "
                f"({best_reasonable['tokens_per_batch']/1e6:.1f}M tokens, "
                f"{best_reasonable['steps']:,.0f} steps) exceeds {ceiling:,.0f} "
                f"steps and is outside published practice. The relative verdict "
                f"ranks within this pair only -- scale up the hardware or raise "
                f"micro/accum beyond this grid for a credible batch."
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
                      'fits_in_memory', 'exceeds_batch_cap', 'steps_absurd']
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
                                          else config['exceeds_batch_cap']),
                    # The ABSOLUTE axis, deliberately a separate column from
                    # assessment/priority rather than folded into them: a row can
                    # read priority 1 "reasonable" and steps_absurd True at the
                    # same time, because the verdict is relative to the pair while
                    # this is measured against published practice. Filtering a
                    # report on priority alone will therefore keep absurd rows.
                    'steps_absurd': ('' if config.get('steps_absurd') is None
                                     else config['steps_absurd'])
                }
                writer.writerow(row)
    print(f"\n✅ Exported batch analysis to {filename}")
