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
    """
    # Check if matches target batch sizes
    if abs(tokens_per_batch - TOKENS_PER_BATCH) / TOKENS_PER_BATCH < MARGIN:
        return ("=== MATCHES TARGET ===", 0)
    elif abs(tokens_per_batch - TOKENS_PER_BATCH * 2) / (TOKENS_PER_BATCH * 2) < MARGIN:
        return ("=== MATCHES 2X TARGET ===", 0)

    # Check step ranges
    if LOWER_BOUND_OPTIM_RANGE < steps < UPPER_BOUND_OPTIM_RANGE:
        return ("reasonable", 1)
    elif LOWER_BOUND_LOW_RANGE < steps < UPPER_BOUND_LOW_RANGE:
        return ("borderline (low steps)", 2)
    elif LOWER_BOUND_HIGH_RANGE < steps < UPPER_BOUND_HIGH_RANGE:
        return ("borderline (high steps)", 2)
    elif steps >= MAX_STEPS:
        return ("too many steps", 3)
    elif steps < MIN_STEPS:
        return ("too few steps", 3)
    else:
        return ("unknown", 2)


def analyze_batch_configuration(hw, micro, accum, dp):
    """
    Analyze a single batch configuration.

    Args:
        hw: Hardware configuration dict
        micro: Micro batch size
        accum: Gradient accumulation steps
        dp: Data parallelism degree

    Returns:
        dict with:
        - micro: Micro batch size
        - accum: Gradient accumulation steps
        - tokens_per_batch: Total tokens per batch
        - steps: Training steps needed
        - assessment: Assessment string
        - priority: Priority level (0-3, lower is better)
        - hardware: Hardware name
    """
    tokens_per_batch = dp * micro * accum * SEQ_LEN
    steps = TOTAL_TOKENS / tokens_per_batch
    assessment, priority = assess_training_steps(steps, tokens_per_batch)

    return {
        "hardware": hw["name"],
        "micro": micro,
        "accum": accum,
        "tokens_per_batch": tokens_per_batch,
        "steps": steps,
        "assessment": assessment,
        "priority": priority,
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
        # Find optimal configurations
        optimal_configs = [c for c in configs if c["priority"] == 0]
        good_configs = [c for c in configs if c["priority"] <= 1]

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
        fieldnames = ['hardware', 'micro_batch', 'grad_accum', 'tokens_per_batch_M',
                      'training_steps', 'assessment', 'priority']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hw_name, configs in batch_results.items():
            for config in configs:
                row = {
                    'hardware': hw_name,
                    'micro_batch': config['micro'],
                    'grad_accum': config['accum'],
                    'tokens_per_batch_M': config['tokens_per_batch'] / 1e6,
                    'training_steps': int(config['steps']),
                    'assessment': config['assessment'],
                    'priority': config['priority']
                }
                writer.writerow(row)

    print(f"\n✅ Exported batch analysis to {filename}")
