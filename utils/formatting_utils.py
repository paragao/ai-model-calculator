"""
Formatting and color utilities for terminal output.

Provides ANSI color formatting functions for various metrics and analysis outputs
across all phases of the model calculations pipeline.
"""

# Global variable for color control (can be set via command-line args)
USE_COLOR = True


def color_text(text, color_name):
    """
    Add ANSI color codes if color output enabled.

    Args:
        text: String to colorize
        color_name: Color name (green, yellow, red, cyan, reset)

    Returns:
        Colored text string with ANSI codes (if enabled)
    """
    if not USE_COLOR:
        return text

    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }

    return f"{colors.get(color_name, '')}{text}{colors['reset']}"


def format_memory_value_with_color(value_gb, headroom_gb):
    """
    Format memory value with color coding based on headroom.

    Args:
        value_gb: Memory value in GB
        headroom_gb: Headroom remaining in GB

    Returns:
        Formatted string with color
    """
    formatted = f"{value_gb:>7.1f}G"

    if headroom_gb > 20:
        return color_text(formatted, 'green')
    elif headroom_gb > 10:
        return color_text(formatted, 'yellow')
    else:
        return color_text(formatted, 'red')


def format_batch_assessment_with_color(assessment, priority):
    """
    Format batch assessment with color coding.

    Args:
        assessment: Assessment string
        priority: Priority level. Phase 2 now produces only 1 (reasonable) and
            3 (outlier); 0 and 2 are retained for callers on the old absolute-band
            scheme and are no longer emitted by classify_step_outliers.

    Returns:
        Colored assessment string
    """
    if priority == 0:  # legacy: matched an absolute target
        return color_text(assessment, 'cyan')
    elif priority == 1:  # reasonable -- inside the pair's step-count spread
        return color_text(assessment, 'green')
    elif priority == 2:  # legacy: borderline against an absolute band
        return color_text(assessment, 'yellow')
    else:
        # Outlier: the smallest or largest step count for this pair. Yellow, not
        # red -- it is extreme relative to its siblings, not an error or a failure.
        return color_text(assessment, 'yellow')


def format_training_time_with_color(months_low, months_high, efficiency_rating):
    """
    Format training time range with color coding based on efficiency rating.

    Args:
        months_low: Lower bound of time estimate (months)
        months_high: Upper bound of time estimate (months)
        efficiency_rating: Rating 0 (best) to 3 (worst), from
            classify_platform_efficiency

    Returns:
        Colored string showing time range

    Takes the rating rather than rel_speed deliberately. This function used to
    re-derive its own bands from rel_speed with a second copy of the thresholds,
    so the colour and the star rating could disagree if either copy changed.
    """
    time_str = f"{months_low:>4.1f}-{months_high:.1f}mo"

    # Color follows the rating directly -- one source of thresholds, in
    # utils/phase3_training.py.
    if efficiency_rating == 0:
        return color_text(time_str, "green")  # Fast
    elif efficiency_rating == 1:
        return color_text(time_str, "cyan")   # Good
    elif efficiency_rating == 2:
        return color_text(time_str, "yellow") # Moderate
    else:
        return color_text(time_str, "red")    # Slow


def format_communication_time_with_color(time_ms, overhead_rating):
    """
    Format communication time with color coding based on overhead rating.

    Args:
        time_ms: Communication time in milliseconds
        overhead_rating: Rating from 0 (excellent) to 3 (poor)

    Returns:
        Colored string showing communication time
    """
    time_str = f"{time_ms:.1f} ms"

    # Color based on overhead rating
    if overhead_rating == 0:
        return color_text(time_str, "green")   # Excellent
    elif overhead_rating == 1:
        return color_text(time_str, "cyan")    # Good
    elif overhead_rating == 2:
        return color_text(time_str, "yellow")  # Moderate
    else:
        return color_text(time_str, "red")     # High


def format_zero1_time_with_color(time_ms, overhead_rating):
    """
    Format ZeRO-1 all-gather time with color coding based on overhead rating.

    Args:
        time_ms: Communication time in milliseconds
        overhead_rating: Rating from 0 (excellent) to 3 (poor)

    Returns:
        Colored string showing communication time
    """
    time_str = f"{time_ms:.1f} ms"

    # Color based on overhead rating
    if overhead_rating == 0:
        return color_text(time_str, "green")   # Excellent
    elif overhead_rating == 1:
        return color_text(time_str, "cyan")    # Good
    elif overhead_rating == 2:
        return color_text(time_str, "yellow")  # Moderate
    else:
        return color_text(time_str, "red")     # High


def format_alltoall_time_with_color(time_ms, performance_rating):
    """
    Format all-to-all time with color coding based on performance rating.

    Args:
        time_ms: Communication time in milliseconds
        performance_rating: Rating from 0 (excellent) to 3 (poor)

    Returns:
        Colored string showing communication time
    """
    time_str = f"{time_ms:.2f} ms"

    # Color based on performance rating
    if performance_rating == 0:
        return color_text(time_str, "green")   # Excellent
    elif performance_rating == 1:
        return color_text(time_str, "cyan")    # Good
    elif performance_rating == 2:
        return color_text(time_str, "yellow")  # Moderate
    else:
        return color_text(time_str, "red")     # High
