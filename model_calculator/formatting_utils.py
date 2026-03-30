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
        priority: Priority level (0-3)

    Returns:
        Colored assessment string
    """
    if priority == 0:  # Optimal (matches target)
        return color_text(assessment, 'cyan')
    elif priority == 1:  # Good (reasonable)
        return color_text(assessment, 'green')
    elif priority == 2:  # Borderline
        return color_text(assessment, 'yellow')
    else:  # Poor (too many/few steps)
        return color_text(assessment, 'red')


def format_training_time_with_color(months_low, months_high, rel_speed):
    """
    Format training time range with color coding based on relative speed.

    Args:
        months_low: Lower bound of time estimate (months)
        months_high: Upper bound of time estimate (months)
        rel_speed: Relative speed multiplier vs baseline

    Returns:
        Colored string showing time range
    """
    time_str = f"{months_low:>4.1f}-{months_high:.1f}mo"

    # Color based on relative speed
    if rel_speed >= 2.0:
        return color_text(time_str, "green")  # Fast
    elif rel_speed >= 1.5:
        return color_text(time_str, "cyan")   # Good
    elif rel_speed >= 1.0:
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
