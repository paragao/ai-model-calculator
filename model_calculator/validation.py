"""
Configuration validation for model calculations.

Validates parallelization parameters, precision settings, and hardware configurations
before running analysis phases.
"""

# Import configurations
from variants_config import VARIANTS
from hardware_config import HARDWARE
from project_config import *
from advanced_config import *


def validate_configuration():
    """
    Validate all configuration parameters and hardware specs.

    Validates:
    - Parallelization parameters (EP, PP, TP, CP) are positive
    - N_EXPERTS is divisible by EP
    - PRECISION is valid (BF16 or FP8)
    - Hardware configurations have sufficient GPUs
    - GPU counts are divisible by parallelization factors

    Raises:
        AssertionError: If any validation check fails
    """
    # Validate parallelization parameters
    assert EP > 0, f"EP must be positive, got {EP}"
    assert N_EXPERTS % EP == 0, (
        f"N_EXPERTS ({N_EXPERTS}) must be divisible by EP ({EP}). "
        f"Current setup would leave {N_EXPERTS % EP} experts unassigned."
    )
    assert PP > 0, f"PP (Pipeline Parallelism) must be positive, got {PP}"
    assert TP > 0, f"TP (Tensor Parallelism) must be positive, got {TP}"
    assert CP > 0, f"CP (Context Parallelism) must be positive, got {CP}"

    # Validate precision setting
    assert PRECISION in ["BF16", "FP8"], (
        f"PRECISION must be 'BF16' or 'FP8', got '{PRECISION}'"
    )

    # Validate hardware configurations
    for hw in HARDWARE:
        total_gpus = hw["gpus"]
        required_gpus = TP * PP * CP

        assert total_gpus >= required_gpus, (
            f"{hw['name']}: Insufficient GPUs. "
            f"Has {total_gpus} GPUs but requires minimum {required_gpus} "
            f"(TP={TP} × PP={PP} × CP={CP})"
        )

        assert total_gpus % required_gpus == 0, (
            f"{hw['name']}: GPU count mismatch. "
            f"{total_gpus} GPUs is not divisible by {required_gpus} "
            f"(TP={TP} × PP={PP} × CP={CP}). "
            f"Cannot evenly distribute workload across DP dimension."
        )

    print("✓ Configuration validation passed")
    print(f"  - Parallelization: PP={PP}, TP={TP}, EP={EP}, CP={CP}")
    print(f"  - Expert sharding: {N_EXPERTS} experts across {EP} GPUs = {EXPERTS_PER_GPU} experts/GPU")
    print(f"  - Precision: {PRECISION} ({PARAM_BYTES} bytes/param)")
    print()
