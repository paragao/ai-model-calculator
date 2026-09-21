"""
Hardware loading and selection helpers for the AI Model Calculator CLI.

Provides:
  - load_hardware_from_json: parse a JSON hardware file into a list of
    hardware-config dicts (validated; no derived fields).
  - select_hardware_by_name: case-insensitive exact match of a hardware name.
"""

import json


class HardwareNotFoundError(ValueError):
    """Raised when a requested hardware name does not match any entry."""

    def __init__(self, name, available):
        self.name = name
        self.available = list(available)
        super().__init__(name)


# Fields the analysis phases read directly from each hardware entry.
REQUIRED_HW_FIELDS = (
    "name",
    "gpus",
    "nodes",
    "mem_gb",
    "inter_node_bw_gb",
    "intra_node_bw_gbps",
    "peak_tflops_bf16",
)


def _validate_hw_fields(hw, index):
    """Ensure a hardware entry carries the fields the phases require."""
    missing = [f for f in REQUIRED_HW_FIELDS if f not in hw]
    if missing:
        raise ValueError(
            f"Hardware '{hw.get('name', f'#{index}')}' is missing required "
            f"field(s): {', '.join(missing)}."
        )


def _extract_hardware_list(data):
    """Normalize parsed JSON into a list of hardware dicts.

    Accepts three shapes:
      - a single hardware object: {"name": ..., ...}
      - a bare list of hardware objects: [ {...}, {...} ]
      - a wrapper object: {"hardware": [ {...}, ... ]}
    """
    if isinstance(data, dict):
        if "hardware" in data:
            hardware = data["hardware"]
        else:
            hardware = [data]
    elif isinstance(data, list):
        hardware = data
    else:
        raise ValueError(
            f"Unsupported JSON structure: expected object or list, got {type(data).__name__}"
        )

    if not isinstance(hardware, list):
        raise ValueError('"hardware" must be a list of hardware objects')
    if not hardware:
        raise ValueError("JSON file contains no hardware entries")
    for i, hw in enumerate(hardware):
        if not isinstance(hw, dict):
            raise ValueError(f"Hardware at index {i} is not an object")
        if "name" not in hw:
            raise ValueError(f"Hardware at index {i} is missing required field 'name'")
        _validate_hw_fields(hw, i)
    return hardware


def load_hardware_from_json(path):
    """Load and validate hardware entries from a JSON file.

    Args:
        path: Path to the JSON hardware file.

    Returns:
        list: Hardware-config dicts (validated; used as-is by the phases).
    """
    with open(path) as f:
        data = json.load(f)

    return _extract_hardware_list(data)


def select_hardware_by_name(hardware, name):
    """Return the single hardware entry whose name matches (case-insensitive).

    Args:
        hardware: List of hardware dicts.
        name: Requested hardware name (matched case-insensitively, exact).

    Returns:
        The matching hardware dict.

    Raises:
        HardwareNotFoundError: If no hardware name matches. Carries the list of
            available names so the caller can present them.
    """
    target = name.strip().lower()
    for hw in hardware:
        if hw["name"].lower() == target:
            return hw
    raise HardwareNotFoundError(name, [hw["name"] for hw in hardware])
