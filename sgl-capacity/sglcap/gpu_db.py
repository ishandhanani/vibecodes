"""
GPU database for common GPU types.

Provides GPU specifications including memory, compute capability,
and cluster topology information for capacity simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class GPUSpec:
    """GPU specification for capacity simulation."""
    name: str
    memory_gb: float
    compute_capability: Tuple[int, int]  # (major, minor) - e.g., (9, 0) for sm90
    aliases: Tuple[str, ...]  # Alternative names for CLI
    gpus_per_node: int = 8  # Default GPUs per node
    gpus_per_rack: Optional[int] = None  # Only some GPUs have rack concepts

    @property
    def sm_version(self) -> str:
        """Return SM version string (e.g., 'sm90')."""
        return f"sm{self.compute_capability[0]}{self.compute_capability[1]}"


# GPU Database with compute capabilities
# Compute capabilities:
#   - L40S: Ada Lovelace (sm89)
#   - H100/H200: Hopper (sm90)
#   - GB200/GB300: Blackwell (sm100)
GPU_DB: Dict[str, GPUSpec] = {
    "l40s": GPUSpec(
        name="NVIDIA L40S",
        memory_gb=48,
        compute_capability=(8, 9),  # Ada Lovelace
        aliases=("l40s", "l40"),
        gpus_per_node=8,
    ),
    "h100": GPUSpec(
        name="NVIDIA H100 80GB HBM3",
        memory_gb=80,
        compute_capability=(9, 0),  # Hopper
        aliases=("h100", "h100-80", "h100-80gb"),
        gpus_per_node=8,
    ),
    "h200": GPUSpec(
        name="NVIDIA H200 141GB HBM3e",
        memory_gb=141,
        compute_capability=(9, 0),  # Hopper
        aliases=("h200", "h200-141", "h200-141gb"),
        gpus_per_node=8,
    ),
    "gb200": GPUSpec(
        name="NVIDIA GB200 192GB HBM3e",
        memory_gb=192,
        compute_capability=(10, 0),  # Blackwell
        aliases=("gb200", "b200"),
        gpus_per_node=4,  # GB200 NVL72: 4 GPUs per node
        gpus_per_rack=72,  # GB200 NVL72: 72 GPUs per rack
    ),
    "gb300": GPUSpec(
        name="NVIDIA GB300 288GB HBM3e",
        memory_gb=288,
        compute_capability=(10, 0),  # Blackwell
        aliases=("gb300", "b300"),
        gpus_per_node=4,  # GB300 nodes have 4 GPUs
        gpus_per_rack=72,  # GB300 racks have 72 GPUs
    ),
}

# Build alias lookup
_ALIAS_MAP: Dict[str, str] = {}
for gpu_id, spec in GPU_DB.items():
    for alias in spec.aliases:
        _ALIAS_MAP[alias.lower()] = gpu_id


def get_gpu_spec(gpu_type: str) -> Optional[GPUSpec]:
    """
    Look up GPU spec by name or alias.

    Args:
        gpu_type: GPU type string (e.g., "l40s", "a100-80", "h100")

    Returns:
        GPUSpec if found, None otherwise
    """
    key = gpu_type.lower().strip()
    if key in _ALIAS_MAP:
        return GPU_DB[_ALIAS_MAP[key]]
    return None


def list_gpus() -> str:
    """Return formatted list of supported GPUs."""
    lines = ["Supported GPUs:"]
    for gpu_id, spec in GPU_DB.items():
        lines.append(
            f"  {gpu_id:8} {spec.memory_gb:5.0f} GB  {spec.sm_version}  "
            f"{spec.gpus_per_node} GPUs/node  ({spec.name})"
        )
    return "\n".join(lines)


__all__ = [
    "GPUSpec",
    "GPU_DB",
    "get_gpu_spec",
    "list_gpus",
]
