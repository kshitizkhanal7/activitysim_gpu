"""ChoiceForge: GPU choice and logsum kernels for travel demand models."""

from .api import ChoiceResult
from .gpu_native import GpuMemoryBudget, GpuNativeRuntime, GpuOnlyViolation
from .reference import choose_from_utilities, linear_choice

__all__ = [
    "ChoiceResult",
    "GpuMemoryBudget",
    "GpuNativeRuntime",
    "GpuOnlyViolation",
    "choose_from_utilities",
    "linear_choice",
]
__version__ = "0.1.0"
