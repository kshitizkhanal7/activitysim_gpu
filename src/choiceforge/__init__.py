"""ChoiceForge: GPU choice and logsum kernels for travel demand models."""

from .api import ChoiceResult
from .reference import choose_from_utilities, linear_choice

__all__ = ["ChoiceResult", "choose_from_utilities", "linear_choice"]
__version__ = "0.1.0"

