"""Portfolio weights and side membership returned by the model."""
from __future__ import annotations

from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class PortfolioSelection:
    weights: Tensor
    long_mask: Tensor
    short_mask: Tensor
    long_count: Tensor
    short_count: Tensor
