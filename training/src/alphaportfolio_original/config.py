"""Encoder and portfolio dimensions used to construct a RECAST checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


TemporalPooling = Literal["last", "mean", "flatten"]


@dataclass(frozen=True)
class ModelConfig:
    """Per-stock temporal encoder architecture."""

    input_dim: int = 51
    lookback_months: int = 12
    d_model: int = 256
    n_heads: int = 4
    feedforward_dim: int = 1024
    temporal_layers: int = 1
    dropout: float = 0.2
    temporal_pooling: TemporalPooling = "last"
    score_hidden_dim: int = 768
    use_learned_position_embedding: bool = False
    input_projection_bias: bool = False
    layer_norm_affine: bool = False
    relation_feedforward: bool = True
    cross_attention_bias: bool = True

    def validate(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.lookback_months < 2:
            raise ValueError("lookback_months must be at least two")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.temporal_layers < 1:
            raise ValueError("temporal_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class PortfolioConfig:
    """Long and short candidate-selection fractions."""

    long_fraction: float = 0.10
    short_fraction: float = 0.10
    normalize_return_by_gross_exposure: bool = True
    minimum_assets: int = 20

    def validate(self) -> None:
        if not 0 < self.long_fraction < 0.5:
            raise ValueError("long_fraction must lie in (0, 0.5)")
        if not 0 < self.short_fraction < 0.5:
            raise ValueError("short_fraction must lie in (0, 0.5)")
        if self.long_fraction + self.short_fraction > 1:
            raise ValueError("long and short fractions cannot exceed one in total")
        if self.minimum_assets < 2:
            raise ValueError("minimum_assets must be at least two")


@dataclass(frozen=True)
class AlphaPortfolioConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    def validate(self) -> None:
        self.model.validate()
        self.portfolio.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AlphaPortfolioConfig":
        config = cls(
            model=ModelConfig(**payload.get("model", {})),
            portfolio=PortfolioConfig(**payload.get("portfolio", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "AlphaPortfolioConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("AlphaPortfolio config JSON must contain an object")
        return cls.from_dict(payload)
