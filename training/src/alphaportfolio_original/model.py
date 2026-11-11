"""Per-stock temporal encoding and portfolio outputs."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .portfolio import PortfolioSelection


@dataclass
class AlphaPortfolioOutput:
    scores: Tensor
    temporal_representation: Tensor
    relation_representation: Tensor
    temporal_attention: Tensor | None
    cross_asset_attention: Tensor | None
    portfolio: PortfolioSelection | None = None


class TemporalEncoderBlock(nn.Module):
    """Transformer block with layer normalization after each residual addition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(
            config.d_model,
            elementwise_affine=config.layer_norm_affine,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model),
        )
        self.norm2 = nn.LayerNorm(
            config.d_model,
            elementwise_affine=config.layer_norm_affine,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        tokens: Tensor,
        *,
        return_attention: bool,
    ) -> tuple[Tensor, Tensor | None]:
        attended, attention = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = self.norm1(tokens + self.dropout1(attended))
        encoded = self.norm2(hidden + self.dropout2(self.ffn(hidden)))
        return encoded, attention


class SequenceRepresentationExtractionModule(nn.Module):
    """Transformer encoder applied independently to each stock and time window."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            config.input_dim,
            config.d_model,
            bias=config.input_projection_bias,
        )
        if config.use_learned_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.empty(config.lookback_months, config.d_model)
            )
        else:
            self.register_buffer(
                "position_embedding",
                _sinusoidal_position_encoding(
                    config.lookback_months,
                    config.d_model,
                ),
                persistent=True,
            )
        self.layers = nn.ModuleList(
            TemporalEncoderBlock(config) for _ in range(config.temporal_layers)
        )
        if config.temporal_pooling == "flatten":
            self.flatten_projection = nn.Linear(
                config.lookback_months * config.d_model,
                config.d_model,
            )
        else:
            self.flatten_projection = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        if isinstance(self.position_embedding, nn.Parameter):
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        states: Tensor,
        *,
        return_attention: bool,
    ) -> tuple[Tensor, Tensor | None]:
        if states.ndim != 4:
            raise ValueError(
                "states must have shape [batch, assets, lookback, features]"
            )
        batch, assets, lookback, features = states.shape
        if lookback != self.config.lookback_months:
            raise ValueError(
                f"expected {self.config.lookback_months} lookback months, got {lookback}"
            )
        if features != self.config.input_dim:
            raise ValueError(
                f"expected {self.config.input_dim} features, got {features}"
            )
        if not torch.isfinite(states).all():
            raise ValueError("states contains non-finite values")

        tokens = self.input_projection(states.reshape(batch * assets, lookback, features))
        tokens = tokens + self.position_embedding.unsqueeze(0)

        attentions: list[Tensor] = []
        for layer in self.layers:
            tokens, attention = layer(
                tokens,
                return_attention=return_attention,
            )
            if attention is not None:
                attentions.append(attention)

        if self.config.temporal_pooling == "last":
            representation = tokens[:, -1, :]
        elif self.config.temporal_pooling == "mean":
            representation = tokens.mean(dim=1)
        elif self.config.temporal_pooling == "flatten":
            assert self.flatten_projection is not None
            representation = self.flatten_projection(tokens.flatten(start_dim=1))
        else:
            raise RuntimeError(
                f"unsupported temporal pooling: {self.config.temporal_pooling}"
            )

        representation = representation.reshape(batch, assets, self.config.d_model)
        stacked_attention = None
        if attentions:
            stacked_attention = torch.stack(attentions, dim=1)
            stacked_attention = stacked_attention.reshape(
                batch,
                assets,
                self.config.temporal_layers,
                self.config.n_heads,
                lookback,
                lookback,
            )
        return representation, stacked_attention


def _sinusoidal_position_encoding(length: int, dimension: int) -> Tensor:
    """Non-trainable standard Transformer position encoding."""

    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales)
    return encoding
