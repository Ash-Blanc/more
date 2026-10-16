"""Validated configuration for the text-only MoRE research prototype."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture settings for a small causal MoRE language model."""

    vocab_size: int = 259
    max_seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_recursions: int = 3
    n_experts: int = 4
    moe_top_k: int = 2
    ffn_multiplier: int = 4
    dropout: float = 0.0
    use_moe: bool = True
    use_mor: bool = True
    tie_embeddings: bool = True
    router_temperature: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_recursions": self.n_recursions,
            "n_experts": self.n_experts,
            "moe_top_k": self.moe_top_k,
            "ffn_multiplier": self.ffn_multiplier,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.moe_top_k > self.n_experts:
            raise ValueError("moe_top_k cannot exceed n_experts")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")


@dataclass
class LossConfig:
    """Weights for routing regularizers."""

    depth_balance_weight: float = 0.01
    moe_balance_weight: float = 0.01
    router_z_weight: float = 0.001
    compute_penalty_weight: float = 0.0
    target_mean_depth: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "depth_balance_weight",
            "moe_balance_weight",
            "router_z_weight",
            "compute_penalty_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.target_mean_depth is not None and self.target_mean_depth <= 0:
            raise ValueError("target_mean_depth must be positive")


@dataclass
class TrainConfig:
    """Reproducibility and optimization settings."""

    seed: int = 7
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100
    eval_interval: int = 25
    eval_batches: int = 10
    grad_clip: float = 1.0
    device: str = "auto"
    output_dir: str = "runs/tiny"
    train_split: float = 0.9

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_steps", "eval_interval", "eval_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError(
                "learning_rate/grad_clip must be positive and weight_decay non-negative"
            )
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")


@dataclass
class MoREConfig:
    """Complete experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> MoREConfig:
        data = data or {}
        model_data = dict(data.get("model", {}))
        loss_data = dict(data.get("loss", {}))
        train_data = dict(data.get("train", {}))
        # Support a flat model section for quick experiments.
        if "architecture" in data:
            model_data.update(data["architecture"])
        return cls(
            model=ModelConfig(**model_data),
            loss=LossConfig(**loss_data),
            train=TrainConfig(**train_data),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MoREConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("configuration root must be a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **values: Any) -> MoREConfig:
        """Return a copy with dotted keys such as ``train.max_steps`` overridden."""
        data = self.to_dict()
        for dotted_key, value in values.items():
            section, key = dotted_key.split(".", 1)
            if section not in data:
                raise KeyError(f"unknown configuration section: {section}")
            data[section][key] = value
        return MoREConfig.from_dict(data)
