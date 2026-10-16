"""MoRE: a text-focused Mixture of Recursions + Experts research starter."""

from .config import LossConfig, ModelConfig, MoREConfig, TrainConfig
from .core import (
    ProjectConfig,
    assign_experts_and_recursions,
    intro_message,
    load_config,
    load_experiment_config,
)
from .model import MoREModel, MoREOutput

__all__ = [
    "LossConfig",
    "MoREConfig",
    "MoREModel",
    "MoREOutput",
    "ModelConfig",
    "ProjectConfig",
    "TrainConfig",
    "assign_experts_and_recursions",
    "intro_message",
    "load_config",
    "load_experiment_config",
]

__version__ = "0.2.0"
