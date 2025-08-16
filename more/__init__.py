"""MoRE: Mixture of Recursions + Experts — research starter."""

from .core import (
	ProjectConfig,
	load_config,
	intro_message,
	assign_experts_and_recursions,
)

__all__ = [
	"ProjectConfig",
	"load_config",
	"intro_message",
	"assign_experts_and_recursions",
]

__version__ = "0.1.0"