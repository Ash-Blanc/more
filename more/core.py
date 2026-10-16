"""Backward-compatible helpers for the text-focused MoRE package."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import MoREConfig

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class ProjectConfig:
    """Legacy presentation and toy-routing settings.

    The model configuration lives in :class:`more.config.MoREConfig`; this
    small type remains so the original demo and notebooks keep working.
    """

    project_name: str = "MoRE"
    default_name: str = "Researcher"
    greeting_template: str = "Hello"
    routing_threshold: float = 0.5
    max_recursion_depth: int = 4

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> ProjectConfig:
        data = data or {}
        project = data.get("project", {}) or {}
        messages = data.get("messages", {}) or {}
        routing = data.get("routing", {}) or {}
        threshold = float(routing.get("threshold", 0.5))
        max_depth = int(routing.get("max_depth", 4))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("routing.threshold must be between 0 and 1")
        if max_depth <= 0:
            raise ValueError("routing.max_depth must be positive")
        return ProjectConfig(
            project_name=str(project.get("name", "MoRE")),
            default_name=str(project.get("default_name", "Researcher")),
            greeting_template=str(messages.get("greeting", "Hello")),
            routing_threshold=threshold,
            max_recursion_depth=max_depth,
        )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("configuration root must be a mapping")
    return data


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load the original presentation/toy-routing configuration."""

    return ProjectConfig.from_dict(_read_yaml(path))


def load_experiment_config(path: str | Path = DEFAULT_CONFIG_PATH) -> MoREConfig:
    """Load validated neural-model settings from YAML."""

    return MoREConfig.from_dict(_read_yaml(path))


def intro_message(config: ProjectConfig, name: str | None = None) -> str:
    person_name = name or config.default_name
    return f"{config.greeting_template}, {person_name}! 🚀 This is MoRE — Mixture of Recursions + Experts."


def assign_experts_and_recursions(
    importance_scores: Iterable[float],
    routing_threshold: float,
    max_recursion_depth: int,
) -> list[tuple[int, int]]:
    """Map scalar scores to legacy expert/depth pairs.

    This is deliberately retained as a toy baseline. Neural routing is
    implemented by :mod:`more.routing` and operates on token representations.
    """

    if not 0.0 <= routing_threshold <= 1.0:
        raise ValueError("routing_threshold must be between 0 and 1")
    if max_recursion_depth <= 0:
        raise ValueError("max_recursion_depth must be positive")
    assignments: list[tuple[int, int]] = []
    for score in importance_scores:
        value = float(score)
        expert_id = (
            0
            if value < routing_threshold / 2
            else 1
            if value < routing_threshold
            else 2
            if value < 0.85
            else 3
        )
        depth = 1 + round((max_recursion_depth - 1) * max(0.0, min(1.0, value)))
        assignments.append((expert_id, depth))
    return assignments
