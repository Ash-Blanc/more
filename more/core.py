from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class ProjectConfig:
	project_name: str
	default_name: str
	greeting_template: str
	routing_threshold: float
	max_recursion_depth: int

	@staticmethod
	def from_dict(data: Dict[str, Any]) -> "ProjectConfig":
		project = data.get("project", {})
		messages = data.get("messages", {})
		routing = data.get("routing", {})
		return ProjectConfig(
			project_name=str(project.get("name", "MoRE")),
			default_name=str(project.get("default_name", "Researcher")),
			greeting_template=str(messages.get("greeting", "Hello")),
			routing_threshold=float(routing.get("threshold", 0.5)),
			max_recursion_depth=int(routing.get("max_depth", 4)),
		)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
	config_path = Path(path)
	if not config_path.exists():
		return ProjectConfig.from_dict({})
	with config_path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	return ProjectConfig.from_dict(data)


def intro_message(config: ProjectConfig, name: str | None = None) -> str:
	person_name = name or config.default_name
	return (
		f"{config.greeting_template}, {person_name}! 🚀 This is MoRE — Mixture of Recursions + Experts."
	)


def assign_experts_and_recursions(
	importance_scores: Iterable[float],
	routing_threshold: float,
	max_recursion_depth: int,
) -> List[Tuple[int, int]]:
	"""Toy routing: map importance scores to (expert_id, recursion_depth).

	- Experts: 0 = 'spatial', 1 = 'temporal', 2 = 'motion', 3 = 'context'
	- Depth: linear scale from 1..max_recursion_depth based on score
	"""
	assignments: List[Tuple[int, int]] = []
	for score in importance_scores:
		s = float(score)
		expert_id = 0 if s < routing_threshold / 2 else 1 if s < routing_threshold else 2 if s < 0.85 else 3
		depth = 1 + int(round((max_recursion_depth - 1) * max(0.0, min(1.0, s))))
		assignments.append((expert_id, depth))
	return assignments