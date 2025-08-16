from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class ProjectConfig:
	project_name: str
	default_name: str
	greeting_template: str

	@staticmethod
	def from_dict(data: Dict[str, Any]) -> "ProjectConfig":
		project = data.get("project", {})
		messages = data.get("messages", {})
		return ProjectConfig(
			project_name=str(project.get("name", "LabStarter")),
			default_name=str(project.get("default_name", "Researcher")),
			greeting_template=str(messages.get("greeting", "Hello")),
		)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
	"""Load YAML config into a ProjectConfig.

	If the file does not exist, falls back to defaults.
	"""
	config_path = Path(path)
	if not config_path.exists():
		# Defaults if missing
		return ProjectConfig.from_dict({})
	with config_path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	return ProjectConfig.from_dict(data)


def make_greeting(config: ProjectConfig, name: str | None = None) -> str:
	"""Compose a friendly greeting string."""
	person_name = name or config.default_name
	message = f"{config.greeting_template}, {person_name}! 🚀 You're running {config.project_name}."
	return message


def compute_stats(values: Iterable[float | int]) -> Dict[str, float]:
	"""Compute simple descriptive statistics.

	Returns count, mean, median, and stdev (stdev is 0.0 when fewer than 2 values).
	"""
	numbers: List[float] = [float(v) for v in values]
	if len(numbers) == 0:
		return {"count": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
	if len(numbers) == 1:
		return {"count": 1, "mean": numbers[0], "median": numbers[0], "stdev": 0.0}
	return {
		"count": float(len(numbers)),
		"mean": float(mean(numbers)),
		"median": float(median(numbers)),
		"stdev": float(stdev(numbers)),
	}