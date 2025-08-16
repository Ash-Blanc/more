from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

from more.core import assign_experts_and_recursions, intro_message, load_config


def run() -> None:
	cfg = load_config("config.yaml")
	print(intro_message(cfg, name="Ada"))
	assignments = assign_experts_and_recursions([0.2, 0.5, 0.8, 0.95], cfg.routing_threshold, cfg.max_recursion_depth)
	print(assignments)


if __name__ == "__main__":
	run()