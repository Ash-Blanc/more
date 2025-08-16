from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

from labstarter.core import compute_stats, load_config, make_greeting


def run() -> None:
	config = load_config("config.yaml")
	print(make_greeting(config, name="Ada"))
	stats = compute_stats([1, 2, 3, 4, 5])
	print(stats)


if __name__ == "__main__":
	run()