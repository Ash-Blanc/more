from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .core import DEFAULT_CONFIG_PATH, compute_stats, load_config, make_greeting


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="labstarter",
		description="Run LabStarter demo CLI",
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG_PATH,
		help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
	)
	parser.add_argument(
		"--name",
		type=str,
		default=None,
		help="Name to greet",
	)
	parser.add_argument(
		"--numbers",
		nargs="*",
		type=float,
		help="Optional list of numbers to compute basic stats",
	)
	return parser


def main(argv: List[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	config = load_config(args.config)
	print(make_greeting(config, args.name))

	if args.numbers:
		stats = compute_stats(args.numbers)
		print(
			f"count={int(stats['count'])} mean={stats['mean']:.2f} "
			f"median={stats['median']:.2f} stdev={stats['stdev']:.2f}"
		)
	return 0