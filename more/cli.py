from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .core import (
	DEFAULT_CONFIG_PATH,
	assign_experts_and_recursions,
	intro_message,
	load_config,
)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="more",
		description="MoRE: Mixture of Recursions + Experts — research CLI",
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG_PATH,
		help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
	)
	subparsers = parser.add_subparsers(dest="command", required=False)

	# `demo` subcommand: prints intro message
	demo = subparsers.add_parser("demo", help="Print a short MoRE intro message")
	demo.add_argument("--name", type=str, default=None, help="Name to greet")

	# `route` subcommand: assign experts & recursion depths
	route = subparsers.add_parser("route", help="Assign experts/depths for scores")
	route.add_argument("--scores", nargs="*", type=float, help="Importance scores in [0,1]")
	route.add_argument("--threshold", type=float, default=None, help="Routing threshold override")
	route.add_argument("--max-depth", type=int, default=None, help="Max recursion depth override")

	return parser


def main(argv: List[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	config = load_config(args.config)
	command = args.command or "demo"

	if command == "demo":
		print(intro_message(config, getattr(args, "name", None)))
		return 0

	if command == "route":
		scores = args.scores or [0.15, 0.35, 0.6, 0.9]
		threshold = args.threshold if args.threshold is not None else config.routing_threshold
		max_depth = args.max_depth if args.max_depth is not None else config.max_recursion_depth
		assignments = assign_experts_and_recursions(scores, threshold, max_depth)
		for i, (expert_id, depth) in enumerate(assignments):
			print(f"score={scores[i]:.2f} -> expert={expert_id} depth={depth}")
		return 0

	parser.print_help()
	return 1