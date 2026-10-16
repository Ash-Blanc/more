"""Command-line entry points for demos and text-model experiments."""

from __future__ import annotations

import utils.session
import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .core import (
    DEFAULT_CONFIG_PATH,
    assign_experts_and_recursions,
    intro_message,
    load_config,
    load_experiment_config,
)
from .data import make_datasets
from .generate import generate_text
from .train import evaluate, load_checkpoint, resolve_device, train_model


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="more",
        description="MoRE: a text-only Mixture of Recursions + Experts research prototype",
    )
    _config_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=False)

    demo = subparsers.add_parser("demo", help="Print a short MoRE intro message")
    _config_argument(demo)
    demo.add_argument("--name", type=str, default=None, help="Name to greet")

    route = subparsers.add_parser("route", help="Run the legacy scalar routing demo")
    _config_argument(route)
    route.add_argument("--scores", nargs="*", type=float, help="Importance scores in [0,1]")
    route.add_argument("--threshold", type=float, default=None, help="Routing threshold override")
    route.add_argument("--max-depth", type=int, default=None, help="Max recursion depth override")

    train = subparsers.add_parser("train", help="Train a small causal model on a UTF-8 text file")
    _config_argument(train)
    train.add_argument("--data", type=Path, required=True, help="Training text file")
    train.add_argument("--out", type=Path, default=None, help="Checkpoint directory override")
    train.add_argument("--steps", type=int, default=None, help="Maximum training steps override")
    train.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto")

    evaluate_parser = subparsers.add_parser("eval", help="Evaluate a saved checkpoint")
    _config_argument(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--device", type=str, default="auto")
    evaluate_parser.add_argument("--batches", type=int, default=None)

    generate = subparsers.add_parser("generate", help="Generate text from a saved checkpoint")
    _config_argument(generate)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", type=str, required=True)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--device", type=str, default="auto")

    return parser


def _experiment_config(args: argparse.Namespace):
    config = load_experiment_config(args.config)
    overrides = {}
    if getattr(args, "steps", None) is not None:
        overrides["train.max_steps"] = args.steps
    if getattr(args, "device", None) is not None and args.command == "train":
        overrides["train.device"] = args.device
    if overrides:
        config = config.with_overrides(**overrides)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", DEFAULT_CONFIG_PATH)
    command = args.command or "demo"

    if command == "demo":
        print(intro_message(load_config(config_path), getattr(args, "name", None)))
        return 0

    if command == "route":
        config = load_config(config_path)
        scores = args.scores or [0.15, 0.35, 0.6, 0.9]
        threshold = args.threshold if args.threshold is not None else config.routing_threshold
        max_depth = args.max_depth if args.max_depth is not None else config.max_recursion_depth
        assignments = assign_experts_and_recursions(scores, threshold, max_depth)
        for score, (expert_id, depth) in zip(scores, assignments):
            print(f"score={score:.2f} -> expert={expert_id} depth={depth}")
        return 0

    if command == "train":
        result = train_model(args.data, _experiment_config(args), args.out)
        print(json.dumps(result, sort_keys=True))
        return 0

    if command == "eval":
        device = resolve_device(args.device)
        model, config, _ = load_checkpoint(args.checkpoint, device)
        if args.batches is not None:
            config = config.with_overrides(**{"train.eval_batches": args.batches})
        _, valid_data, _ = make_datasets(
            args.data,
            config.model.max_seq_len,
            config.train.train_split,
        )
        print(json.dumps(evaluate(model, valid_data, config, device), sort_keys=True))
        return 0

    if command == "generate":
        device = resolve_device(args.device)
        model, _, _ = load_checkpoint(args.checkpoint, device)
        print(
            generate_text(
                model,
                args.prompt,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )
        return 0

    parser.print_help()
    return 1
