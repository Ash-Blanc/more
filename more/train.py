"""Reproducible CPU-friendly training and checkpoint helpers."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import MoREConfig
from .data import ByteTokenizer, make_datasets
from .model import MoREModel, MoREOutput


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _loader(dataset: Dataset[Any], batch_size: int, shuffle: bool) -> DataLoader[Any]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _scalar_metrics(output: MoREOutput) -> dict[str, float]:
    lm_loss = float(output.lm_loss.detach().cpu()) if output.lm_loss is not None else float("nan")
    loss = (
        float(output.loss.detach().cpu())
        if output.loss is not None
        else float(output.aux_loss.detach().cpu())
    )
    return {
        "loss": loss,
        "lm_loss": lm_loss,
        "aux_loss": float(output.aux_loss.detach().cpu()),
        "perplexity": math.exp(min(lm_loss, 20.0)) if math.isfinite(lm_loss) else float("nan"),
        "mean_depth": float(output.routing_stats["mean_depth"].cpu()),
        "depth_entropy": float(output.routing_stats["depth_entropy"].cpu()),
        "expert_entropy": float(output.routing_stats["expert_entropy"].cpu()),
    }


def evaluate(
    model: MoREModel,
    dataset: Dataset[Any],
    config: MoREConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = _loader(dataset, config.train.batch_size, shuffle=False)
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= config.train.eval_batches:
                break
            output = model(inputs.to(device), labels=targets.to(device))
            metrics = _scalar_metrics(output)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    if not count:
        raise RuntimeError("validation dataset produced no batches")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model: MoREModel,
    optimizer: torch.optim.Optimizer,
    config: MoREConfig,
    step: int,
    metrics: dict[str, float],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "step": step,
            "metrics": metrics,
            "tokenizer": {"type": "utf8-byte", "vocab_size": ByteTokenizer.vocab_size},
        },
        destination,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[MoREModel, MoREConfig, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only parameter
        checkpoint = torch.load(path, map_location=device)
    config = MoREConfig.from_dict(checkpoint["config"])
    model = MoREModel(config.model, config.loss).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, config, checkpoint


def train_model(
    data_path: str | Path,
    config: MoREConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    train_data, valid_data, _ = make_datasets(
        data_path,
        config.model.max_seq_len,
        config.train.train_split,
    )
    model = MoREModel(config.model, config.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    train_loader = _loader(train_data, config.train.batch_size, shuffle=True)
    train_iter = iter(train_loader)
    run_dir = Path(output_dir or config.train.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    best_loss = float("inf")
    last_metrics: dict[str, float] = {}

    for step in range(1, config.train.max_steps + 1):
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        model.train()
        output = model(inputs.to(device), labels=targets.to(device))
        if output.loss is None:
            raise RuntimeError("training requires a language-model loss")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if step == 1 or step % config.train.eval_interval == 0 or step == config.train.max_steps:
            last_metrics = evaluate(model, valid_data, config, device)
            last_metrics["step"] = float(step)
            print(json.dumps(last_metrics, sort_keys=True))
            save_checkpoint(run_dir / "last.pt", model, optimizer, config, step, last_metrics)
            if last_metrics["loss"] < best_loss:
                best_loss = last_metrics["loss"]
                save_checkpoint(run_dir / "best.pt", model, optimizer, config, step, last_metrics)

    return {
        "output_dir": str(run_dir),
        "metrics": last_metrics,
        "parameters": model.num_parameters(),
    }
