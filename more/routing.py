"""Token-choice depth and expert routers used by the MoRE model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def router_z_loss(logits: Tensor, mask: Tensor | None = None) -> Tensor:
    """Penalize very large router logits for numerical stability."""

    return _masked_mean(torch.logsumexp(logits, dim=-1).square(), mask)


def load_balance_loss(
    probabilities: Tensor,
    assignments: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Switch-style soft/hard load balancing objective.

    ``probabilities`` is ``[..., choices]`` and ``assignments`` is the hard
    top-1 choice with the same leading dimensions.
    """

    choices = probabilities.shape[-1]
    if mask is None:
        mask = torch.ones_like(assignments, dtype=torch.bool)
    valid_probs = probabilities[mask]
    valid_assignments = assignments[mask]
    if valid_probs.numel() == 0:
        return probabilities.new_zeros(())
    mean_probability = valid_probs.mean(dim=0)
    frequency = torch.nn.functional.one_hot(valid_assignments, choices).float().mean(dim=0)
    return choices * (mean_probability * frequency).sum()


def hierarchical_active_mask(assignments: Tensor, n_recursions: int) -> Tensor:
    """Return ``[..., recursion]`` mask for paths active through each pass."""

    if assignments.min().item() < 0 or assignments.max().item() >= n_recursions:
        raise ValueError("depth assignments are outside the recursion range")
    steps = torch.arange(n_recursions, device=assignments.device)
    return steps.view(*([1] * assignments.ndim), n_recursions) <= assignments.unsqueeze(-1)


@dataclass
class DepthRouting:
    logits: Tensor
    probabilities: Tensor
    assignments: Tensor
    hard_active: Tensor
    soft_active: Tensor
    straight_through_active: Tensor


class DepthRouter(nn.Module):
    """Choose a maximum recursive depth once per token."""

    def __init__(self, d_model: int, n_recursions: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, n_recursions)
        self.n_recursions = n_recursions
        self.temperature = temperature

    def forward(self, hidden: Tensor) -> DepthRouting:
        logits = self.projection(hidden)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        assignments = probabilities.argmax(dim=-1)
        hard_active = hierarchical_active_mask(assignments, self.n_recursions)
        soft_active = probabilities.flip(-1).cumsum(-1).flip(-1)
        straight_through = hard_active.to(hidden.dtype) + soft_active - soft_active.detach()
        return DepthRouting(
            logits=logits,
            probabilities=probabilities,
            assignments=assignments,
            hard_active=hard_active,
            soft_active=soft_active,
            straight_through_active=straight_through,
        )


@dataclass
class ExpertRouting:
    logits: Tensor
    probabilities: Tensor
    indices: Tensor
    gates: Tensor


class TopKRouter(nn.Module):
    """Select and renormalize the top-k experts for each token."""

    def __init__(
        self, d_model: int, n_experts: int, top_k: int = 2, temperature: float = 1.0
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.projection = nn.Linear(d_model, n_experts)
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, hidden: Tensor) -> ExpertRouting:
        logits = self.projection(hidden)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        values, indices = probabilities.topk(self.top_k, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        gates = torch.zeros_like(probabilities).scatter(-1, indices, values)
        return ExpertRouting(
            logits=logits, probabilities=probabilities, indices=indices, gates=gates
        )
