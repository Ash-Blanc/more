"""Causal Transformer implementation of the text-only MoRE hypothesis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import LossConfig, ModelConfig
from .routing import (
    DepthRouter,
    ExpertRouting,
    TopKRouter,
    load_balance_loss,
    router_z_loss,
)


@dataclass
class MoREOutput:
    logits: Tensor
    loss: Tensor | None
    lm_loss: Tensor | None
    aux_loss: Tensor
    depth_balance_loss: Tensor
    moe_balance_loss: Tensor
    router_z_loss: Tensor
    depth_assignments: Tensor
    depth_active_mask: Tensor
    routing_stats: dict[str, Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            if attention_mask is None:
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
                allowed = causal.view(1, 1, length, length)
                allowed = allowed & attention_mask[:, None, None, :].bool()
                allowed = allowed & attention_mask[:, None, :, None].bool()
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=allowed,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:  # pragma: no cover - exercised only on old PyTorch releases
            scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            allowed = causal.view(1, 1, length, length)
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()
                allowed = allowed & attention_mask[:, None, :, None].bool()
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=-1)
            attended = weights @ value

        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class DenseFFN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_width = config.d_model * config.ffn_multiplier
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, config.d_model),
        )

    def forward(self, hidden: Tensor) -> tuple[Tensor, None]:
        return self.net(hidden), None


class MoEFFN(nn.Module):
    """Simple top-k dispatch that favors correctness over kernel efficiency."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_width = config.d_model * config.ffn_multiplier
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.d_model, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, config.d_model),
            )
            for _ in range(config.n_experts)
        )
        self.router = TopKRouter(
            config.d_model,
            config.n_experts,
            config.moe_top_k,
            config.router_temperature,
        )

    def forward(self, hidden: Tensor) -> tuple[Tensor, ExpertRouting]:
        routing = self.router(hidden)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_gates = routing.gates.reshape(-1, routing.gates.shape[-1])
        output = torch.zeros_like(flat_hidden)
        for expert_id, expert in enumerate(self.experts):
            gate = flat_gates[:, expert_id]
            selected = gate > 0
            if selected.any():
                output[selected] += expert(flat_hidden[selected]) * gate[selected, None]
        return output.view_as(hidden), routing


class SharedTransformerBlock(nn.Module):
    """One block whose parameters are reused by every recursive pass."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_ffn = nn.LayerNorm(config.d_model)
        self.ffn: nn.Module = MoEFFN(config) if config.use_moe else DenseFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, ExpertRouting | None]:
        hidden = hidden + self.dropout(self.attention(self.norm_attn(hidden), attention_mask))
        ffn_output, routing = self.ffn(self.norm_ffn(hidden))  # type: ignore[misc]
        hidden = hidden + self.dropout(ffn_output)
        return hidden, routing


class MoREModel(nn.Module):
    """A compact causal LM combining shared recursion and sparse FFN experts.

    The first version computes the shared block densely and masks residual
    updates for inactive tokens. This is semantically correct but not yet a
    wall-clock speedup; packed token dispatch is a follow-up optimization.
    """

    def __init__(
        self, config: ModelConfig | None = None, loss_config: LossConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.loss_config = loss_config or LossConfig()
        self.n_recursions = self.config.n_recursions if self.config.use_mor else 1
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.depth_router = DepthRouter(
            self.config.d_model,
            self.n_recursions,
            self.config.router_temperature,
        )
        self.shared_block = SharedTransformerBlock(self.config)
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.logit_scale = self.config.d_model**-0.5

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        return sum(p.numel() for p in parameters if not trainable_only or p.requires_grad)

    def _language_model_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=256
        )

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> MoREOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if input_ids.dtype not in (torch.int32, torch.int64):
            input_ids = input_ids.long()
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)

        depth = self.depth_router(hidden)
        expert_routes: list[ExpertRouting] = []
        expert_balance_terms: list[Tensor] = []
        expert_z_terms: list[Tensor] = []
        valid_mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        for recursion in range(self.n_recursions):
            block_output, expert_route = self.shared_block(hidden, attention_mask)
            gate = depth.straight_through_active[..., recursion, None]
            hidden = hidden + gate * (block_output - hidden)
            if expert_route is not None:
                expert_routes.append(expert_route)
                active = depth.hard_active[..., recursion] & valid_mask
                top1 = expert_route.probabilities.argmax(dim=-1)
                expert_balance_terms.append(
                    load_balance_loss(expert_route.probabilities, top1, active)
                )
                expert_z_terms.append(router_z_loss(expert_route.logits, active))

        logits = self.lm_head(self.final_norm(hidden)) * self.logit_scale
        lm_loss = self._language_model_loss(logits, labels) if labels is not None else None
        depth_balance = load_balance_loss(depth.probabilities, depth.assignments, valid_mask)
        depth_z = router_z_loss(depth.logits, valid_mask)
        moe_balance = (
            torch.stack(expert_balance_terms).mean()
            if expert_balance_terms
            else logits.new_zeros(())
        )
        moe_z = torch.stack(expert_z_terms).mean() if expert_z_terms else logits.new_zeros(())
        z_loss = depth_z + moe_z
        target_depth = self.loss_config.target_mean_depth or (self.n_recursions + 1) / 2
        mean_depth = (depth.assignments.float() + 1.0)[valid_mask].mean()
        compute_penalty = (mean_depth - target_depth).square()
        aux_loss = (
            self.loss_config.depth_balance_weight * depth_balance
            + self.loss_config.moe_balance_weight * moe_balance
            + self.loss_config.router_z_weight * z_loss
            + self.loss_config.compute_penalty_weight * compute_penalty
        )
        total_loss = lm_loss + aux_loss if lm_loss is not None else None
        depth_fraction = depth.hard_active.to(logits.dtype)[valid_mask].mean(dim=0)
        stats: dict[str, Tensor] = {
            "mean_depth": mean_depth.detach(),
            "depth_fraction": depth_fraction.detach(),
            "depth_entropy": (
                -(depth.probabilities * depth.probabilities.clamp_min(1e-8).log()).sum(-1)
            )[valid_mask]
            .mean()
            .detach(),
        }
        if expert_routes:
            expert_prob = torch.stack([route.probabilities for route in expert_routes])
            stats["expert_fraction"] = (expert_prob.mean(dim=(0, 1, 2))).detach()
            stats["expert_entropy"] = (
                (-(expert_prob * expert_prob.clamp_min(1e-8).log()).sum(-1)).mean().detach()
            )
        else:
            stats["expert_fraction"] = logits.new_zeros(self.config.n_experts)
            stats["expert_entropy"] = logits.new_zeros(())
        return MoREOutput(
            logits=logits,
            loss=total_loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            depth_balance_loss=depth_balance,
            moe_balance_loss=moe_balance,
            router_z_loss=z_loss,
            depth_assignments=depth.assignments,
            depth_active_mask=depth.hard_active,
            routing_stats=stats,
        )
