"""Autoregressive generation utilities for :class:`more.model.MoREModel`."""

from __future__ import annotations

import torch
from torch import Tensor

from .data import ByteTokenizer
from .model import MoREModel


def sample_next_token(logits: Tensor, temperature: float = 0.0, top_k: int | None = None) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


def generate_tokens(
    model: MoREModel,
    input_ids: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = ByteTokenizer.EOS_ID,
) -> Tensor:
    """Generate by recomputing the bounded prefix at each step.

    This is deliberately simple and correct. A KV-cache implementation should
    be added only after recursive routing semantics are benchmarked.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -model.config.max_seq_len :]
            output = model(context)
            next_token = sample_next_token(output.logits[:, -1, :], temperature, top_k)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_id is not None and torch.all(next_token.squeeze(-1).eq(eos_id)):
                break
    if was_training:
        model.train()
    return generated


def generate_text(
    model: MoREModel,
    prompt: str,
    max_new_tokens: int,
    tokenizer: ByteTokenizer | None = None,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> str:
    tokenizer = tokenizer or ByteTokenizer()
    device = next(model.parameters()).device
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True, add_eos=False)], device=device
    )
    generated = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.EOS_ID,
    )
    return tokenizer.decode(generated[0].tolist())
