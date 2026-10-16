"""Small, dependency-light byte data pipeline for causal language modeling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """UTF-8 byte tokenizer with three reserved special tokens."""

    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    vocab_size = 259

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS_ID)
        tokens.extend(text.encode("utf-8"))
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(self, tokens: Sequence[int], *, stop_at_eos: bool = True) -> str:
        output = bytearray()
        for token in tokens:
            token = int(token)
            if token == self.EOS_ID and stop_at_eos:
                break
            if token < 256:
                output.append(token)
        return bytes(output).decode("utf-8", errors="replace")


class TextChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Contiguous fixed-length next-token examples."""

    def __init__(
        self, token_ids: Sequence[int], block_size: int, stride: int | None = None
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.tokens = torch.tensor(list(token_ids), dtype=torch.long)
        self.block_size = block_size
        self.stride = stride or block_size
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.tokens.numel() < block_size + 1:
            raise ValueError("corpus must contain at least block_size + 1 tokens")

    def __len__(self) -> int:
        return 1 + (self.tokens.numel() - self.block_size - 1) // self.stride

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_token_ids(path: str | Path, tokenizer: ByteTokenizer | None = None) -> list[int]:
    tokenizer = tokenizer or ByteTokenizer()
    return tokenizer.encode(Path(path).read_text(encoding="utf-8"))


def split_token_ids(
    token_ids: Sequence[int], train_fraction: float = 0.9
) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = max(1, min(len(token_ids) - 1, int(len(token_ids) * train_fraction)))
    return list(token_ids[:split]), list(token_ids[split:])


def make_datasets(
    path: str | Path,
    block_size: int,
    train_fraction: float = 0.9,
    tokenizer: ByteTokenizer | None = None,
) -> tuple[TextChunkDataset, TextChunkDataset, ByteTokenizer]:
    tokenizer = tokenizer or ByteTokenizer()
    train_ids, valid_ids = split_token_ids(load_token_ids(path, tokenizer), train_fraction)
    # Give validation a little context when the split lands exactly at a boundary.
    if len(valid_ids) < block_size + 1:
        valid_ids = train_ids[-(block_size + 1) :] + valid_ids
    return (
        TextChunkDataset(train_ids, block_size),
        TextChunkDataset(valid_ids, block_size),
        tokenizer,
    )
