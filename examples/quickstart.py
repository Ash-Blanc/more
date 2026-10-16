"""Run the compatibility demo and inspect a tiny text MoRE model."""

from __future__ import annotations

import torch

from more.config import MoREConfig
from more.data import ByteTokenizer
from more.model import MoREModel


def run() -> None:
    config = MoREConfig.from_dict(
        {
            "model": {
                "vocab_size": ByteTokenizer.vocab_size,
                "max_seq_len": 32,
                "d_model": 32,
                "n_heads": 4,
                "n_recursions": 2,
                "n_experts": 2,
                "moe_top_k": 1,
                "ffn_multiplier": 2,
            }
        }
    )
    model = MoREModel(config.model, config.loss)
    tokens = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(tokens, labels=tokens.roll(-1, dims=1))
    print(f"parameters={model.num_parameters():,}")
    print(f"logits={tuple(output.logits.shape)} loss={output.loss.item():.3f}")
    print(f"mean_depth={output.routing_stats['mean_depth'].item():.2f}")


if __name__ == "__main__":
    run()
