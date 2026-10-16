# MoRE: text generation with adaptive computation

MoRE is a small, runnable research project for testing a specific hypothesis:
combine **Mixture of Experts** (specialized feed-forward transformations) with
**Mixture of Recursions** (a learned token-level recursion budget) in a causal
language model. The main branch is text-generation/reasoning focused. Video
generation is intentionally separated; see [`docs/video-separation.md`](docs/video-separation.md).

This is a correctness-first prototype, not a claim of state-of-the-art quality
or speed. The first implementation computes masked recursive updates densely.
Packed active-token kernels, KV caching, and large-scale benchmarks are future
work and must be measured before claiming efficiency gains.

## Install

Python 3.10+ is recommended. With `uv`:

```bash
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

Or with pip:

```bash
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch wheel, follow the install command for your platform at
[pytorch.org](https://pytorch.org/get-started/locally/) and then install the
remaining requirements.

## 30-second workflow

The bundled corpus is deliberately tiny so the complete workflow can run on a
laptop:

```bash
# Preserve the original scaffold smoke tests.
python -m more demo --name Ada
python -m more route --scores 0.2 0.5 0.8 0.95

# Train a tiny MoRE language model.
python -m more train --config config.yaml --data data/tiny.txt --out runs/tiny

# Evaluate the saved best checkpoint.
python -m more eval --checkpoint runs/tiny/best.pt --data data/tiny.txt

# Generate a continuation (temperature 0 is greedy).
python -m more generate --checkpoint runs/tiny/best.pt --prompt "MoRE" --max-new-tokens 32
```

The train command prints JSON metrics containing language-model loss,
perplexity, routing losses, mean depth, and routing entropy. Checkpoints store
the resolved model configuration and optimizer state, so experiments are
reproducible.

## Architecture

The model is a decoder-only causal Transformer:

1. UTF-8 bytes plus learned positions enter the model.
2. A token-choice depth router selects a maximum depth for each token.
3. One shared Transformer block is recursively reused for that many passes.
4. The block's feed-forward sublayer uses top-k expert routing when MoE is on.
5. A tied language-model head predicts the next byte.

Depth masks are hierarchical: a token assigned depth `d` is active on the
first `d` passes and never returns after it halts. A straight-through
cumulative soft gate keeps the depth router trainable. The objective combines
causal language-model loss with depth/expert load balance, router z-loss, and
an optional compute-budget penalty.

Read the detailed design and limitations in [`docs/architecture.md`](docs/architecture.md).
The implementation is inspired by [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
and is intentionally explicit about where it differs from the paper and from
production sparse-MoE systems.

## Ablations

Run the same corpus, seed, and training budget with:

```bash
python -m more train --config configs/dense.yaml --data data/tiny.txt --out runs/dense
python -m more train --config configs/mor.yaml   --data data/tiny.txt --out runs/mor
python -m more train --config configs/moe.yaml   --data data/tiny.txt --out runs/moe
python -m more train --config config.yaml         --data data/tiny.txt --out runs/more
```

Compare validation perplexity, parameter count, mean active depth, expert
utilization, and tokens/sec. A higher routing depth or entropy is not itself
evidence of better reasoning; add held-out reasoning tasks before drawing
that conclusion.

## Python API

```python
import torch

from more.config import ModelConfig
from more.model import MoREModel

model = MoREModel(ModelConfig(max_seq_len=32, d_model=32, n_heads=4))
input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
targets = input_ids.roll(-1, dims=1)
output = model(input_ids, labels=targets)
print(output.loss, output.routing_stats["mean_depth"])
```

The original toy API remains available for compatibility:

```python
from more.core import assign_experts_and_recursions, load_config

cfg = load_config("config.yaml")
print(
    assign_experts_and_recursions([0.2, 0.5, 0.8], cfg.routing_threshold, cfg.max_recursion_depth)
)
```

That scalar function is a baseline demonstration, not the learned neural
router.

## Layout

```text
more/
  config.py       validated model/loss/training settings
  core.py         compatibility config and toy routing API
  data.py         UTF-8 byte tokenizer and causal chunks
  routing.py      depth/top-k routers and auxiliary losses
  model.py        shared causal block, MoE FFN, recursive LM
  generate.py     greedy/top-k generation
  train.py        training, evaluation, checkpoints
  cli.py          demo/route/train/eval/generate commands
configs/          dense, MoR-only, and MoE-only ablations
data/tiny.txt    bundled smoke-test corpus
docs/             architecture and video boundary
tests/            CPU correctness and CLI tests
```

## Status and next experiments

- [x] causal text model with shared recursive block
- [x] token-choice depth routing and top-k expert routing
- [x] routing auxiliary losses and metrics
- [x] deterministic tiny train/eval/generate workflow
- [x] dense/MoR/MoE/MoRE configuration ablations
- [ ] packed active-token dispatch and real compute benchmarks
- [ ] KV-cache support for recursive generation
- [ ] held-out reasoning-task evaluation
- [ ] separate `videogen` project and adapter

Contributions should preserve the text/video boundary and add tests for any
new routing or cache semantics. See `CONTRIBUTING.md`.
