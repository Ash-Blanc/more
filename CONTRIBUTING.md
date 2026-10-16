# Contributing

MoRE is a text-generation/reasoning research prototype. Keep the main package
text-only and preserve the boundary described in `docs/video-separation.md`.
Do not add video/media dependencies to `more`.

## Local setup

```bash
git clone <your-fork-url>
cd more
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

For a CPU-only PyTorch wheel, install the wheel recommended for your platform
at [pytorch.org](https://pytorch.org/get-started/locally/) before running the
commands above.

## Verification

Run the complete local checks before opening a pull request:

```bash
ruff format .
ruff check .
pytest -q
python -m more train --config config.yaml --data data/tiny.txt --out /tmp/more-run --steps 1
python -m more eval --checkpoint /tmp/more-run/best.pt --data data/tiny.txt
```

Model changes should add or update tests for causal masking, recursive depth
semantics, expert top-k behavior, router gradients, checkpoint round trips, or
generation behavior as appropriate. Do not claim compute savings from a mask;
include a benchmark when changing packed dispatch or KV-cache code.

## Submitting changes

1. Create a focused `feat/<thing>` or `fix/<thing>` branch.
2. Explain the hypothesis or behavior change and its expected measurement.
3. Include tests and note any CPU/GPU or PyTorch-version assumptions.
4. Run formatting, lint, tests, and the tiny end-to-end workflow.
5. Open a pull request with remaining limitations and follow-up experiments.
