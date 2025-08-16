# MoRE: Mixture of Recursions + Experts 🚀

A fast, copy‑paste‑friendly starter to explore MoRE ideas: route tokens to **experts** and apply **adaptive recursion** based on importance. Use a clean CLI, YAML config, and runnable examples to prototype quickly.

## Why you should care
- **Research-ready**: minimal but opinionated scaffolding for MoRE experiments
- **Runs in 30 seconds**: config + CLI + example, no extra glue code
- **Copy‑paste first**: real commands and snippets below

## Install (one command)
```bash
pip install -r requirements.txt
```

- Requires Python 3.9+.

## 30‑second quickstart
- CLI (module):
```bash
python -m more demo --name "Ada"
```
- Routing demo:
```bash
python -m more route --scores 0.2 0.5 0.8 0.95 --threshold 0.5 --max-depth 4
```
- Programmatic:
```python
from more.core import load_config, intro_message, assign_experts_and_recursions

cfg = load_config("config.yaml")
print(intro_message(cfg, name="Ada"))
print(assign_experts_and_recursions([0.2, 0.5, 0.8, 0.95], cfg.routing_threshold, cfg.max_recursion_depth))
```

## Real examples you can copy‑paste
- **Config knobs**:
```yaml
# config.yaml
project:
  name: "MoRE"
  default_name: "Researcher"
messages:
  greeting: "Hello"
routing:
  threshold: 0.5
  max_depth: 4
```
```bash
python -m more route --scores 0.1 0.3 0.7 0.9
# → score=0.10 -> expert=0 depth=1
#   score=0.30 -> expert=1 depth=2
#   score=0.70 -> expert=2 depth=3
#   score=0.90 -> expert=3 depth=4
```

- **Run the example script**:
```bash
python examples/quickstart.py
```

## Project layout
```text
.
├── more/
│   ├── __init__.py
│   ├── __main__.py        # enables `python -m more`
│   ├── cli.py             # argparse CLI with demo + route
│   └── core.py            # config, intro_message, toy expert/recursion routing
├── examples/
│   └── quickstart.py      # MoRE demo example
├── config.yaml            # routing/defaults
├── requirements.txt       # minimal runtime deps (PyYAML)
├── CONTRIBUTING.md        # short contributor guide
└── README.md
```

## What this gives you
- **MoRE‑themed CLI** to kickstart routing/recursion experiments
- **YAML config** for thresholds and depth
- **Importable API** for notebooks and benchmarking

If this saves you time, ⭐️ the repo and send a PR with your improvements!
