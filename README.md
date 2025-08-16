# LabStarter 🚀

A tiny, batteries-included Python starter so you can run experiments fast — no yak-shaving, no 2000-word paper dump. Load a config, call a clean API, or use a CLI that works on day one.

## Why you should care
- **Zero friction**: clone → install → run in under a minute
- **Practical defaults**: YAML config, CLI, examples, and a sane layout
- **Copy-paste friendly**: real snippets you can use immediately

## Install (one command)
```bash
pip install -r requirements.txt
```

- Requires Python 3.9+.

## 30‑second quickstart
- CLI:
```bash
python -m labstarter --name "Ada Lovelace"
```
- Programmatic:
```python
from labstarter.core import load_config, make_greeting

config = load_config("config.yaml")
print(make_greeting(config, name="Ada"))
```

## Real examples you can copy‑paste
- **Customize greeting via config**:
```yaml
# config.yaml
project:
  name: "LabStarter"
  default_name: "Researcher"
messages:
  greeting: "Howdy"
```
```bash
python -m labstarter --name "Grace Hopper"  # → "Howdy, Grace Hopper! 🚀 You're running LabStarter."
```

- **Use as a library**:
```python
from labstarter.core import load_config, make_greeting, compute_stats

cfg = load_config("config.yaml")
print(make_greeting(cfg, "Kim"))
print(compute_stats([1, 2, 3, 4, 5]))  # {"count": 5, "mean": 3.0, "median": 3, "stdev": 1.58}
```

- **Run the provided example script**:
```bash
python examples/quickstart.py
```

## Project layout
```text
.
├── labstarter/
│   ├── __init__.py
│   ├── __main__.py        # enables `python -m labstarter`
│   ├── cli.py             # argparse CLI entrypoint
│   └── core.py            # core API: load_config, make_greeting, compute_stats
├── examples/
│   └── quickstart.py      # working example
├── config.yaml            # tweak defaults here
├── requirements.txt       # minimal runtime deps (PyYAML)
├── CONTRIBUTING.md        # short & sweet contributor guide
└── README.md
```

## What this gives you
- **Clean CLI** with sensible flags
- **YAML config** that loads by default
- **Importable API** for notebooks and scripts

If this helps you move faster, consider ⭐️ the repo and sharing improvements via a PR!
