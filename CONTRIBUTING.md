# Contributing 🤝

Thanks for helping make this better! This guide is short on purpose.

## Quick local setup
```bash
# 1) Clone your fork
git clone <your-fork-url>
cd <repo>

# 2) Create a virtualenv (Python 3.9+)
python -m venv .venv
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Dev tools (optional but recommended)
pip install ruff pytest
```

## Dev rules
- **Formatting & lint** (use Ruff):
```bash
ruff format .
ruff check .
```
- **Run tests**:
```bash
pytest -q
```

## Submitting changes
1. Create a branch (`feat/<thing>` or `fix/<thing>`)
2. Make your changes with clear commits
3. Run format/lint/tests (see above)
4. Open a PR with:
   - What changed and why
   - Screenshots or logs if relevant
   - Any follow‑ups you’d like help with

That’s it. Thanks for contributing! 🌟