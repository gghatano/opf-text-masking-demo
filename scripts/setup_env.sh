#!/usr/bin/env bash
# Idempotent environment setup (uv-based) for the OPF Japanese-PII verification.
# - Uses `uv` to provision a Python 3.12 venv (torch lacks 3.14 wheels; see
#   docs/findings-opf-cli.md). uv auto-downloads 3.12 if missing.
# - Clones openai/privacy-filter into third_party/ and installs it editable.
# Re-running is safe: existing steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

PYVER="${PYVER:-3.12}"
OPF_SRC="third_party/privacy-filter"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install from https://docs.astral.sh/uv/" >&2; exit 1; }
echo ">> uv $(uv --version)"

# 1) venv with a torch-compatible Python (idempotent; uv fetches 3.12 if needed)
if [[ ! -d .venv ]]; then
  echo ">> Creating .venv (Python $PYVER) via uv"
  uv venv --python "$PYVER" .venv
fi
# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
echo ">> Python: $(python --version)"

# 2) Clone OPF upstream (idempotent)
if [[ ! -d "$OPF_SRC" ]]; then
  echo ">> Cloning openai/privacy-filter"
  git clone --depth 1 https://github.com/openai/privacy-filter.git "$OPF_SRC"
fi

# 3) Install OPF (editable) + project deps via uv pip
echo ">> Installing OPF (editable) + deps"
uv pip install -e "$OPF_SRC"
uv pip install -r requirements.txt

# 4) Smoke test (CPU: the default CPU torch wheel is not CUDA-enabled)
echo ">> Smoke test (--device cpu)"
opf --device cpu "Alice was born on 1990-01-02." || echo "(smoke run failed — check install)"
echo ">> Done. Activate with: source .venv/Scripts/activate (Windows) | source .venv/bin/activate"
echo ">> Freeze deps when stable: uv pip freeze > requirements.txt"
