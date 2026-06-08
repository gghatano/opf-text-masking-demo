#!/usr/bin/env bash
# Idempotent environment setup for the OPF Japanese-PII verification.
# - Creates a Python 3.11/3.12 venv (torch lacks 3.14 wheels; see docs/findings-opf-cli.md)
# - Clones openai/privacy-filter into third_party/ and installs it editable
# Re-running is safe: existing steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

OPF_SRC="third_party/privacy-filter"

# 1) Pick a compatible Python (3.12 > 3.11). Override with PYTHON env var.
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; return; fi
  for c in python3.12 python3.11 py; do
    if command -v "$c" >/dev/null 2>&1; then
      if [[ "$c" == "py" ]]; then
        if py -3.12 --version >/dev/null 2>&1; then echo "py -3.12"; return; fi
        if py -3.11 --version >/dev/null 2>&1; then echo "py -3.11"; return; fi
      else echo "$c"; return; fi
    fi
  done
  echo ""  # none found
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  echo "ERROR: Python 3.11/3.12 not found. torch has no 3.14 wheel yet." >&2
  echo "Install Python 3.12 (or set PYTHON=...) and re-run." >&2
  exit 1
fi
echo ">> Using interpreter: $PY"

# 2) venv (idempotent)
if [[ ! -d .venv ]]; then
  echo ">> Creating .venv"
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
python -m pip install -q --upgrade pip

# 3) Clone + install OPF (idempotent)
if [[ ! -d "$OPF_SRC" ]]; then
  echo ">> Cloning openai/privacy-filter"
  git clone --depth 1 https://github.com/openai/privacy-filter.git "$OPF_SRC"
fi
echo ">> Installing OPF (editable)"
python -m pip install -q -e "$OPF_SRC"

# 4) Project deps (eval/plot/compare). GiNZA etc. installed on demand by their scripts.
python -m pip install -q -r requirements.txt || true

echo ">> Smoke test"
opf "Alice was born on 1990-01-02." || echo "(smoke run failed — check install)"
echo ">> Done. Activate with: source .venv/Scripts/activate (Windows) | source .venv/bin/activate"
