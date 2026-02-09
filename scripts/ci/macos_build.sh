#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if [[ -f requirements-dev.txt ]]; then
  python -m pip install -r requirements-dev.txt
fi

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

# Basic sanity check without launching the GUI.
python -m py_compile src/pipeline_calculator_v3.py
python -m pytest

# Delegate actual PyInstaller invocation to the user-facing scripts.
bash scripts/macos/build_app.sh
