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

# Sanity check: Tk 8.5 aborts on macOS 26 when creating a window. Ensure Tk 8.6+.
python - <<'PY'
import sys
try:
    import tkinter as tk
except Exception as e:
    print("ERROR: tkinter is not available in this build environment.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    raise SystemExit(2)

ver = float(getattr(tk, "TkVersion", 0.0))
if ver < 8.6:
    print(f"ERROR: Tcl/Tk {ver} detected. Tk 8.6+ is required for macOS 26 compatibility.", file=sys.stderr)
    raise SystemExit(2)
print(f"Tk OK (TkVersion={ver})")
PY

# Basic sanity check without launching the GUI.
python -m py_compile src/pipeline_calculator_v3.py
if [[ -f "src/pipeline_calculator_entry.py" ]]; then
  python -m py_compile src/pipeline_calculator_entry.py
fi
python -m pytest -vv -rA --tb=short -o faulthandler_timeout=30

# Delegate actual PyInstaller invocation to the user-facing scripts.
bash scripts/macos/build_app.sh
