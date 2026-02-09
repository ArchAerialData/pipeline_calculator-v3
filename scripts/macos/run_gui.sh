#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Venv not found. Run: bash scripts/macos/setup_macos.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
try:
    import tkinter as tk
except Exception as e:
    print("ERROR: tkinter is not available in this environment.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("Fix: run `bash scripts/macos/setup_macos.sh` to install Homebrew python + Tk bindings.", file=sys.stderr)
    raise SystemExit(2)

ver = float(getattr(tk, "TkVersion", 0.0))
if ver < 8.6:
    print(f"ERROR: Tcl/Tk {ver} detected. Tk 8.5 aborts on macOS 26 when creating a window.", file=sys.stderr)
    print("Fix: run `bash scripts/macos/setup_macos.sh` (it installs python@3.11 + python-tk@3.11).", file=sys.stderr)
    raise SystemExit(2)
PY

python "${REPO_DIR}/src/pipeline_calculator_v3.py"
