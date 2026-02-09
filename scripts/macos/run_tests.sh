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

python -m compileall "${REPO_DIR}/src"

if command -v pytest >/dev/null 2>&1; then
  pytest
else
  echo "pytest not installed (install requirements-dev.txt to run unit tests)."
fi

echo "OK"
