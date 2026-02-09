#!/usr/bin/env bash
set -euo pipefail

# Pipeline Calculator v3 macOS setup script
# - Installs Homebrew (if missing)
# - Installs Python 3.11 via brew
# - Creates repo-local venv at .venv/ and installs requirements
#
# Run:
#   bash scripts/macos/setup_macos.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

echo "Script: ${SCRIPT_DIR}"
echo "Repo: ${REPO_DIR}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is intended for macOS."
  exit 1
fi

if [[ ! -f "${REPO_DIR}/requirements.txt" ]] || [[ ! -f "${REPO_DIR}/src/pipeline_calculator_v3.py" ]]; then
  echo "Could not locate repo root from ${SCRIPT_DIR}."
  echo "Expected: ${REPO_DIR}/requirements.txt and ${REPO_DIR}/src/pipeline_calculator_v3.py"
  exit 1
fi

BREW_BIN=""
if command -v brew >/dev/null 2>&1; then
  BREW_BIN="$(command -v brew)"
elif [[ -x "/opt/homebrew/bin/brew" ]]; then
  BREW_BIN="/opt/homebrew/bin/brew"
elif [[ -x "/usr/local/bin/brew" ]]; then
  BREW_BIN="/usr/local/bin/brew"
fi

if [[ -z "${BREW_BIN}" ]]; then
  echo "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x "/opt/homebrew/bin/brew" ]]; then
    BREW_BIN="/opt/homebrew/bin/brew"
  elif [[ -x "/usr/local/bin/brew" ]]; then
    BREW_BIN="/usr/local/bin/brew"
  fi
fi

if [[ -z "${BREW_BIN}" ]]; then
  echo "Homebrew install completed but brew is still not available."
  exit 1
fi

BREW_PREFIX="$("${BREW_BIN}" --prefix)"
if [[ -d "${BREW_PREFIX}/bin" && ":${PATH}:" != *":${BREW_PREFIX}/bin:"* ]]; then
  export PATH="${BREW_PREFIX}/bin:${PATH}"
fi

echo "Updating brew..."
"${BREW_BIN}" update

if ! "${BREW_BIN}" list python@3.11 >/dev/null 2>&1; then
  echo "Installing python@3.11..."
  "${BREW_BIN}" install python@3.11
fi

PY_BIN="$("${BREW_BIN}" --prefix python@3.11)/bin/python3.11"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "python@3.11 not found at ${PY_BIN}."
  exit 1
fi
echo "Using Python: ${PY_BIN}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR} ..."
  "${PY_BIN}" -m venv "${VENV_DIR}"
else
  echo "Using existing venv at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${REPO_DIR}/requirements.txt"
if [[ -f "${REPO_DIR}/requirements-dev.txt" ]]; then
  python -m pip install -r "${REPO_DIR}/requirements-dev.txt"
fi

echo "Verifying installs..."
python --version
python -c "import tkinter; print('tkinter OK')"
python -c "import customtkinter; print('customtkinter OK')"
python -c "import tkinterdnd2; print('tkinterdnd2 OK')"
python -c "import numpy, pandas, scipy, pyproj; print('scientific stack OK')"

echo "Setup complete."
echo "Next: bash scripts/macos/run_gui.sh"
