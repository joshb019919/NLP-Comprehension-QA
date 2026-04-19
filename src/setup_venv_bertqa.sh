#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/src/qa_hf_package/requirements.txt"
VENV_DIR="$SCRIPT_DIR/.venv_bertqa"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements file not found at $REQ_FILE" >&2
  exit 1
fi

if [[ "$(uname -s 2>/dev/null || true)" != "Linux" ]]; then
  echo "Error: this Bash script is Linux-only. Use setup_venv_bertqa.bat on Windows." >&2
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Error: Python interpreter not found in PATH." >&2
  exit 1
fi

echo "Detected OS: linux"
echo "Using interpreter: $PYTHON_BIN"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
  echo "Error: activation script not found at $ACTIVATE_SCRIPT" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

echo "Installing pinned requirements from $REQ_FILE"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo
echo "Setup complete."
echo "Virtual environment directory: $VENV_DIR"
echo "Requirements installed from: $REQ_FILE"
echo
echo "To activate manually in a new shell:"
echo "  source .venv_bertqa/bin/activate"
