#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/myenv"
PYTHON_PKG="python-3.12.7-macos11.pkg"
PYTHON_URL="https://www.python.org/ftp/python/3.12.7/$PYTHON_PKG"
PYTHON_BIN="/usr/local/bin/python3.12"

echo "==> Removing old virtual environment..."
rm -rf "$VENV_DIR"

echo "==> Checking for Python 3.12 (python.org)..."
if [ ! -f "$PYTHON_BIN" ]; then
    echo "    Downloading Python 3.12 from python.org..."
    curl -O "$PYTHON_URL"
    echo "    Installing Python 3.12 (you may be prompted for your password)..."
    sudo installer -pkg "$PYTHON_PKG" -target /
    rm -f "$PYTHON_PKG"
fi

echo "==> Creating virtual environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing packages..."
pip install torch
pip install tyro
pip install "gymnasium[atari]"
pip install tensorboard
pip install opencv-python
pip install "autorom[accept-rom-license]"

echo "==> Installing Atari ROMs..."
autorom

echo "==> Setting KMP_DUPLICATE_LIB_OK in activate script..."
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> "$VENV_DIR/bin/activate"

echo ""
echo "✓ Done! Activate your environment with:"
echo "  source myenv/bin/activate"
