
#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/myenv"

echo "==> Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv ffmpeg

echo "==> Removing old virtual environment..."
rm -rf "$VENV_DIR"

echo "==> Creating virtual environment..."
python3.12 -m venv "$VENV_DIR"
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

echo ""
echo "Done! Activate with: source myenv/bin/activate"