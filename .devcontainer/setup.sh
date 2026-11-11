#!/usr/bin/env bash
# Prepare a Codespace to run Click. Everything here is free and local to the
# container: no account, no key and no paid service is involved.
set -euo pipefail

echo "Python in this container: $(python3.11 --version)"

# espeak-ng gives spoken playback on Linux; ffmpeg decodes uploaded recordings.
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends espeak-ng ffmpeg

# Build the same environment start.py expects, so the first launch is quick.
python3.11 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
# CPU wheels only: a Codespace has no GPU and the CUDA packages are enormous.
.venv/bin/python -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
.venv/bin/python -m pip install --quiet -r requirements.txt

echo
echo "Click is ready. Start it with:   python3.11 start.py --no-browser"
echo "Then open the forwarded port 7860 from the Ports tab."
