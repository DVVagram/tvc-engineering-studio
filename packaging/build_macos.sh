#!/usr/bin/env bash
# Build the macOS application bundle of TVC Engineering Studio.
# Run from anywhere: the script moves to the repository root itself.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
    echo "[build] creating virtual environment"
    python3 -m venv .venv
fi

echo "[build] installing runtime + packaging dependencies"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[packaging,hardware]"

echo "[build] running headless self-test"
.venv/bin/python -m tvc_studio --self-test

echo "[build] running validation suite"
.venv/bin/python -m pytest

echo "[build] packaging with PyInstaller"
.venv/bin/python -m PyInstaller --noconfirm --clean packaging/tvc_studio.spec

ZIP="dist/TVC-Engineering-Studio-0.1.0-macos.zip"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "dist/TVC Engineering Studio.app" "$ZIP"

echo
echo "[build] done: dist/TVC Engineering Studio.app"
echo "[build] shareable archive: $ZIP (unsigned; first launch needs right-click > Open)"
