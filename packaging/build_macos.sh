#!/usr/bin/env bash
# Build the macOS application bundle of Static Rocket Flight Simulator.
# Run from anywhere: the script moves to the repository root itself.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
    echo "[build] creating virtual environment"
    python3 -m venv .venv
fi

echo "[build] installing runtime + packaging dependencies"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[packaging]"

echo "[build] running headless self-test"
.venv/bin/python -m rocket_sim --self-test

echo "[build] running validation suite"
.venv/bin/python -m pytest

echo "[build] packaging with PyInstaller"
.venv/bin/python -m PyInstaller --noconfirm --clean packaging/rocket_sim.spec

ZIP="dist/Static-Rocket-Simulator-0.1.0-macos.zip"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "dist/Static Rocket Flight Simulator.app" "$ZIP"

echo
echo "[build] done: dist/Static Rocket Flight Simulator.app"
echo "[build] shareable archive: $ZIP (unsigned; first launch needs right-click > Open)"
