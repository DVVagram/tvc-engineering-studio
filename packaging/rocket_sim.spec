# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Static Rocket Flight Simulator.

Windows build: ``packaging\\build_windows.bat``.
macOS build:   ``packaging/build_macos.sh`` (produces a ``.app`` bundle).
Both run from the repository root. Qt is collected by PyInstaller's bundled PySide6 hook;
pyqtgraph is collected explicitly because its OpenGL module is imported lazily.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent

datas, binaries, hiddenimports = [], [], []
for package in ("pyqtgraph",):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += ["pyqtgraph.opengl", "OpenGL"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "PyQt5", "PyQt6", "pytest", "scipy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Static Rocket Flight Simulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Static Rocket Flight Simulator",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Static Rocket Flight Simulator.app",
        bundle_identifier="com.staticrocketsim.desktop",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
