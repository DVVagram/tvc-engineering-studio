# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TVC Engineering Studio (Windows onedir build).

Build with ``packaging\\build_windows.bat`` from the repository root. The OCP wheel
contains its own OCCT DLLs, so they are collected as package binaries rather than copied
by hand. Qt is collected by PyInstaller's bundled PySide6 hook.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent

datas, binaries, hiddenimports = [], [], []
for package in ("OCP", "pyqtgraph"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Optional at runtime: HIL serial transport. PyInstaller cannot see the dynamic import.
hiddenimports += ["serial", "serial.tools", "serial.tools.list_ports"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "IPython", "PyQt5", "PyQt6", "pytest",
        "OCP.Build", "OCP.Express", "OCP.Draw",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TVC Engineering Studio",
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
    name="TVC Engineering Studio",
)
