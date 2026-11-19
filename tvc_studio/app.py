"""Application entry point."""

from __future__ import annotations

import argparse
import os
import sys


def _self_test_out(text: str) -> None:
    """Emit self-test output; windowed (PyInstaller --noconsole) builds have no stdout."""
    import tempfile
    if sys.stdout is None:
        path = os.path.join(tempfile.gettempdir(), "tvc_studio_self_test.log")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def _self_test() -> int:
    """Headless smoke test used by packaging/CI (no GUI)."""
    from .assembly.massprops import AssemblyMassCalculator
    from .mech.build import build_world
    from .physics.flight import FlightSimulator
    from .physics.rocket import build_rocket
    from .project.io import JsonProjectStore
    from .project.sample import build_sample_project
    import tempfile

    project = build_sample_project()
    project.parts.rebuild_all()
    calc = AssemblyMassCalculator(project.parts, project.materials, project.fasteners)
    mass = calc.calculate(project.assembly)
    _self_test_out(f"parts={len(project.parts.parts)} components={len(project.assembly.components)} "
                   f"mass={mass.mass*1000:.1f} g")
    world = build_world(project.assembly, calc, project.parts, project.hardware,
                        backend=project.parts.backend)
    _self_test_out(f"bodies={len(world.world.bodies)} joints={len(world.world.joints)} "
                   f"servos={len(world.servos)}")
    rocket = build_rocket(project.assembly, calc, project.parts, project.motor(),
                          project.hardware, project.aero)
    sim = FlightSimulator(rocket.properties, project.motor(), project.aero, project.flight,
                          rocket.stack, control=project.control)
    result = sim.run(20.0)
    _self_test_out(f"apogee={result.summary['apogee_m']:.1f} m "
                   f"max_speed={result.summary['max_speed_ms']:.1f} m/s")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip.tvcproj")
        store = JsonProjectStore(project.parts.backend)
        store.save(project, path)
        loaded = store.load(path)
        _self_test_out(f"project round-trip: {len(loaded.parts.parts)} parts, "
                       f"{len(loaded.assembly.components)} components")
    _self_test_out("self-test OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tvc-studio", description="TVC Engineering Studio")
    parser.add_argument("project", nargs="?", help="project file to open")
    parser.add_argument("--sample", action="store_true", help="open the tutorial project")
    parser.add_argument("--self-test", action="store_true", help="run headless checks and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    from .ui.mainwindow import MainWindow
    from .project.io import JsonProjectStore
    from .project.sample import build_sample_project
    from .cad import get_backend

    try:
        from PySide6 import QtWidgets
    except Exception as exc:  # pragma: no cover
        print(f"PySide6 is required to run the application: {exc}", file=sys.stderr)
        return 1

    app = QtWidgets.QApplication(sys.argv[:1] + [])
    app.setApplicationName("TVC Engineering Studio")
    app.setOrganizationName("TVC Engineering Studio")
    app.setStyle("Fusion")

    project = None
    if args.project:
        try:
            project = JsonProjectStore(get_backend()).load(args.project)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(None, "Could not open project", str(exc))
            return 1
    elif args.sample:
        project = build_sample_project()

    window = MainWindow(project)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
