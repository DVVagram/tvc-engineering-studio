"""Entry point and headless self-test."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile


def _emit(text: str) -> None:
    """Windowed PyInstaller builds have no stdout, so fall back to a log file."""
    if sys.stdout is None:
        path = os.path.join(tempfile.gettempdir(), "rocket_sim_self_test.log")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


def _self_test() -> int:
    """Headless checks: fixed hardware, motor validation, flight and repeatability."""
    from .config import DRY_MASS, LAUNCH_MASS, MOTOR, ROCKET
    from .motor import PRESETS, MotorPerformance, MotorSpecError
    from .physics import Simulation

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    _emit(f"rocket: {ROCKET.body_length:.2f} m x {ROCKET.body_diameter * 1000:.0f} mm, "
          f"{ROCKET.fin_count} fixed fins, static margin "
          f"{ROCKET.static_margin_calibers:.1f} calibers")
    _emit(f"motor: {MOTOR.diameter_mm:.0f} x {MOTOR.length_mm:.0f} mm, casing "
          f"{MOTOR.casing_mass * 1000:.0f} g + propellant {MOTOR.propellant_mass * 1000:.0f} g "
          f"(fixed)")
    _emit(f"mass: launch {LAUNCH_MASS:.3f} kg, dry {DRY_MASS:.3f} kg")

    performance = MotorPerformance().validate()
    curve = performance.curve()
    check(abs(curve.integrated_impulse() - performance.total_impulse) < 1e-6,
          "thrust curve does not integrate to the stated impulse")
    check(abs(curve.peak_thrust - performance.maximum_thrust) < 0.02 * performance.maximum_thrust,
          "thrust curve peak does not match the stated maximum thrust")
    _emit(f"curve: {performance.display_name}, impulse {curve.total_impulse:.1f} N s, "
          f"peak {curve.peak_thrust:.1f} N, burn {curve.burn_time:.2f} s "
          f"({curve.source_label})")

    simulation = Simulation(performance).run_to_end()
    check(simulation.status == "Landed", f"flight did not land (status {simulation.status})")
    names = [event.name for event in simulation.events]
    for required in ("Liftoff", "Motor burnout", "Apogee", "Ground impact"):
        check(required in names, f"missing event {required}")
    check(simulation.max_altitude > 50.0, "apogee below 50 m with the standard motor")
    _emit(f"flight: apogee={simulation.max_altitude:.1f} m "
          f"max_speed={simulation.max_speed:.1f} m/s "
          f"impact={simulation.state.time:.2f} s")

    repeat = Simulation(performance).run_to_end()
    check(abs(repeat.max_altitude - simulation.max_altitude) < 1e-9,
          "simulation is not repeatable")
    _emit(f"repeatability: apogee delta = {abs(repeat.max_altitude - simulation.max_altitude):.2e} m")

    low = Simulation(PRESETS["SR-30 — low power"]).run_to_end()
    high = Simulation(PRESETS["SR-90 — high power"]).run_to_end()
    check(high.max_altitude > low.max_altitude, "higher impulse did not increase apogee")
    _emit(f"comparison: SR-30 {low.max_altitude:.1f} m, SR-90 {high.max_altitude:.1f} m")

    try:
        MotorPerformance("broken", 40.0, 44.0, 54.9, 1.8, "progressive").validate()
    except MotorSpecError:
        _emit("validation: inconsistent motor performance rejected")
    else:
        failures.append("inconsistent motor performance was accepted")

    if failures:
        for failure in failures:
            _emit(f"FAILED: {failure}")
        return 1
    _emit("self-test OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rocket-sim", description="Static Rocket Flight Simulator")
    parser.add_argument("--self-test", action="store_true",
                        help="run headless physics checks and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    from PySide6 import QtWidgets

    from .ui import STYLE, MainWindow

    application = QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("Static Rocket Flight Simulator")
    application.setOrganizationName("Static Rocket Flight Simulator")
    application.setStyle("Fusion")
    application.setStyleSheet(STYLE)

    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
