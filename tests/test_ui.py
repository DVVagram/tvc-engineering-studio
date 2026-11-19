"""Interface smoke test: the window builds, flies, resets and reports invalid motors."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def application():
    from PySide6 import QtWidgets

    from rocket_sim.ui import STYLE

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    yield app


def test_window_flies_and_resets(application) -> None:
    from rocket_sim.ui import MainWindow

    window = MainWindow()
    try:
        assert window.sim is None
        window.start_flight()
        assert window.sim is not None
        window.sim.advance(0.5)
        window._render()
        assert "m" in window.readouts["altitude"].text()
        assert window.event_list.count() >= 1  # liftoff

        window.reset_flight()
        assert window.sim.state.time == 0.0
        assert window.readouts["altitude"].text() == "—"
        assert window.event_list.count() == 0

        window.start_flight()
        window.pause_flight()
        assert window.resume_button.isEnabled()
        window.resume_flight()
        assert window.pause_button.isEnabled()
    finally:
        window.close()


def test_inconsistent_motor_is_refused(application) -> None:
    from rocket_sim.ui import MainWindow

    window = MainWindow()
    try:
        window.impulse_spin.setValue(60.0)  # 30.5 N x 1.8 s = 54.9 N s
        window.start_flight()
        assert window.sim is None
        assert "impulse" in window.error_label.text().lower()
        window.fit_button.click()
        window.start_flight()
        assert window.sim is not None
    finally:
        window.close()


def test_motor_hardware_has_no_editor(application) -> None:
    from rocket_sim.ui import MainWindow

    window = MainWindow()
    try:
        labels = [widget.text() for widget in window.findChildren(type(window.twr_label))]
        joined = " ".join(labels)
        assert "casing" in joined.lower()
        # No widget edits motor dimensions: only the performance spins exist.
        spins = window.findChildren(type(window.average_spin))
        assert len(spins) == 4
    finally:
        window.close()
