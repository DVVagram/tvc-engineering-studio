"""Interface: one large 3D viewport, one motor-performance panel, one flight-data panel."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from .config import LAUNCH_MASS, MOTOR, ROCKET, DRY_MASS
from .motor import (
    PRESETS,
    SHAPES,
    SHAPE_LABELS,
    MotorPerformance,
    MotorSpecError,
)
from .physics import Simulation
from .viewport import RocketViewport

STYLE = """
QWidget { background: #11151b; color: #dce5ef; font-size: 12px; }
QGroupBox { border: 1px solid #25303c; border-radius: 6px; margin-top: 12px;
            padding: 8px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #8ea3b9; font-weight: 600; }
QPushButton { background: #1d2631; border: 1px solid #33404f; border-radius: 4px;
              padding: 6px 10px; }
QPushButton:hover { background: #26323f; }
QPushButton:pressed { background: #172029; }
QPushButton:disabled { color: #5a6572; background: #161b22; border-color: #222a34; }
QPushButton#primary { background: #2b5f8c; border-color: #3d7cb0; font-weight: 600; }
QLineEdit, QDoubleSpinBox, QComboBox { background: #151b23; border: 1px solid #2a3542;
                                       border-radius: 4px; padding: 4px; }
QComboBox QAbstractItemView { background: #151b23; selection-background-color: #2b5f8c; }
QListWidget { background: #0e1319; border: 1px solid #222b36; }
QLabel#error { color: #ff9182; }
QLabel#hint { color: #7f93a8; }
QLabel#sectionnote { color: #7f93a8; }
QLabel#value { font-family: Consolas, "DejaVu Sans Mono", monospace; color: #f0f5fa; }
QCheckBox { spacing: 6px; }
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Static Rocket Flight Simulator")
        self.resize(1280, 820)
        self.sim: Simulation | None = None
        self._event_index = 0
        self._running = False
        self._paused = False
        self._last_wall = time.perf_counter()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._on_tick)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        left = QtWidgets.QVBoxLayout()
        self.viewport = RocketViewport()
        self.viewport.setMinimumSize(640, 480)
        left.addWidget(self.viewport, 1)
        left.addLayout(self._build_controls())
        layout.addLayout(left, 1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(340)
        side = QtWidgets.QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)
        side.addWidget(self._build_motor_group())
        side.addWidget(self._build_hardware_group())
        side.addWidget(self._build_data_group())
        side.addWidget(self._build_events_group(), 1)
        layout.addWidget(panel)

        self.setCentralWidget(central)
        self.follow_box.setChecked(True)  # keep the vehicle in frame by default
        self._reset_readouts()
        self._update_controls()

    # ------------------------------------------------------------------ building
    def _build_controls(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_flight)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_flight)
        self.resume_button = QtWidgets.QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_flight)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_flight)
        self.restart_button = QtWidgets.QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart_flight)
        for button in (self.start_button, self.pause_button, self.resume_button,
                       self.reset_button, self.restart_button):
            row.addWidget(button)
        row.addStretch(1)
        self.follow_box = QtWidgets.QCheckBox("Follow rocket")
        self.follow_box.toggled.connect(self.viewport.set_follow)
        row.addWidget(self.follow_box)
        self.camera_button = QtWidgets.QPushButton("Reset camera")
        self.camera_button.clicked.connect(self.reset_camera)
        row.addWidget(self.camera_button)
        return row

    def _build_motor_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Motor performance")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.preset_box = QtWidgets.QComboBox()
        self.preset_box.addItem("Custom")
        for name in PRESETS:
            self.preset_box.addItem(name)
        self.preset_box.currentTextChanged.connect(self._apply_preset)
        form.addRow("Preset", self.preset_box)

        self.name_edit = QtWidgets.QLineEdit("SR-55")
        self.name_edit.setMaxLength(32)
        form.addRow("Designation", self.name_edit)

        self.average_spin = self._thrust_spin(30.5)
        self.peak_spin = self._thrust_spin(44.0)
        self.impulse_spin = self._impulse_spin(54.9)
        self.burn_spin = self._burn_spin(1.8)
        self.shape_box = QtWidgets.QComboBox()
        for shape in SHAPES:
            self.shape_box.addItem(SHAPE_LABELS[shape], shape)
        self.shape_box.setCurrentIndex(1)

        form.addRow("Average thrust", self.average_spin)
        form.addRow("Maximum thrust", self.peak_spin)
        form.addRow("Total impulse", self.impulse_spin)
        form.addRow("Burn time", self.burn_spin)
        form.addRow("Curve shape", self.shape_box)

        self.fit_button = QtWidgets.QPushButton("Set impulse = average × burn time")
        self.fit_button.clicked.connect(self._fit_impulse)
        form.addRow(self.fit_button)

        note = QtWidgets.QLabel(
            "Thrust curve is estimated from these numbers — it is not measured motor data.\n"
            "Values are applied when you press Start or Restart."
        )
        note.setObjectName("sectionnote")
        note.setWordWrap(True)
        form.addRow(note)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        form.addRow(self.error_label)

        self.twr_label = QtWidgets.QLabel("")
        self.twr_label.setObjectName("hint")
        form.addRow(self.twr_label)
        for widget in (self.average_spin, self.peak_spin, self.impulse_spin, self.burn_spin):
            widget.valueChanged.connect(self._update_twr_hint)
        self.shape_box.currentIndexChanged.connect(self._update_twr_hint)
        self._update_twr_hint()
        return group

    def _build_hardware_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Fixed hardware (read-only)")
        layout = QtWidgets.QVBoxLayout(group)
        text = (
            f"Rocket: {ROCKET.body_diameter * 1000:.0f} mm × "
            f"{ROCKET.body_length:.2f} m, {ROCKET.structural_mass * 1000:.0f} g structure, "
            f"{ROCKET.fin_count} fixed fins\n"
            f"CG {ROCKET.center_of_gravity_from_nose_tip:.3f} m from nose tip, "
            f"CP {ROCKET.center_of_pressure_from_nose_tip:.3f} m, "
            f"static margin {ROCKET.static_margin_calibers:.1f} calibers\n"
            f"Motor: Ø{MOTOR.diameter_mm:.0f} × {MOTOR.length_mm:.0f} mm, "
            f"casing {MOTOR.casing_mass * 1000:.0f} g + propellant "
            f"{MOTOR.propellant_mass * 1000:.0f} g, fixed mount\n"
            f"Launch mass {LAUNCH_MASS:.3f} kg, dry mass {DRY_MASS:.3f} kg"
        )
        label = QtWidgets.QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        layout.addWidget(label)
        return group

    def _build_data_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Flight data")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.readouts: dict[str, QtWidgets.QLabel] = {}
        for key, title in (
            ("altitude", "Altitude"),
            ("vertical_speed", "Vertical velocity"),
            ("speed", "Total velocity"),
            ("acceleration", "Acceleration"),
            ("time", "Flight time"),
            ("thrust", "Thrust"),
            ("motor", "Motor status"),
            ("max_altitude", "Maximum altitude"),
            ("max_speed", "Maximum velocity"),
        ):
            value = QtWidgets.QLabel("—")
            value.setObjectName("value")
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.readouts[key] = value
            form.addRow(title, value)
        return group

    def _build_events_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Flight events")
        layout = QtWidgets.QVBoxLayout(group)
        self.event_list = QtWidgets.QListWidget()
        self.event_list.setMaximumHeight(140)
        layout.addWidget(self.event_list)
        return group

    def _thrust_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.1, 8000.0)
        spin.setDecimals(1)
        spin.setSuffix(" N")
        spin.setValue(value)
        return spin

    def _impulse_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.1, 5000.0)
        spin.setDecimals(1)
        spin.setSuffix(" N s")
        spin.setValue(value)
        return spin

    def _burn_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.02, 30.0)
        spin.setDecimals(2)
        spin.setSuffix(" s")
        spin.setSingleStep(0.1)
        spin.setValue(value)
        return spin

    # ------------------------------------------------------------------- inputs
    def performance(self) -> MotorPerformance:
        return MotorPerformance(
            designation=self.name_edit.text(),
            average_thrust=self.average_spin.value(),
            maximum_thrust=self.peak_spin.value(),
            total_impulse=self.impulse_spin.value(),
            burn_time=self.burn_spin.value(),
            curve_shape=self.shape_box.currentData(),
            measured=False,
        )

    def _apply_preset(self, name: str) -> None:
        preset = PRESETS.get(name)
        if preset is None:
            return
        self.name_edit.setText(preset.designation)
        self.average_spin.setValue(preset.average_thrust)
        self.peak_spin.setValue(preset.maximum_thrust)
        self.impulse_spin.setValue(preset.total_impulse)
        self.burn_spin.setValue(preset.burn_time)
        index = self.shape_box.findData(preset.curve_shape)
        if index >= 0:
            self.shape_box.setCurrentIndex(index)
        self._set_error("")

    def _fit_impulse(self) -> None:
        self.impulse_spin.setValue(round(self.average_spin.value() * self.burn_spin.value(), 1))
        self._set_error("")

    def _update_twr_hint(self) -> None:
        average = self.average_spin.value()
        ratio = average / (LAUNCH_MASS * 9.80665)
        self.twr_label.setText(
            f"Average thrust / launch weight = {ratio:.2f} m s⁻² "
            f"({'lifts off' if ratio > 1.0 else 'cannot lift the rocket'})"
        )

    def _set_error(self, message: str) -> None:
        self.error_label.setText(message)

    # ------------------------------------------------------------------ actions
    def start_flight(self) -> None:
        try:
            performance = self.performance().validate()
        except MotorSpecError as exc:
            self._set_error("\n".join(f"• {problem}" for problem in exc.errors))
            return
        self._set_error("")
        try:
            simulation = Simulation(performance)
        except MotorSpecError as exc:  # defensive; validate() already ran
            self._set_error(" ".join(exc.errors))
            return
        self.sim = simulation
        self.sim.start()
        self.event_list.clear()
        self._event_index = 0
        self.viewport.clear_trail()
        self.viewport.update_trail([])
        self.viewport.update_rocket(self.sim.state.position, self.sim.state.attitude)
        self.viewport.follow = self.follow_box.isChecked()
        self._running = True
        self._paused = False
        self._last_wall = time.perf_counter()
        self.timer.start()
        self._render()
        self._update_controls()

    def pause_flight(self) -> None:
        if not self._running or self._paused:
            return
        self.timer.stop()
        self._paused = True
        self._running = False
        self._update_controls()

    def resume_flight(self) -> None:
        if not self._paused or self.sim is None:
            return
        self._paused = False
        self._running = True
        self._last_wall = time.perf_counter()
        self.timer.start()
        self._update_controls()

    def reset_flight(self) -> None:
        self.timer.stop()
        self._running = False
        self._paused = False
        if self.sim is not None:
            self.sim.reset()
            self.viewport.update_rocket(self.sim.state.position, self.sim.state.attitude)
        self.viewport.clear_trail()
        self.event_list.clear()
        self._event_index = 0
        self._reset_readouts()
        self._set_error("")
        self._update_controls()

    def restart_flight(self) -> None:
        self.reset_flight()
        self.start_flight()

    def _reset_readouts(self) -> None:
        for value in self.readouts.values():
            value.setText("—")

    def reset_camera(self) -> None:
        self.follow_box.setChecked(False)
        self.viewport.reset_view()

    def _update_controls(self) -> None:
        self.start_button.setEnabled(not self._running and not self._paused)
        self.pause_button.setEnabled(self._running and not self._paused)
        self.resume_button.setEnabled(self._paused)
        self.reset_button.setEnabled(self.sim is not None)
        self.restart_button.setEnabled(self.sim is not None)

    # ---------------------------------------------------------------- animation
    def _on_tick(self) -> None:
        if self.sim is None:
            return
        now = time.perf_counter()
        elapsed = min(now - self._last_wall, 0.1)
        self._last_wall = now
        self.sim.advance(elapsed)
        self._render()
        if self.sim.finished:
            self.timer.stop()
            self._running = False
            self._paused = False
            self._update_controls()

    def _render(self) -> None:
        if self.sim is None:
            return
        self.viewport.update_rocket(self.sim.state.position, self.sim.state.attitude)

        points = self.sim.trail
        if len(points) >= 2:
            self.viewport.update_trail(points)

        while self._event_index < len(self.sim.events):
            event = self.sim.events[self._event_index]
            detail = f" — {event.detail}" if event.detail else ""
            self.event_list.addItem(f"t = {event.time:6.2f} s   {event.name}{detail}")
            self._event_index += 1

        self.readouts["altitude"].setText(f"{self.sim.altitude:.1f} m")
        self.readouts["vertical_speed"].setText(f"{self.sim.vertical_speed:+.1f} m/s")
        self.readouts["speed"].setText(f"{self.sim.speed:.1f} m/s")
        self.readouts["acceleration"].setText(f"{self.sim.acceleration:.1f} m/s²")
        self.readouts["time"].setText(f"{self.sim.time:.2f} s")
        self.readouts["thrust"].setText(f"{self.sim.thrust:.1f} N")
        self.readouts["motor"].setText(self.sim.motor_status)
        self.readouts["max_altitude"].setText(f"{self.sim.max_altitude:.1f} m")
        self.readouts["max_speed"].setText(f"{self.sim.max_speed:.1f} m/s")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        self.timer.stop()
        super().closeEvent(event)
