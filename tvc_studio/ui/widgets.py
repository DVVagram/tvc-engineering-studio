"""Small reusable widgets for the workspaces."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np


def form_group(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QFormLayout(box)
    layout.setLabelAlignment(QtCore.Qt.AlignRight)
    layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    return box, layout


def spin(minimum: float, maximum: float, value: float, decimals: int = 3,
         suffix: str = "", step: float | None = None) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setValue(value)
    if suffix:
        box.setSuffix(f" {suffix}")
    box.setSingleStep(step if step is not None else 10 ** (-max(decimals - 1, 0)))
    box.setKeyboardTracking(False)
    return box


def int_spin(minimum: int, maximum: int, value: int, suffix: str = "") -> QtWidgets.QSpinBox:
    box = QtWidgets.QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    if suffix:
        box.setSuffix(f" {suffix}")
    box.setKeyboardTracking(False)
    return box


def line_edit(text: str = "", read_only: bool = False) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(text)
    edit.setReadOnly(read_only)
    if read_only:
        edit.setStyleSheet("color:#9aa4b0;")
    return edit


def table(headers: list[str], stretch_last: bool = True) -> QtWidgets.QTableWidget:
    widget = QtWidgets.QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    widget.verticalHeader().setVisible(False)
    widget.setAlternatingRowColors(True)
    header = widget.horizontalHeader()
    header.setStretchLastSection(stretch_last)
    header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    widget.setMinimumHeight(120)
    return widget


def fill_table(widget: QtWidgets.QTableWidget, rows: list[dict], headers: list[str] | None = None,
               colours: dict[int, QtGui.QColor] | None = None) -> None:
    if headers is None:
        headers = list(rows[0].keys()) if rows else [widget.horizontalHeaderItem(i).text()
                                                     for i in range(widget.columnCount())]
    widget.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, key in enumerate(headers):
            value = row.get(key, "")
            if isinstance(value, float):
                if math.isnan(value):
                    text = "n/a"
                elif abs(value) >= 1e5 or (0 < abs(value) < 1e-3):
                    text = f"{value:.4g}"
                else:
                    text = f"{value:,.4f}".rstrip("0").rstrip(".")
            else:
                text = "" if value is None else str(value)
            item = QtWidgets.QTableWidgetItem(text)
            if colours and r in colours:
                item.setForeground(colours[r])
            widget.setItem(r, c, item)


def selected_row(widget: QtWidgets.QTableWidget) -> int:
    rows = widget.selectionModel().selectedRows() if widget.selectionModel() else []
    return rows[0].row() if rows else -1


class PlotPanel(QtWidgets.QWidget):
    """A compact pyqtgraph plot with named series."""

    def __init__(self, x_label: str = "t (s)", y_label: str = "", parent=None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg
        self._pg = pg
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#1b1e23")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", x_label)
        if y_label:
            self.plot.setLabel("left", y_label)
        self.plot.addLegend(offset=(-10, 10))
        layout.addWidget(self.plot)
        self._curves: dict[str, object] = {}

    def clear(self) -> None:
        for curve in self._curves.values():
            self.plot.removeItem(curve)
        self._curves.clear()
        self.plot.clear()

    def plot_series(self, x: np.ndarray, y: np.ndarray, name: str, color: str = "#6ab0ff") -> None:
        if len(x) == 0:
            return
        curve = self.plot.plot(x, y, name=name, pen=self._pg.mkPen(color, width=2))
        self._curves[name] = curve

    def set_y_label(self, text: str) -> None:
        self.plot.setLabel("left", text)


class DiagnosticsView(QtWidgets.QTableWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Level", "Area", "Message"])
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(True)

    def set_diagnostics(self, diagnostics) -> None:
        rows = []
        for d in diagnostics:
            level = getattr(d, "level", "info")
            area = getattr(d, "area", getattr(d, "source", ""))
            message = getattr(d, "message", str(d))
            rows.append((level, area, message))
        self.setRowCount(len(rows))
        colours = {"error": "#ff7b72", "warning": "#e3b341", "info": "#7fb3d5"}
        for r, (level, area, message) in enumerate(rows):
            for c, text in enumerate((level.upper(), area, message)):
                item = QtWidgets.QTableWidgetItem(str(text))
                if c == 0:
                    item.setForeground(QtGui.QColor(colours.get(level, "#c9d1d9")))
                self.setItem(r, c, item)
