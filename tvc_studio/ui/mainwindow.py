"""Main window: menus, docked viewport, workspaces and project lifecycle."""

from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from ..assembly.massprops import AssemblyMassCalculator
from ..cad import get_backend
from ..project.io import JsonProjectStore, load_recent_files, remember_recent_file
from ..project.sample import build_blank_project, build_sample_project
from ..project.schema import Project
from ..project.undo import UndoStack
from .widgets import DiagnosticsView
from .workspaces import (
    AnalysisWorkspace, AssemblyWorkspace, CadWorkspace, ElectronicsWorkspace,
    FastenerWorkspace, FlightWorkspace, MechanicsWorkspace, PropulsionWorkspace,
)

CAPABILITIES = [
    ("Implemented", [
        "Exact B-rep CAD kernel (OCCT/OCP): primitives, booleans, fillet/chamfer, extrude/revolve, patterns",
        "STEP / IGES / STL / BREP import and export",
        "Assembly tree with parent-child transforms, 13 mate types and a constraint solver with DOF/rank diagnostics",
        "Mass properties from exact geometry + material density, manual overrides and extra masses with provenance",
        "ISO/UTS fastener catalogue, thread engagement, clamp/strip estimates, install state machine",
        "Servo plant model (MG90S/MG996R/DS3218) with lag, deadband, backlash, slew and torque limits",
        "6-DOF flight simulation with ISA atmosphere, motor curves, Barrowman CP estimate and TVC autopilot",
        "IMU model, flight computer, SITL and COBS/CRC HIL transport",
        "Project files, undo/redo, multi-document, viewport picking/section/exploded view, reports",
    ]),
    ("Partial / approximate", [
        "Mechanical simulation is a bench model: unconstrained components are treated as rigidly mounted, gravity is off, and servo joints are kinematically driven (torque is estimated and clamped)",
        "Aerodynamics uses coefficient estimates; no CFD, transonic or supersonic correction",
        "Barrowman CP uses component bounding envelopes, not the exact fin planform",
        "Thread engagement is a reduced-order model, not FEA",
        "Section views cut at triangle granularity without a capping surface",
    ]),
    ("Not implemented", [
        "Nonlinear FEA / stress analysis",
        "Thermal, combustion or internal ballistics simulation",
        "Multi-body contact between the full rocket structure during flight",
        "Import of native SolidWorks/Onshape/Fusion files (use STEP)",
        "User-authored flight firmware (SITL/HIL use the built-in controller)",
    ]),
]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.store = JsonProjectStore(backend=get_backend())
        self.project = project or build_sample_project()
        self.undo = UndoStack()
        self._dirty = False
        self._closing = False
        self.setWindowTitle(f"{APP_NAME} — {self.project.name}")
        self.resize(1500, 950)
        self._build_ui()
        self._build_menus()
        self._load_project(self.project)
        self._update_title()

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        from ..viz.viewport import Viewport
        self.viewport = Viewport(self)
        dock = QtWidgets.QDockWidget("3D view", self)
        dock.setObjectName("viewportDock")
        dock.setWidget(self.viewport)
        dock.setMinimumWidth(520)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.viewport_dock = dock

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.workspaces = [
            CadWorkspace(self), AssemblyWorkspace(self), FastenerWorkspace(self),
            MechanicsWorkspace(self), ElectronicsWorkspace(self), PropulsionWorkspace(self),
            FlightWorkspace(self), AnalysisWorkspace(self),
        ]
        for workspace in self.workspaces:
            self.tabs.addTab(workspace, workspace.title)
        self.setCentralWidget(self.tabs)

        self.diagnostics = DiagnosticsView()
        diag_dock = QtWidgets.QDockWidget("Diagnostics", self)
        diag_dock.setObjectName("diagnosticsDock")
        diag_dock.setWidget(self.diagnostics)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, diag_dock)
        self.diagnostics_dock = diag_dock
        self.resizeDocks([diag_dock], [180], QtCore.Qt.Vertical)

        self.status_mass = QtWidgets.QLabel("—")
        self.status_dof = QtWidgets.QLabel("—")
        self.status_provenance = QtWidgets.QLabel("—")
        self.statusBar().addPermanentWidget(self.status_dof)
        self.statusBar().addPermanentWidget(self.status_mass)
        self.statusBar().addPermanentWidget(self.status_provenance)
        self.statusBar().showMessage("Ready")

        self.viewport.componentPicked.connect(self._component_picked)
        self.tabs.currentChanged.connect(self._workspace_changed)

    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        self._action(file_menu, "New project", "Ctrl+N", self.new_project)
        self._action(file_menu, "Open sample TVC rocket", "Ctrl+Shift+N", self.open_sample)
        self._action(file_menu, "Open…", "Ctrl+O", self.open_project)
        file_menu.addSeparator()
        self._action(file_menu, "Save", "Ctrl+S", self.save_project)
        self._action(file_menu, "Save as…", "Ctrl+Shift+S", self.save_project_as)
        file_menu.addSeparator()
        self._action(file_menu, "Import STEP/IGES/STL…", "", self.import_geometry)
        self._action(file_menu, "Export assembly STEP…", "", self.export_assembly)
        file_menu.addSeparator()
        self._action(file_menu, "Exit", "Ctrl+Q", self.close)

        edit_menu = menu.addMenu("&Edit")
        self.undo_action = self._action(edit_menu, "Undo", "Ctrl+Z", self.undo_edit)
        self.redo_action = self._action(edit_menu, "Redo", "Ctrl+Y", self.redo_edit)

        view_menu = menu.addMenu("&View")
        self._action(view_menu, "Fit view", "F", self.viewport.fit)
        self._action(view_menu, "Solid", "S", lambda: self.viewport.set_mode("solid"))
        self._action(view_menu, "Wireframe", "W", lambda: self.viewport.set_mode("wireframe"))
        self._action(view_menu, "Transparent", "T", lambda: self.viewport.set_mode("transparent"))
        self._action(view_menu, "Orthographic", "O",
                     lambda: self.viewport.set_orthographic(not self.viewport.opts.get("ortho", False)))
        self.section_action = self._action(view_menu, "Section cut", "C", self._toggle_section)
        self.section_action.setCheckable(True)

        sim_menu = menu.addMenu("&Simulation")
        self._action(sim_menu, "Build mechanism", "", self._open_mechanics)
        self._action(sim_menu, "Run flight simulation", "Ctrl+R", self._open_flight)

        help_menu = menu.addMenu("&Help")
        self._action(help_menu, "Capability status", "", self._show_capabilities)
        self._action(help_menu, "About", "", self._show_about)

        toolbar = QtWidgets.QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(toolbar)
        for i, workspace in enumerate(self.workspaces):
            action = QtGui.QAction(workspace.title, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, index=i: self.tabs.setCurrentIndex(index))
            toolbar.addAction(action)
            if i == 0:
                self._workspace_actions = []
            self._workspace_actions.append(action)
        toolbar.addSeparator()
        toolbar.addAction("Fit", self.viewport.fit)
        toolbar.addAction("Undo", self.undo_edit)
        toolbar.addAction("Redo", self.redo_edit)
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)

    def _action(self, menu, text: str, shortcut: str, slot):
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # -------------------------------------------------------------- project
    def _load_project(self, project: Project) -> None:
        self.project = project
        self.undo.clear()
        self._dirty = False
        self.viewport.set_project(project)
        for workspace in self.workspaces:
            workspace.refresh()
        self.refresh_status()
        self._update_title()
        self._update_undo_actions()

    def apply_edit(self, label: str, fn) -> None:
        self.undo.begin(label, self.project)
        try:
            fn()
        except Exception as exc:
            self.undo.abort()
            QtWidgets.QMessageBox.warning(self, f"{label} failed", str(exc))
            return
        self.project.touch()
        changed = self.undo.commit(self.project)
        self._dirty = True
        self.rebuild_view()
        self.refresh_status()
        self._update_undo_actions()
        self.set_status(label if changed else f"{label}: no change")
        self._update_title()

    def refresh_all(self) -> None:
        current = self.tabs.currentIndex()
        for workspace in self.workspaces:
            workspace.refresh()
        self.rebuild_view()
        self.refresh_status()
        self.tabs.setCurrentIndex(current)

    def rebuild_view(self) -> None:
        try:
            self.viewport.set_project(self.project)
        except Exception as exc:
            self.set_status(f"Viewport rebuild failed: {exc}")
        if getattr(self, "workspaces", None):
            self.workspaces[1].refresh()

    def refresh_status(self) -> None:
        try:
            calc = AssemblyMassCalculator(self.project.parts, self.project.materials,
                                          self.project.fasteners)
            result = calc.calculate(self.project.assembly)
            self.status_mass.setText(f"Mass {result.mass*1000:.1f} g")
            self.status_provenance.setText(f"Mass provenance: {result.properties.provenance}")
        except Exception as exc:
            self.status_mass.setText("Mass unavailable")
            self.status_provenance.setText(str(exc)[:60])
        try:
            from ..assembly.constraints import AssemblySolver
            summary = AssemblySolver(self.project.assembly).solve(apply=False)
            self.status_dof.setText(summary.summary())
        except Exception:
            self.status_dof.setText("DOF unavailable")

    def set_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)

    def set_diagnostics(self, diagnostics) -> None:
        try:
            self.diagnostics.set_diagnostics(diagnostics)
        except Exception:
            pass

    def _component_picked(self, component_id: str) -> None:
        if not component_id or component_id.startswith("fastener:"):
            return
        self.tabs.setCurrentIndex(1)
        assembly = self.workspaces[1]
        for i in range(assembly.tree.topLevelItemCount()):
            item = assembly.tree.topLevelItem(i)
            if self._select_in_tree(item, component_id):
                return

    def _select_in_tree(self, item, component_id: str) -> bool:
        if item.data(0, QtCore.Qt.UserRole) == component_id:
            item.setSelected(True)
            self.workspaces[1].tree.scrollToItem(item)
            return True
        for i in range(item.childCount()):
            if self._select_in_tree(item.child(i), component_id):
                return True
        return False

    def _workspace_changed(self, index: int) -> None:
        for i, action in enumerate(getattr(self, "_workspace_actions", [])):
            action.setChecked(i == index)

    # ------------------------------------------------------------------ file
    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._load_project(build_blank_project())

    def open_sample(self) -> None:
        if not self._confirm_discard():
            return
        self._load_project(build_sample_project())

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", "",
                                                        "TVC project (*.tvcproj);;All files (*)")
        if not path:
            return
        try:
            project = self.store.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            return
        remember_recent_file(path)
        self._load_project(project)

    def save_project(self) -> bool:
        if not self.project.path:
            return self.save_project_as()
        try:
            self.store.save(self.project, self.project.path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return False
        remember_recent_file(self.project.path)
        self._dirty = False
        self._update_title()
        self.set_status(f"Saved {self.project.path}")
        return True

    def save_project_as(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project",
                                                        f"{self.project.name}.tvcproj",
                                                        "TVC project (*.tvcproj)")
        if not path:
            return False
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        try:
            saved = self.store.save(self.project, path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return False
        remember_recent_file(saved)
        self.project.path = saved
        self._dirty = False
        self._update_title()
        self.set_status(f"Saved {saved}")
        return True

    def import_geometry(self) -> None:
        self.tabs.setCurrentIndex(0)
        self.workspaces[0]._import()

    def export_assembly(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export assembly STEP",
                                                        f"{self.project.name}.step",
                                                        "STEP (*.step *.stp)")
        if not path:
            return
        try:
            self.store.export_step(self.project, path)
            self.set_status(f"Assembly exported to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    # ------------------------------------------------------------------ edit
    def undo_edit(self) -> None:
        snapshot = self.undo.undo()
        if snapshot is None:
            return
        self._restore(snapshot)
        self.set_status("Undo")

    def redo_edit(self) -> None:
        snapshot = self.undo.redo()
        if snapshot is None:
            return
        self._restore(snapshot)
        self.set_status("Redo")

    def _restore(self, snapshot: dict) -> None:
        try:
            project = Project.from_dict(snapshot, backend=self.project.parts.backend)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Undo failed", str(exc))
            return
        project.path = self.project.path
        self.project = project
        self.viewport.set_project(project)
        for workspace in self.workspaces:
            workspace.refresh()
        self.refresh_status()
        self._update_undo_actions()
        self._update_title()

    def _update_undo_actions(self) -> None:
        if not hasattr(self, "undo_action"):
            return
        self.undo_action.setEnabled(self.undo.can_undo())
        self.redo_action.setEnabled(self.undo.can_redo())
        self.undo_action.setText(f"Undo {self.undo.undo_label()}".strip())
        self.redo_action.setText(f"Redo {self.undo.redo_label()}".strip())

    def _update_title(self) -> None:
        star = " •" if self._dirty else ""
        self.setWindowTitle(f"{APP_NAME} — {self.project.name}{star}")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        reply = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Save changes to the current project first?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if reply == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return reply == QtWidgets.QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    # -------------------------------------------------------------- actions
    def _toggle_section(self) -> None:
        self.viewport.set_section(self.section_action.isChecked(), axis=1, offset=0.0)

    def _open_mechanics(self) -> None:
        self.tabs.setCurrentIndex(3)
        self.workspaces[3]._build_world()

    def _open_flight(self) -> None:
        self.tabs.setCurrentIndex(6)
        self.workspaces[6]._run()

    def _show_capabilities(self) -> None:
        text = []
        for level, items in CAPABILITIES:
            text.append(level.upper())
            text.extend(f"  • {item}" for item in items)
            text.append("")
        QtWidgets.QMessageBox.information(self, "Capability status", "\n".join(text))

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {__version__}<br><br>"
            "Native Windows engineering studio for thrust-vector-controlled rockets: "
            "parametric CAD, assembly mates, fasteners, servo mechanics, avionics in the loop "
            "and 6-DOF flight simulation.<br><br>"
            "All data stays on this computer. Geometry is solved with OCCT; simulations use the "
            "project's own mass, motor and aerodynamic data, and every estimate is labelled.")
