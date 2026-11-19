"""The eight engineering workspaces.

Each workspace edits the live project through the main window's edit system, so every change
is undoable and the viewport/status bar refresh from the same data.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from ..assembly.constraints import AssemblyConstraint, ConstraintType, EntityKind, EntityRef
from ..assembly.massprops import AssemblyMassCalculator
from ..cad.features import BoxFeature, ConeFeature, CylinderFeature, HoleFeature, TubeFeature
from ..components.properties import MassMode
from ..core.interfaces import TessellationQuality
from ..core.math3d import Quat, Transform, vec3
from ..fasteners.install import InstalledFastener, ScrewInstaller
from .widgets import (
    DiagnosticsView, PlotPanel, fill_table, form_group, int_spin, line_edit, selected_row,
    spin, table,
)

MM = 1000.0


class Workspace(QtWidgets.QWidget):
    title = "Workspace"

    def __init__(self, main) -> None:
        super().__init__()
        self.main = main
        self._build()
        self.refresh()

    @property
    def project(self):
        return self.main.project

    def _build(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def refresh(self) -> None:
        pass

    def _apply(self, label: str, fn) -> None:
        if not getattr(self.main, "workspaces", None):
            fn()
            return
        self.main.apply_edit(label, fn)


# --------------------------------------------------------------------------- CAD
class CadWorkspace(Workspace):
    title = "CAD"

    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.part_list = QtWidgets.QListWidget()
        self.part_list.currentRowChanged.connect(self._part_selected)
        left.addWidget(QtWidgets.QLabel("Parts"))
        left.addWidget(self.part_list, 1)

        create_box, form = form_group("New part")
        self.new_name = line_edit("New part")
        self.new_type = QtWidgets.QComboBox()
        self.new_type.addItems(["Box", "Cylinder", "Tube", "Cone"])
        self.new_type.currentTextChanged.connect(self._update_new_fields)
        self.dim_a = spin(0.1, 900.0, 40.0, 2, "mm")
        self.dim_b = spin(0.1, 900.0, 40.0, 2, "mm")
        self.dim_c = spin(0.1, 900.0, 40.0, 2, "mm")
        form.addRow("Name", self.new_name)
        form.addRow("Primitive", self.new_type)
        form.addRow("Size 1", self.dim_a)
        form.addRow("Size 2", self.dim_b)
        form.addRow("Size 3", self.dim_c)
        create = QtWidgets.QPushButton("Create part")
        create.clicked.connect(self._create_part)
        form.addRow(create)
        left.addWidget(create_box)
        left.addStretch(1)
        layout.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout()
        self.feature_table = table(["#", "Feature", "Operation"])
        right.addWidget(QtWidgets.QLabel("Feature history"))
        right.addWidget(self.feature_table, 1)

        add_box, add_form = form_group("Add feature to selected part")
        self.feat_type = QtWidgets.QComboBox()
        self.feat_type.addItems(["Box", "Cylinder", "Tube", "Cone", "Hole"])
        self.feat_type.currentTextChanged.connect(self._update_feat_fields)
        self.fd1 = spin(0.1, 500.0, 10.0, 2, "mm")
        self.fd2 = spin(0.0, 500.0, 5.0, 2, "mm")
        self.fd3 = spin(0.1, 500.0, 20.0, 2, "mm")
        self.fx = spin(-500.0, 500.0, 0.0, 2, "mm")
        self.fy = spin(-500.0, 500.0, 0.0, 2, "mm")
        self.fz = spin(-500.0, 500.0, 0.0, 2, "mm")
        add_form.addRow("Type", self.feat_type)
        add_form.addRow("Diameter / inner", self.fd1)
        add_form.addRow("Outer / radius 2", self.fd2)
        add_form.addRow("Height / depth", self.fd3)
        add_form.addRow("X", self.fx)
        add_form.addRow("Y", self.fy)
        add_form.addRow("Z", self.fz)
        add_feature = QtWidgets.QPushButton("Add feature")
        add_feature.clicked.connect(self._add_feature)
        add_form.addRow(add_feature)
        right.addWidget(add_box)

        tools = QtWidgets.QHBoxLayout()
        self.material = QtWidgets.QComboBox()
        self.material.currentTextChanged.connect(self._material_changed)
        rebuild = QtWidgets.QPushButton("Rebuild")
        rebuild.clicked.connect(self._rebuild)
        delete = QtWidgets.QPushButton("Delete part")
        delete.clicked.connect(self._delete_part)
        import_btn = QtWidgets.QPushButton("Import STEP/IGES/STL…")
        import_btn.clicked.connect(self._import)
        export_btn = QtWidgets.QPushButton("Export STEP…")
        export_btn.clicked.connect(self._export)
        tools.addWidget(QtWidgets.QLabel("Material"))
        tools.addWidget(self.material, 1)
        tools.addWidget(rebuild)
        tools.addWidget(delete)
        tools.addWidget(import_btn)
        tools.addWidget(export_btn)
        right.addLayout(tools)
        self.mass_label = QtWidgets.QLabel("Select a part")
        self.mass_label.setWordWrap(True)
        right.addWidget(self.mass_label)
        layout.addLayout(right, 2)

    # -- helpers -----------------------------------------------------------
    def _update_new_fields(self, kind: str) -> None:
        self.dim_c.setEnabled(kind != "Cylinder")
        self.dim_b.setEnabled(kind != "Box" or True)

    def _update_feat_fields(self, kind: str) -> None:
        self.fd2.setEnabled(kind in ("Tube", "Cone"))

    def _feature_from_fields(self, kind: str):
        dx = self.fd1.value() / MM
        dy = self.fd2.value() / MM
        dz = self.fd3.value() / MM if self.feat_type.currentText() != "Hole" else self.fd3.value() / MM
        origin = (self.fx.value() / MM, self.fy.value() / MM, self.fz.value() / MM)
        if kind == "Box":
            return BoxFeature(dx, max(dy, 1e-4), dz, origin=origin, name="Box")
        if kind == "Cylinder":
            return CylinderFeature(radius=dx / 2.0, height=dz, origin=origin, name="Cylinder")
        if kind == "Tube":
            inner = dx / 2.0
            outer = max(dy / 2.0, inner + 1e-4)
            return TubeFeature(inner_radius=inner, outer_radius=outer, height=dz,
                               origin=origin, name="Tube")
        if kind == "Cone":
            return ConeFeature(radius1=dx / 2.0, radius2=max(dy / 2.0, 0.0), height=dz,
                               origin=origin, name="Cone")
        return HoleFeature(position=origin, axis=(0, 0, -1), diameter=dx,
                           depth=self.fd3.value() / MM, through=True, name="Hole")

    def _selected_part(self):
        row = self.part_list.currentRow()
        ids = list(self.project.parts.parts)
        if 0 <= row < len(ids):
            return self.project.parts.parts[ids[row]]
        return None

    def _create_part(self) -> None:
        name = self.new_name.text().strip() or "Part"
        kind = self.new_type.currentText()
        if kind == "Box":
            feature = BoxFeature(self.dim_a.value() / MM, self.dim_b.value() / MM,
                                 self.dim_c.value() / MM, name="Base block")
        elif kind == "Cylinder":
            feature = CylinderFeature(radius=self.dim_a.value() / 2 / MM,
                                      height=self.dim_b.value() / MM, name="Base cylinder")
        elif kind == "Tube":
            inner = self.dim_a.value() / 2 / MM
            outer = max(self.dim_b.value() / 2 / MM, inner + 1e-4)
            feature = TubeFeature(inner_radius=inner, outer_radius=outer,
                                  height=self.dim_c.value() / MM, name="Base tube")
        else:
            feature = ConeFeature(radius1=self.dim_a.value() / 2 / MM,
                                  radius2=self.dim_b.value() / 2 / MM,
                                  height=self.dim_c.value() / MM, name="Base cone")
        material = self.material.currentText() or "pla"

        def make():
            part = self.project.parts.create(name, [feature], material_id=material)
            try:
                part.rebuild(self.project.parts.backend)
            except Exception:
                pass
            self.project.parts.invalidate(part.id)
            return part

        self._apply(f"Create part {name}", make)

    def _add_feature(self) -> None:
        part = self._selected_part()
        if part is None:
            self.main.set_status("Select a part first")
            return
        feature = self._feature_from_fields(self.feat_type.currentText())

        def add():
            part.features.append(feature)
            part.invalidate()

        self._apply(f"Add {feature.label.lower()} to {part.name}", add)

    def _rebuild(self) -> None:
        part = self._selected_part()
        if part is None:
            return
        self._apply(f"Rebuild {part.name}", lambda: part.invalidate())

    def _delete_part(self) -> None:
        part = self._selected_part()
        if part is None:
            return
        name = part.name
        self._apply(f"Delete part {name}", lambda: self.project.parts.remove(part.id))

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import geometry", "", "CAD files (*.step *.stp *.iges *.igs *.stl);;All files (*)")
        if not path:
            return
        name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext in ("step", "stp"):
            fmt = "step"
        elif ext in ("iges", "igs"):
            fmt = "iges"
        else:
            fmt = "stl"

        def do_import():
            self.project.parts.import_file(path, name)

        self._apply(f"Import {name}", do_import)

    def _export(self) -> None:
        part = self._selected_part()
        if part is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export STEP", f"{part.name}.step",
                                                        "STEP (*.step *.stp)")
        if not path:
            return
        try:
            shape = part.shape(self.project.parts.backend)
            self.project.parts.backend.export_step(shape, path)
            self.main.set_status(f"Exported {part.name} to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    def _material_changed(self, name: str) -> None:
        part = self._selected_part()
        material = next((m for m in self.project.materials.all() if m.name == name), None)
        if part is None or material is None:
            return

        def set_material():
            part.material_id = material.id
            part.physical.material_id = material.id
            part.invalidate()

        self._apply(f"Set material of {part.name}", set_material)

    def _part_selected(self) -> None:
        part = self._selected_part()
        if part is None:
            return
        self.main.viewport.highlight("")
        try:
            self.material.blockSignals(True)
            index = self.material.findText(self.project.materials.get(part.material_id).name)
            self.material.setCurrentIndex(max(index, 0))
            self.material.blockSignals(False)
        except Exception:
            self.material.blockSignals(False)
        try:
            mp = part.cad_mass_properties(self.project.parts.backend,
                                          self.project.materials.get(part.material_id).density)
            self.mass_label.setText(
                f"{part.name}: {mp.mass*1000:.2f} g from exact B-rep; "
                f"volume {getattr(mp, 'volume', 0)*1e6:.1f} cm³; provenance CAD-derived.")
        except Exception as exc:
            self.mass_label.setText(f"{part.name}: rebuild failed — {exc}")
        self._refresh_features(part)

    def _refresh_features(self, part) -> None:
        self.feature_table.setRowCount(len(part.features))
        for r, feature in enumerate(part.features):
            for c, text in enumerate((str(r + 1), feature.label, feature.operation)):
                self.feature_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(text)))


    def refresh(self) -> None:
        names = [m.name for m in self.project.materials.all()]
        current = self.material.currentText()
        self.material.blockSignals(True)
        self.material.clear()
        self.material.addItems(names)
        if current in names:
            self.material.setCurrentText(current)
        self.material.blockSignals(False)
        self.material.blockSignals(True)
        current_part = self._selected_part().id if self._selected_part() else None
        self.part_list.blockSignals(True)
        self.part_list.clear()
        for part in self.project.parts.parts.values():
            self.part_list.addItem(f"{part.name}")
        self.part_list.blockSignals(False)
        ids = list(self.project.parts.parts)
        if current_part in ids:
            self.part_list.setCurrentRow(ids.index(current_part))
        elif ids:
            self.part_list.setCurrentRow(0)
        self.material.blockSignals(False)


# ---------------------------------------------------------------------- Assembly
class AssemblyWorkspace(Workspace):
    title = "Assembly"

    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Component", "Type"])
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        left.addWidget(QtWidgets.QLabel("Assembly tree"))
        left.addWidget(self.tree, 1)

        place_box, form = form_group("Place instance")
        self.part_combo = QtWidgets.QComboBox()
        self.parent_combo = QtWidgets.QComboBox()
        self.inst_name = line_edit("")
        self.px, self.py, self.pz = (spin(-500, 500, 0, 2, "mm") for _ in range(3))
        form.addRow("Part", self.part_combo)
        form.addRow("Name (optional)", self.inst_name)
        form.addRow("Parent", self.parent_combo)
        form.addRow("X", self.px)
        form.addRow("Y", self.py)
        form.addRow("Z", self.pz)
        place = QtWidgets.QPushButton("Place component")
        place.clicked.connect(self._place)
        form.addRow(place)
        left.addWidget(place_box)
        layout.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        prop_box, prop_form = form_group("Selected component")
        self.c_name = line_edit("", True)
        self.rx, self.ry, self.rz = (spin(-360, 360, 0, 1, "°") for _ in range(3))
        self.c_grounded = QtWidgets.QCheckBox("Grounded (datum)")
        self.c_visible = QtWidgets.QCheckBox("Visible")
        self.c_visible.setChecked(True)
        self.c_material = QtWidgets.QComboBox()
        self.c_mass_mode = QtWidgets.QComboBox()
        self.c_mass_mode.addItems(["Calculated from CAD", "Manual override", "CAD + extra masses"])
        self.c_mass = spin(0, 100000, 0, 3, "g")
        apply_btn = QtWidgets.QPushButton("Apply changes")
        apply_btn.clicked.connect(self._apply_component)
        prop_form.addRow("Name", self.c_name)
        prop_form.addRow("Rotate yaw", self.rx)
        prop_form.addRow("Rotate pitch", self.ry)
        prop_form.addRow("Rotate roll", self.rz)
        prop_form.addRow(self.c_grounded)
        prop_form.addRow(self.c_visible)
        prop_form.addRow("Material", self.c_material)
        prop_form.addRow("Mass mode", self.c_mass_mode)
        prop_form.addRow("Manual mass", self.c_mass)
        prop_form.addRow(apply_btn)
        right.addWidget(prop_box)

        cons_box, cons_form = form_group("Mate constraints")
        self.cons_table = table(["Constraint", "Type", "A", "B"])
        cons_form.addRow(self.cons_table)
        row = QtWidgets.QHBoxLayout()
        self.cons_type = QtWidgets.QComboBox()
        self.cons_type.addItems([c.label() for c in ConstraintType])
        self.cons_a = QtWidgets.QComboBox()
        self.cons_b = QtWidgets.QComboBox()
        row.addWidget(self.cons_type, 2)
        row.addWidget(self.cons_a, 2)
        row.addWidget(self.cons_b, 2)
        self.cons_point = [spin(-500, 500, 0, 2, "mm") for _ in range(3)]
        self.cons_axis = [spin(-1, 1, 0, 3, "") for _ in range(3)]
        self.cons_axis[2].setValue(1)
        point_row = QtWidgets.QHBoxLayout()
        for widget in self.cons_point:
            point_row.addWidget(widget)
        axis_row = QtWidgets.QHBoxLayout()
        for widget in self.cons_axis:
            axis_row.addWidget(widget)
        cons_form.addRow("Type / A / B", row)
        cons_form.addRow("Point (mm)", point_row)
        cons_form.addRow("Axis", axis_row)
        buttons = QtWidgets.QHBoxLayout()
        add_cons = QtWidgets.QPushButton("Add constraint")
        add_cons.clicked.connect(self._add_constraint)
        remove_cons = QtWidgets.QPushButton("Remove selected")
        remove_cons.clicked.connect(self._remove_constraint)
        solve = QtWidgets.QPushButton("Solve / check DOF")
        solve.clicked.connect(self._solve)
        buttons.addWidget(add_cons)
        buttons.addWidget(remove_cons)
        buttons.addWidget(solve)
        cons_form.addRow(buttons)
        self.dof_label = QtWidgets.QLabel("—")
        self.dof_label.setWordWrap(True)
        cons_form.addRow(self.dof_label)
        right.addWidget(cons_box, 1)
        layout.addLayout(right, 5)

    def _place(self) -> None:
        row = self.part_combo.currentIndex()
        ids = list(self.project.parts.parts)
        if row < 0 or row >= len(ids):
            return
        part = self.project.parts.parts[ids[row]]
        parent = self.parent_combo.currentData()
        name = self.inst_name.text().strip() or part.name
        position = (self.px.value() / MM, self.py.value() / MM, self.pz.value() / MM)

        def place():
            comp = self.project.assembly.create_component(part.id, name, parent, position)
            comp.color = part.color
            comp.physical.material_id = part.material_id

        self._apply(f"Place {name}", place)

    def _current_component(self):
        items = self.tree.selectedItems()
        if not items:
            return None
        cid = items[0].data(0, QtCore.Qt.UserRole)
        return self.project.assembly.components.get(cid)

    def _selection_changed(self) -> None:
        comp = self._current_component()
        if comp is None:
            return
        self.main.viewport.highlight(comp.id)
        self.c_name.setText(comp.name)
        world_pos = self.project.assembly.world_transform(comp.id).translation
        for widget, value in zip((self.px, self.py, self.pz), world_pos):
            widget.setValue(float(value) * MM)
        yaw, pitch, roll = comp.transform.rotation.to_euler()
        self.rx.setValue(math.degrees(yaw))
        self.ry.setValue(math.degrees(pitch))
        self.rz.setValue(math.degrees(roll))
        self.c_grounded.setChecked(comp.grounded)
        self.c_visible.setChecked(comp.visible)
        names = [m.name for m in self.project.materials.all()]
        try:
            self.c_material.setCurrentText(self.project.materials.get(comp.physical.material_id).name)
        except Exception:
            pass
        self.c_mass_mode.setCurrentIndex({"cad": 0, "manual": 1, "cad_plus_extras": 2}.get(
            comp.physical.mass_mode.value, 0))
        self.c_mass.setValue(comp.physical.manual_mass * 1000)

    def _apply_component(self) -> None:
        comp = self._current_component()
        if comp is None:
            return
        pos = tuple(p.value() / MM for p in (self.px, self.py, self.pz))
        yaw, pitch, roll = (math.radians(w.value()) for w in (self.rx, self.ry, self.rz))
        world_before = self.project.assembly.world_transform(comp.id)
        material = next((m for m in self.project.materials.all()
                         if m.name == self.c_material.currentText()), None)
        mode = [MassMode.CAD, MassMode.MANUAL, MassMode.CAD_PLUS_EXTRAS][self.c_mass_mode.currentIndex()]

        def apply():
            comp.grounded = self.c_grounded.isChecked()
            comp.visible = self.c_visible.isChecked()
            new_world = Transform(Quat.from_euler(yaw, pitch, roll), vec3(*pos))
            self.project.assembly.set_world_transform(comp.id, new_world)
            if material is not None:
                comp.physical.material_id = material.id
            comp.physical.mass_mode = mode
            if mode is MassMode.MANUAL:
                comp.physical.manual_mass = self.c_mass.value() / 1000.0
                comp.physical.manual_com = self.project.assembly.world_transform(comp.id).translation

        self._apply(f"Edit {comp.name}", apply)

    def _add_constraint(self) -> None:
        a = self.cons_a.currentData()
        if not a:
            return
        b = self.cons_b.currentData()
        kind = ConstraintType(list(ConstraintType)[self.cons_type.currentIndex()].value)
        point = tuple(w.value() / MM for w in self.cons_point)
        axis = tuple(w.value() for w in self.cons_axis)
        if not any(axis):
            axis = (0, 0, 1)
        name = f"{kind.label()} {self.cons_a.currentText()}" + (
            f" ↔ {self.cons_b.currentText()}" if b else "")

        def add():
            self.project.assembly.constraints.append(AssemblyConstraint(
                id=f"con_{len(self.project.assembly.constraints)+1}",
                type=kind, name=name,
                a=EntityRef(a, EntityKind.AXIS, point, axis, "A"),
                b=EntityRef(b, EntityKind.AXIS, point, axis, "B") if b else None,
            ))

        self._apply(f"Add {name}", add)

    def _remove_constraint(self) -> None:
        row = selected_row(self.cons_table)
        constraints = self.project.assembly.constraints
        if row < 0 or row >= len(constraints):
            return
        target = constraints[row]
        self._apply(f"Remove {target.name}", lambda: self.project.assembly.constraints.remove(target))

    def _solve(self) -> None:
        from ..assembly.constraints import AssemblySolver
        result = AssemblySolver(self.project.assembly).solve(apply=False)
        self.dof_label.setText(result.summary())
        self.main.set_diagnostics(result.diagnostics)

    def refresh(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        by_parent: dict[str | None, list] = {}
        for comp in self.project.assembly.components.values():
            by_parent.setdefault(comp.parent_id, []).append(comp)

        def add(parent_item, parent_id):
            for comp in by_parent.get(parent_id, []):
                text = comp.name
                if comp.hardware_type:
                    text += f" [{comp.hardware_type}]"
                item = QtWidgets.QTreeWidgetItem([text, "subassembly" if comp.is_subassembly else "part"])
                item.setData(0, QtCore.Qt.UserRole, comp.id)
                if comp.grounded:
                    item.setForeground(0, QtGui.QColor("#e3b341"))
                if parent_item is None:
                    self.tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                self.tree.expandItem(item)
                add(item, comp.id)

        add(None, None)
        self.tree.blockSignals(False)

        parts = [p.name for p in self.project.parts.parts.values()]
        self.part_combo.blockSignals(True)
        self.part_combo.clear()
        for part_id, part in self.project.parts.parts.items():
            self.part_combo.addItem(part.name, part_id)
        self.part_combo.blockSignals(False)

        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItem("(top level)", None)
        for comp in self.project.assembly.components.values():
            self.parent_combo.addItem(comp.name, comp.id)
        self.parent_combo.blockSignals(False)

        cons = self.project.assembly.constraints
        self.cons_table.setRowCount(len(cons))
        self.cons_a.clear()
        self.cons_b.clear()
        self.cons_b.addItem("(world)", None)
        for comp in self.project.assembly.components.values():
            self.cons_a.addItem(comp.name, comp.id)
            self.cons_b.addItem(comp.name, comp.id)
        for r, constraint in enumerate(cons):
            names = [self.project.assembly.components[c.component_id].name
                     if c and c.component_id in self.project.assembly.components else "(world)"
                     for c in (constraint.a, constraint.b)]
            for c, text in enumerate((constraint.name, constraint.type.label(), names[0], names[1])):
                self.cons_table.setItem(r, c, QtWidgets.QTableWidgetItem(text))

        names = [m.name for m in self.project.materials.all()]
        self.c_material.clear()
        self.c_material.addItems(names)


# --------------------------------------------------------------------- Fasteners
class FastenerWorkspace(Workspace):
    title = "Fasteners"

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        scan = QtWidgets.QPushButton("Scan part holes")
        scan.clicked.connect(self._scan_holes)
        top.addWidget(scan)
        top.addStretch(1)
        layout.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("Holes"))
        self.hole_table = table(["Component", "Hole", "⌀ mm", "Depth mm", "Thread"])
        left_layout.addWidget(self.hole_table, 1)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("Installed fasteners"))
        self.fastener_table = table(["Fastener", "Hole", "Depth mm", "Engagement mm", "Clamp N",
                                     "Strip torque N·m", "Risk", "Status"])
        right_layout.addWidget(self.fastener_table, 1)
        split.addWidget(right)
        layout.addWidget(split, 1)

        install_box, form = form_group("Install fastener")
        self.thread_filter = QtWidgets.QComboBox()
        self.thread_filter.addItems(["M2", "M2.5", "M3", "M4", "M5", "M6", "8-32", "4-40"])
        self.thread_filter.setCurrentText("M3")
        self.fastener_combo = QtWidgets.QComboBox()
        self.depth = spin(0, 100, 6.0, 1, "mm")
        self.torque = spin(0, 50, 0.8, 2, "N·m")
        row = QtWidgets.QHBoxLayout()
        install = QtWidgets.QPushButton("Install into selected hole")
        install.clicked.connect(self._install)
        remove = QtWidgets.QPushButton("Remove selected fastener")
        remove.clicked.connect(self._remove)
        row.addWidget(install)
        row.addWidget(remove)
        form.addRow("Thread filter", self.thread_filter)
        form.addRow("Fastener", self.fastener_combo)
        form.addRow("Depth", self.depth)
        form.addRow("Torque (design)", self.torque)
        form.addRow(row)
        self.summary = QtWidgets.QLabel("No fasteners installed.")
        self.summary.setWordWrap(True)
        form.addRow(self.summary)
        layout.addWidget(install_box)
        self.thread_filter.currentTextChanged.connect(self._filter_fasteners)
        self.fastener_combo.currentIndexChanged.connect(self._fastener_changed)

    def _filter_fasteners(self) -> None:
        thread = self.thread_filter.currentText()
        self.fastener_combo.clear()
        for spec in self.project.fasteners.all():
            if spec.thread.designation == thread:
                self.fastener_combo.addItem(f"{spec.name}  ({spec.length*1000:.0f} mm)", spec.id)
        self._fastener_changed()

    def _fastener_changed(self) -> None:
        spec = self.project.fasteners.get(self.fastener_combo.currentData()) \
            if self.fastener_combo.currentData() else None
        if spec:
            self.depth.setValue(min(spec.length, 0.02) * 1000)

    def _scan_holes(self) -> None:
        def scan():
            added = 0
            for comp in self.project.assembly.active_components():
                if not comp.part_id:
                    continue
                try:
                    holes = self.project.parts.holes_for(comp.part_id, comp.id)
                except Exception:
                    continue
                for hole in holes:
                    if hole.id not in self.project.assembly.holes:
                        self.project.assembly.add_hole(hole)
                        added += 1
            self.main.set_status(f"Added {added} hole reference(s) from part features")

        self._apply("Scan part holes", scan)

    def _install(self) -> None:
        row = selected_row(self.hole_table)
        holes = list(self.project.assembly.holes.values())
        spec = next((s for s in self.project.fasteners.all()
                     if s.id == self.fastener_combo.currentData()), None)
        if row < 0 or row >= len(holes) or spec is None:
            self.main.set_status("Select a hole and a fastener")
            return
        hole = holes[row]
        depth = self.depth.value() / MM
        torque = self.torque.value()

        def install():
            state = InstalledFastener(
                id=f"fast_{hole.id}_{len(self.project.assembly.fasteners)+1}",
                fastener_id=spec.id, hole_id=hole.id,
                with_nut=hole.diameter > spec.thread.major_diameter + 1e-6,
            )
            state.set_depth(min(depth, hole.depth), spec.thread)
            state.applied_torque = torque
            state.installed = state.depth > 1e-6
            self.project.assembly.install(state)

        self._apply(f"Install {spec.name}", install)

    def _remove(self) -> None:
        row = selected_row(self.fastener_table)
        ids = list(self.project.assembly.fasteners)
        if row < 0 or row >= len(ids):
            return
        fid = ids[row]
        self._apply("Remove fastener", lambda: self.project.assembly.remove_fastener(fid))

    def refresh(self) -> None:
        holes = list(self.project.assembly.holes.values())
        rows = []
        for hole in holes:
            comp = self.project.assembly.components.get(hole.component_id)
            rows.append({"component": comp.name if comp else hole.component_id,
                         "hole": hole.label, "diameter": round(hole.diameter * 1000, 2),
                         "depth": round(hole.depth * 1000, 2),
                         "thread": "pre-threaded" if hole.pre_threaded else "forming"})
        fill_table(self.hole_table, rows, ["component", "hole", "diameter", "depth", "thread"])

        installer = ScrewInstaller()
        hole_by_id = {h.id: h for h in holes}
        inst_rows = []
        for state in self.project.assembly.fasteners.values():
            result = None
            spec = self.project.fasteners.get(state.fastener_id)
            hole = hole_by_id.get(state.hole_id)
            if spec is None or hole is None:
                continue
            comp = self.project.assembly.components.get(hole.component_id)
            material = self.project.materials.get(hole.material_id or
                                                  (comp.physical.material_id if comp else ""))
            try:
                result = installer.inspect(state, spec, hole,
                                           getattr(material, "ultimate_strength", None),
                                           getattr(material, "youngs_modulus", None))
                risk = round(result.strip_risk, 2) if np.isfinite(result.strip_risk) else "n/a"
                strip = round(result.strip_torque_estimate, 3) \
                    if np.isfinite(result.strip_torque_estimate) else "n/a"
                engagement = round(result.engagement_length * 1000, 2)
                status = "seated" if state.installed and not state.is_interfering(spec, hole) else "check"
            except Exception as exc:
                risk = strip = "n/a"
                engagement = round(state.depth * 1000, 2)
                status = f"error: {exc}"
            inst_rows.append({"fastener": spec.name, "hole": hole.label,
                              "depth": round(state.depth * 1000, 2), "engagement": engagement,
                              "clamp": round(result.clamp_force_estimate, 1) if result else "",
                              "strip": strip, "risk": risk, "status": status})
        fill_table(self.fastener_table, inst_rows,
                   ["fastener", "hole", "depth", "engagement", "clamp", "strip", "risk", "status"])
        self.hole_table.resizeColumnsToContents()
        self.fastener_table.resizeColumnsToContents()
        self.summary.setText(
            f"{len(holes)} hole(s), {len(inst_rows)} installed. Values are reduced-order estimates; "
            "check the Analysis workspace for the full report.")


# -------------------------------------------------------------------- Mechanics
class MechanicsWorkspace(Workspace):
    title = "Mechanics"

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.build_button = QtWidgets.QPushButton("Build mechanism")
        self.build_button.clicked.connect(self._build_world)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self._build_world)
        self.dt = spin(0.0002, 0.01, 0.001, 4, "s")
        self.contacts = QtWidgets.QCheckBox("Contacts")
        self.contacts.setChecked(False)
        self.explode = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.explode.setRange(0, 100)
        self.explode.valueChanged.connect(lambda v: self.main.viewport.set_exploded(v / 100.0))
        controls.addWidget(self.build_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(QtWidgets.QLabel("dt"))
        controls.addWidget(self.dt)
        controls.addWidget(self.contacts)
        controls.addWidget(QtWidgets.QLabel("Exploded view"))
        controls.addWidget(self.explode, 1)
        layout.addLayout(controls)

        self.servo_group, servo_form = form_group("Servo commands")
        self.servo_sliders: dict[str, tuple[QtWidgets.QSlider, QtWidgets.QLabel]] = {}
        layout.addWidget(self.servo_group)

        self.joint_table = table(["Joint", "Type", "Angle", "Rate", "Torque", "Current", "Saturated"])
        layout.addWidget(QtWidgets.QLabel("Joints"))
        layout.addWidget(self.joint_table, 1)
        self.diagnostics = DiagnosticsView()
        layout.addWidget(QtWidgets.QLabel("Diagnostics"))
        layout.addWidget(self.diagnostics, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self.world = None
        self.world_build = None

    def _build_world(self) -> None:
        from ..mech.build import build_world
        try:
            calc = AssemblyMassCalculator(self.project.parts, self.project.materials,
                                          self.project.fasteners)
            result = build_world(self.project.assembly, calc, self.project.parts,
                                 self.project.hardware, backend=self.project.parts.backend)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Mechanism build failed", str(exc))
            return
        self.world_build = result
        self.world = result.world
        self.world.contacts_enabled = self.contacts.isChecked()
        self.main.set_diagnostics(result.diagnostics)
        self.diagnostics.set_diagnostics(result.diagnostics)
        self.reset_button.setEnabled(True)
        # rebuild servo sliders
        for slider, label in self.servo_sliders.values():
            slider.setParent(None)
            label.setParent(None)
        self.servo_sliders.clear()
        form = self.servo_group.layout()
        for joint in self.world.joints.values():
            if joint.servo is None:
                continue
            channel = joint.metadata.get("tvc_channel") or joint.id
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-8000, 8000)
            slider.setValue(0)
            value = QtWidgets.QLabel("0.00°")
            slider.valueChanged.connect(
                lambda v, j=joint.id, lbl=value: (self._set_command(j, v / 1000.0), lbl.setText(f"{v/1000.0:.2f}°")))
            form.addRow(f"{joint.name} ({channel})", slider)
            form.addRow("", value)
            self.servo_sliders[joint.id] = (slider, value)
        self._update_readouts()

    def _set_command(self, joint_id: str, degrees: float) -> None:
        if self.world is None:
            return
        self.world.set_joint_target(joint_id, math.radians(degrees))

    def _toggle_play(self, playing: bool) -> None:
        if self.world is None:
            self._build_world()
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self) -> None:
        if self.world is None:
            return
        dt = self.dt.value()
        for _ in range(max(int(0.03 / dt), 1)):
            self.world.step(dt)
        self._update_readouts()
        poses = {bid: body.transform for bid, body in self.world.bodies.items()}
        self.main.viewport.show_pose(poses)

    def _update_readouts(self) -> None:
        if self.world is None:
            return
        rows = []
        for joint in self.world.joints.values():
            coord = joint.coordinate(self.world.bodies)
            rate = joint.coordinate_velocity(self.world.bodies)
            servo = joint.servo
            rows.append({
                "joint": joint.name,
                "type": joint.kind.value,
                "angle": round(math.degrees(coord), 2),
                "rate": round(math.degrees(rate), 1),
                "torque": round(float(joint.metadata.get("servo_torque", 0.0)), 4),
                "current": round(float(joint.metadata.get("servo_current", 0.0)), 3),
                "saturated": "yes" if servo and servo.state.saturated else "no",
            })
        fill_table(self.joint_table, rows,
                   ["joint", "type", "angle", "rate", "torque", "current", "saturated"])

    def refresh(self) -> None:
        if self.world is not None:
            self.play_button.setChecked(False)
            self.timer.stop()
            self.world = None
        self.joint_table.setRowCount(0)


# ------------------------------------------------------------------ Electronics
class ElectronicsWorkspace(Workspace):
    title = "Electronics"

    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        box, form = form_group("Avionics")
        self.board = QtWidgets.QComboBox()
        self.imu = QtWidgets.QComboBox()
        self.pitch_servo = QtWidgets.QComboBox()
        self.yaw_servo = QtWidgets.QComboBox()
        self.supply = spin(3.0, 8.4, 5.0, 2, "V")
        self.loop_rate = spin(50, 2000, 250, 0, "Hz")
        form.addRow("Board", self.board)
        form.addRow("IMU", self.imu)
        form.addRow("Pitch servo", self.pitch_servo)
        form.addRow("Yaw servo", self.yaw_servo)
        form.addRow("Supply", self.supply)
        form.addRow("Loop rate", self.loop_rate)
        left.addWidget(box)

        gains_box, gains_form = form_group("TVC control gains")
        self.kp = spin(0, 20, 2.2, 3)
        self.ki = spin(0, 5, 0.15, 3)
        self.kd = spin(0, 2, 0.08, 3)
        self.damping = spin(0, 5, 0.18, 3)
        self.max_gimbal = spin(0.5, 20, 8.0, 2, "°")
        gains_form.addRow("Tilt Kp", self.kp)
        gains_form.addRow("Tilt Ki", self.ki)
        gains_form.addRow("Tilt Kd", self.kd)
        gains_form.addRow("Rate damping", self.damping)
        gains_form.addRow("Max gimbal", self.max_gimbal)
        apply = QtWidgets.QPushButton("Apply electronics settings")
        apply.clicked.connect(self._apply_settings)
        gains_form.addRow(apply)
        left.addWidget(gains_box)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        buttons = QtWidgets.QHBoxLayout()
        imu_test = QtWidgets.QPushButton("IMU bench test")
        imu_test.clicked.connect(self._imu_test)
        hil_test = QtWidgets.QPushButton("Loopback HIL test")
        hil_test.clicked.connect(self._hil_test)
        sitl = QtWidgets.QPushButton("Run SITL (4 s)")
        sitl.clicked.connect(self._sitl)
        buttons.addWidget(imu_test)
        buttons.addWidget(hil_test)
        buttons.addWidget(sitl)
        right.addLayout(buttons)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Bench-test output appears here.")
        right.addWidget(self.output, 1)
        layout.addLayout(left, 2)
        layout.addLayout(right, 3)

    def _apply_settings(self) -> None:
        def apply():
            e = self.project.electrical
            e.board_id = self.board.currentData() or "pico2w"
            e.imu_id = self.imu.currentData() or "mpu6050"
            e.pitch_servo_id = self.pitch_servo.currentData() or "mg90s"
            e.yaw_servo_id = self.yaw_servo.currentData() or "mg90s"
            e.supply_voltage = self.supply.value()
            e.loop_rate_hz = self.loop_rate.value()
            self.project.control.tilt_kp = self.kp.value()
            self.project.control.tilt_ki = self.ki.value()
            self.project.control.tilt_kd = self.kd.value()
            self.project.control.rate_damping = self.damping.value()
            self.project.control.max_gimbal = math.radians(self.max_gimbal.value())

        self._apply("Electronics settings", apply)

    def _imu_model(self):
        from ..electronics.imu import IMUModel
        spec = self.project.hardware.sensor(self.project.electrical.imu_id)
        return IMUModel(spec)

    def _imu_test(self) -> None:
        imu = self._imu_model()
        samples = []
        for _ in range(1000):
            # The model adds the gravity reaction itself; a stationary sensor has zero
            # kinematic acceleration and therefore reads +1 g of specific force.
            sample = imu.measure(Transform.identity(), vec3(), vec3(), 0.001)
            if sample is not None:
                samples.append(sample)
        if not samples:
            self.output.setPlainText("No samples produced (check the IMU range/rate).")
            return
        accel = np.array([s.accel for s in samples])
        gyro = np.array([s.gyro for s in samples])
        self.output.setPlainText(
            "IMU bench test (stationary, 1 g up)\n"
            f"samples: {len(samples)}\n"
            f"accel mean (m/s²): {np.round(accel.mean(axis=0), 3).tolist()}\n"
            f"accel std  (m/s²): {np.round(accel.std(axis=0), 4).tolist()}\n"
            f"gyro mean (rad/s): {np.round(gyro.mean(axis=0), 5).tolist()}\n"
            f"gyro std  (rad/s): {np.round(gyro.std(axis=0), 5).tolist()}\n"
            f"bias estimate: {imu.bias_estimate()}\n"
            "Noise/bias come from the sensor definition in the hardware library.")

    def _hil_test(self) -> None:
        from ..electronics.hil import LoopbackHIL
        link = LoopbackHIL(lambda accel, gyro: (0.01, -0.02))
        for i in range(20):
            link.exchange(i * 0.004, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
        self.output.setPlainText("Loopback HIL\n" + str(link.report()) +
                                 "\nFrames use COBS + CRC16 framing; the real link uses the same codec on a serial port.")

    def _sitl(self) -> None:
        from ..electronics.flightcomputer import FlightComputer, ServoChannel
        from ..electronics.sitl import SoftwareInTheLoop
        from ..mech.servo import ServoPlant
        from ..physics.aero import make_aero
        from ..physics.flight import FlightSimulator
        from ..physics.rocket import build_rocket
        try:
            calc = AssemblyMassCalculator(self.project.parts, self.project.materials,
                                          self.project.fasteners)
            rocket = build_rocket(self.project.assembly, calc, self.project.parts,
                                  self.project.motor(), self.project.hardware, self.project.aero)
            sim = FlightSimulator(rocket.properties, self.project.motor(), self.project.aero,
                                  self.project.flight, rocket.stack, control=None)
            imu = self._imu_model()
            pitch = ServoPlant(self.project.hardware.servo(self.project.electrical.pitch_servo_id))
            yaw = ServoPlant(self.project.hardware.servo(self.project.electrical.yaw_servo_id))
            computer = FlightComputer(
                self.project.hardware.board(self.project.electrical.board_id),
                self.project.control, imu,
                [ServoChannel("pitch", pitch), ServoChannel("yaw", yaw)],
                supply_voltage=self.project.electrical.supply_voltage)
            sitl = SoftwareInTheLoop(sim, computer, imu, dt=0.001)
            result = sitl.run(4.0)
            summary = result.summary()
            self.output.setPlainText(
                "SITL (4 s)\n" + "\n".join(f"{k}: {v}" for k, v in summary.items()) +
                "\n\nSensor/estimator/control loop is exercised here; the flight workspace "
                "uses the ideal-attitude autopilot.")
        except Exception as exc:
            self.output.setPlainText(f"SITL failed: {exc}")

    def refresh(self) -> None:
        self.board.clear()
        for spec in self.project.hardware.boards.values():
            self.board.addItem(spec.name, spec.id)
        self.imu.clear()
        for spec in self.project.hardware.sensors.values():
            self.imu.addItem(spec.name, spec.id)
        for combo in (self.pitch_servo, self.yaw_servo):
            combo.clear()
            for spec in self.project.hardware.servos.values():
                combo.addItem(spec.name, spec.id)
        e = self.project.electrical
        try:
            self.board.setCurrentIndex(max(self.board.findData(e.board_id), 0))
            self.imu.setCurrentIndex(max(self.imu.findData(e.imu_id), 0))
            self.pitch_servo.setCurrentIndex(max(self.pitch_servo.findData(e.pitch_servo_id), 0))
            self.yaw_servo.setCurrentIndex(max(self.yaw_servo.findData(e.yaw_servo_id), 0))
        except Exception:
            pass
        self.supply.setValue(e.supply_voltage)
        self.loop_rate.setValue(e.loop_rate_hz)
        c = self.project.control
        self.kp.setValue(c.tilt_kp)
        self.ki.setValue(c.tilt_ki)
        self.kd.setValue(c.tilt_kd)
        self.damping.setValue(c.rate_damping)
        self.max_gimbal.setValue(math.degrees(c.max_gimbal))


# -------------------------------------------------------------- Propulsion / aero
class PropulsionWorkspace(Workspace):
    title = "Propulsion & aero"

    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        box, form = form_group("Motor")
        self.motor_combo = QtWidgets.QComboBox()
        self.motor_combo.currentIndexChanged.connect(self._motor_selected)
        import_btn = QtWidgets.QPushButton("Import .eng / .csv…")
        import_btn.clicked.connect(self._import_motor)
        form.addRow("Active motor", self.motor_combo)
        form.addRow(import_btn)
        self.motor_stats = QtWidgets.QLabel("—")
        self.motor_stats.setWordWrap(True)
        form.addRow(self.motor_stats)
        left.addWidget(box)
        self.thrust_plot = PlotPanel("t (s)", "Thrust (N)")
        left.addWidget(self.thrust_plot, 2)

        aero_box, aero_form = form_group("Aerodynamics")
        self.cd0 = spin(0, 3, 0.48, 3)
        self.cna = spin(0, 20, 2.0, 2)
        self.cp = spin(0, 2, 0.0, 4, "m")
        self.area = spin(0, 1, 0.0, 6, "m²")
        self.uncertainty = spin(0, 100, 25, 0, "%")
        aero_form.addRow("Cd0", self.cd0)
        aero_form.addRow("Cn_alpha /rad", self.cna)
        aero_form.addRow("CP from nose (0 = estimate)", self.cp)
        aero_form.addRow("Reference area", self.area)
        aero_form.addRow("Coefficient uncertainty", self.uncertainty)
        buttons = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton("Apply aero")
        apply_btn.clicked.connect(self._apply_aero)
        estimate = QtWidgets.QPushButton("Estimate from geometry")
        estimate.clicked.connect(self._estimate_cp)
        buttons.addWidget(apply_btn)
        buttons.addWidget(estimate)
        aero_form.addRow(buttons)
        self.aero_note = QtWidgets.QLabel("—")
        self.aero_note.setWordWrap(True)
        aero_form.addRow(self.aero_note)
        left.addWidget(aero_box, 3)
        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Barrowman stack estimate"))
        self.stack_table = table(["Element", "Kind", "Start mm", "Length mm", "⌀ mm", "Fins", "Span mm"])
        right.addWidget(self.stack_table, 1)
        self.cp_table = table(["Element", "Cn_alpha", "CP mm"])
        right.addWidget(QtWidgets.QLabel("CP contributions"))
        right.addWidget(self.cp_table, 1)
        layout.addLayout(right, 3)

    def _motor_selected(self) -> None:
        if self.motor_combo.currentData():
            self.project.active_motor_id = self.motor_combo.currentData()
        self._refresh_motor()

    def _refresh_motor(self) -> None:
        try:
            motor = self.project.motor()
        except Exception:
            return
        self.thrust_plot.clear()
        t, f = motor.plot_points(300)
        self.thrust_plot.plot_series(t, f, motor.name[:16])
        self.motor_stats.setText(
            f"{motor.name}\n"
            f"Impulse {motor.total_impulse:.1f} N·s · burn {motor.burn_time:.2f} s · "
            f"avg {motor.average_thrust:.1f} N\n"
            f"Propellant {motor.propellant_mass*1000:.0f} g · "
            f"provenance {motor.provenance.value}")
        if motor.provenance.value == "estimated":
            self.motor_stats.setStyleSheet("color:#e3b341;")

    def _import_motor(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import motor", "",
                                                        "Motor files (*.eng *.csv);;All files (*)")
        if not path:
            return

        def load():
            from ..physics.propulsion import MotorDefinition
            if path.lower().endswith(".eng"):
                motor = self.project.motors.load_eng_file(path)
            else:
                with open(path, encoding="utf-8") as fh:
                    motor = self.project.motors.add(
                        MotorDefinition.from_csv(fh.read(), name=os.path.basename(path)))
            self.project.active_motor_id = motor.id

        self._apply(f"Import motor {os.path.basename(path)}", load)

    def _apply_aero(self) -> None:
        def apply():
            a = self.project.aero
            a.cd0 = self.cd0.value()
            a.cn_alpha = self.cna.value()
            a.cp_position = self.cp.value()
            a.reference_area = self.area.value()
            a.reference_length = self.project.aero.reference_length
            a.uncertainty_percent = self.uncertainty.value()

        self._apply("Aerodynamics settings", apply)

    def _estimate_cp(self) -> None:
        from ..physics.aero import barrowman_cp
        from ..physics.rocket import build_rocket
        try:
            calc = AssemblyMassCalculator(self.project.parts, self.project.materials,
                                          self.project.fasteners)
            rocket = build_rocket(self.project.assembly, calc, self.project.parts,
                                  self.project.motor(), self.project.hardware, self.project.aero)
        except Exception as exc:
            self.aero_note.setText(f"Estimate failed: {exc}")
            return
        rows = [{"element": e.name, "kind": e.kind, "start": round(e.start * 1000, 1),
                 "length": round(e.length * 1000, 1), "diameter": round(e.diameter * 1000, 1),
                 "fins": e.fin_count, "span": round(e.fin_span * 1000, 1)} for e in rocket.stack]
        fill_table(self.stack_table, rows,
                   ["element", "kind", "start", "length", "diameter", "fins", "span"])
        estimate = barrowman_cp(rocket.stack)
        fill_table(self.cp_table, [{"element": c["element"], "cn": round(c["cn_alpha"], 2),
                                    "cp": round(c["cp_position"] * 1000, 1)}
                                   for c in estimate.contributions],
                   ["element", "cn", "cp"])
        self.cp.setValue(rocket.properties.cp_position)
        self.area.setValue(rocket.properties.reference_area)
        self.aero_note.setText(
            f"Barrowman CP {rocket.properties.cp_position*1000:.1f} mm from the tail datum, "
            f"CG {float(rocket.properties.com_dry[2])*1000:.1f} mm, "
            f"static margin {rocket.summary()['static_margin_m']*1000:+.1f} mm "
            f"(positive = stable). Estimate is subsonic and approximate.")

    def refresh(self) -> None:
        self.motor_combo.blockSignals(True)
        self.motor_combo.clear()
        for motor in self.project.motors.all():
            self.motor_combo.addItem(motor.name, motor.id)
        idx = self.motor_combo.findData(self.project.active_motor_id)
        self.motor_combo.setCurrentIndex(max(idx, 0))
        self.motor_combo.blockSignals(False)
        self.cd0.setValue(self.project.aero.cd0)
        self.cna.setValue(self.project.aero.cn_alpha)
        self.cp.setValue(self.project.aero.cp_position)
        self.area.setValue(self.project.aero.reference_area)
        self.uncertainty.setValue(self.project.aero.uncertainty_percent)
        self._refresh_motor()


# ------------------------------------------------------------------------ Flight
class FlightWorkspace(Workspace):
    title = "Flight"

    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        box, form = form_group("Flight configuration")
        self.dt = spin(0.0002, 0.01, 0.001, 4, "s")
        self.rail = spin(0, 10, 1.2, 2, "m")
        self.elevation = spin(60, 90, 90, 1, "°")
        self.wind = spin(0, 20, 0.0, 1, "m/s")
        self.chute = spin(0, 10, 0.6, 2, "m²")
        self.max_time = spin(5, 240, 40, 0, "s")
        self.use_servos = QtWidgets.QCheckBox("Use servo dynamics")
        self.use_servos.setChecked(True)
        self.use_autopilot = QtWidgets.QCheckBox("TVC autopilot (ideal attitude)")
        self.use_autopilot.setChecked(True)
        form.addRow("Timestep", self.dt)
        form.addRow("Rail length", self.rail)
        form.addRow("Launch elevation", self.elevation)
        form.addRow("Wind", self.wind)
        form.addRow("Parachute area", self.chute)
        form.addRow("Max time", self.max_time)
        form.addRow(self.use_servos)
        form.addRow(self.use_autopilot)
        run = QtWidgets.QPushButton("Run 6-DOF simulation")
        run.clicked.connect(self._run)
        form.addRow(run)
        left.addWidget(box)
        self.summary_table = table(["Quantity", "Value"])
        left.addWidget(QtWidgets.QLabel("Result"))
        left.addWidget(self.summary_table, 1)
        self.warnings = QtWidgets.QPlainTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setMaximumHeight(120)
        left.addWidget(QtWidgets.QLabel("Warnings & provenance"))
        left.addWidget(self.warnings)
        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        self.altitude_plot = PlotPanel("t (s)", "Altitude (m)")
        self.speed_plot = PlotPanel("t (s)", "Speed (m/s)")
        self.tilt_plot = PlotPanel("t (s)", "Tilt (°)")
        right.addWidget(self.altitude_plot, 1)
        right.addWidget(self.speed_plot, 1)
        right.addWidget(self.tilt_plot, 1)
        layout.addLayout(right, 3)

    def _run(self) -> None:
        from ..mech.servo import ServoPlant
        from ..physics.flight import FlightSimulator
        from ..physics.rocket import build_rocket
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            calc = AssemblyMassCalculator(self.project.parts, self.project.materials,
                                          self.project.fasteners)
            rocket = build_rocket(self.project.assembly, calc, self.project.parts,
                                  self.project.motor(), self.project.hardware, self.project.aero)
            pitch = yaw = None
            if self.use_servos.isChecked():
                pitch = ServoPlant(self.project.hardware.servo(self.project.electrical.pitch_servo_id),
                                   supply_voltage=self.project.electrical.supply_voltage)
                yaw = ServoPlant(self.project.hardware.servo(self.project.electrical.yaw_servo_id),
                                 supply_voltage=self.project.electrical.supply_voltage)
            cfg = replace(self.project.flight,
                          dt=self.dt.value(), rail_length=self.rail.value(),
                          launch_elevation_deg=self.elevation.value(),
                          max_time=self.max_time.value())
            cfg.wind = (0.0, self.wind.value(), 0.0)
            cfg.parachute_area = self.chute.value()
            sim = FlightSimulator(rocket.properties, self.project.motor(), self.project.aero,
                                  cfg, rocket.stack, pitch_servo=pitch, yaw_servo=yaw,
                                  control=self.project.control if self.use_autopilot.isChecked() else None)
            result = sim.run(self.max_time.value())
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Flight simulation failed", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.result = result
        self.main.set_diagnostics(result.diagnostics)
        summary = result.summary
        rows = [
            {"quantity": "Apogee", "value": f"{summary.get('apogee_m', 0):.1f} m"},
            {"quantity": "Time to apogee", "value": f"{summary.get('apogee_time_s', 0):.2f} s"},
            {"quantity": "Max speed", "value": f"{summary.get('max_speed_ms', 0):.1f} m/s"},
            {"quantity": "Max Mach", "value": f"{summary.get('max_mach', 0):.2f}"},
            {"quantity": "Max dynamic pressure", "value": f"{summary.get('max_dynamic_pressure_pa', 0):.0f} Pa"},
            {"quantity": "Burn time", "value": f"{summary.get('burn_time_s', 0):.2f} s"},
            {"quantity": "Total impulse", "value": f"{summary.get('total_impulse_ns', 0):.1f} N·s"},
            {"quantity": "Launch mass", "value": f"{summary.get('launch_mass_kg', 0)*1000:.0f} g"},
            {"quantity": "Static margin", "value": f"{summary.get('static_margin_m', 0)*1000:.1f} mm"},
            {"quantity": "Flight time", "value": f"{summary.get('flight_time_s', 0):.1f} s"},
            {"quantity": "Landed", "value": "yes" if summary.get("landed") else "not within run"},
        ]
        fill_table(self.summary_table, rows, ["quantity", "value"])
        self.warnings.setPlainText("\n".join(f"! {w}" for w in summary.get("warnings", [])) or
                                   "No warnings.")
        self.altitude_plot.clear()
        t, alt = result.plot_series("altitude")
        self.altitude_plot.plot_series(t, alt, "altitude", "#6ab0ff")
        t, speed = result.plot_series("speed")
        self.speed_plot.clear()
        self.speed_plot.plot_series(t, speed, "speed", "#7bd88f")
        t, tilt = result.plot_series("tilt_deg")
        self.tilt_plot.clear()
        self.tilt_plot.plot_series(t, tilt, "tilt", "#e3b341")
        self.main.set_status(
            f"Flight: apogee {summary.get('apogee_m', 0):.1f} m, max {summary.get('max_speed_ms', 0):.1f} m/s")

    def refresh(self) -> None:
        cfg = self.project.flight
        self.dt.setValue(cfg.dt)
        self.rail.setValue(cfg.rail_length)
        self.elevation.setValue(cfg.launch_elevation_deg)
        self.chute.setValue(cfg.parachute_area)
        self.max_time.setValue(cfg.max_time)


# ---------------------------------------------------------------------- Analysis
class AnalysisWorkspace(Workspace):
    title = "Analysis"

    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        box, form = form_group("Reports")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Project summary", "Mass properties", "Fasteners", "Clearance",
                            "Constraints", "Aerodynamics"])
        form.addRow("Report", self.kind)
        buttons = QtWidgets.QHBoxLayout()
        generate = QtWidgets.QPushButton("Generate")
        generate.clicked.connect(self._generate)
        save = QtWidgets.QPushButton("Save…")
        save.clicked.connect(self._save)
        csv = QtWidgets.QPushButton("Export mass CSV…")
        csv.clicked.connect(self._csv)
        buttons.addWidget(generate)
        buttons.addWidget(save)
        buttons.addWidget(csv)
        form.addRow(buttons)
        left.addWidget(box)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch(1)
        layout.addLayout(left, 1)
        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.view.setFont(font)
        layout.addWidget(self.view, 4)
        self.report = None

    def _report(self):
        from ..analysis import reports
        kind = self.kind.currentIndex()
        project = self.project
        if kind == 0:
            return reports.project_report(project)
        if kind == 1:
            return reports.mass_report(project)
        if kind == 2:
            return reports.fastener_report(project)
        if kind == 3:
            return reports.clearance_report_for_project(project)
        if kind == 4:
            return reports.joint_report(project)
        from ..physics.rocket import build_rocket
        calc = AssemblyMassCalculator(self.project.parts, self.project.materials, self.project.fasteners)
        try:
            rocket = build_rocket(project.assembly, calc, project.parts, project.motor(),
                                  project.hardware, project.aero)
        except Exception:
            rocket = None
        return reports.aero_report(project, rocket)

    def _generate(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.report = self._report()
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.view.setPlainText(f"Report failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.view.setPlainText(self.report.to_text())
        self.status.setText(f"{self.report.title} generated {self.report.generated}")

    def _save(self) -> None:
        if self.report is None:
            self._generate()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save report",
                                                        "report.txt", "Text (*.txt);;JSON (*.json)")
        if path and self.report is not None:
            self.report.save(path)
            self.main.set_status(f"Report saved to {path}")

    def _csv(self) -> None:
        from ..assembly.massprops import AssemblyMassCalculator
        calc = AssemblyMassCalculator(self.project.parts, self.project.materials, self.project.fasteners)
        result = calc.calculate(self.project.assembly)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export mass CSV", "mass.csv",
                                                        "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["item", "kind", "mass_kg", "percent", "provenance", "cg_x", "cg_y", "cg_z"])
            for row in result.rows:
                writer.writerow([row.name, row.kind, f"{row.mass:.6f}", f"{row.percent:.2f}",
                                 row.provenance, f"{row.com[0]:.6f}", f"{row.com[1]:.6f}",
                                 f"{row.com[2]:.6f}"])
        self.main.set_status(f"Mass CSV written to {path}")

    def refresh(self) -> None:
        pass
