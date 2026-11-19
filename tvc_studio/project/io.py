"""Saving and loading project files.

JSON on purpose: inspectable, diffable, no vendor lock. Geometry that cannot be rebuilt from
features is embedded as base64 BREP (exact), not as a mesh. Writes are atomic (temp file +
replace) so an interrupted save cannot destroy the previous project.
"""

from __future__ import annotations

import json
import os
import tempfile

from ..core.errors import ProjectFormatError
from .schema import Project

PROJECT_EXTENSION = ".tvcproj"


class JsonProjectStore:
    """Reads/writes :class:`Project` documents."""

    def __init__(self, backend=None) -> None:
        self.backend = backend

    # -- save --------------------------------------------------------------
    def save(self, project: Project, path: str) -> str:
        if not path.lower().endswith(PROJECT_EXTENSION):
            path = path + PROJECT_EXTENSION
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        payload = project.to_dict()
        fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        project.path = path
        project.touch()
        return path

    # -- load --------------------------------------------------------------
    def load(self, path: str) -> Project:
        if not os.path.exists(path):
            raise ProjectFormatError(f"project file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ProjectFormatError(f"project file is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or "assembly" not in payload:
            raise ProjectFormatError("file is not a TVC Engineering Studio project")
        try:
            project = Project.from_dict(payload, backend=self.backend)
        except Exception as exc:
            raise ProjectFormatError(f"could not load project: {exc}") from exc
        project.path = path
        return project

    # -- import/export helpers --------------------------------------------
    def export_step(self, project: Project, path: str) -> None:
        if self.backend is None:
            raise ProjectFormatError("CAD backend unavailable; cannot export STEP")
        shapes = []
        for part_id in project.parts.parts:
            try:
                shapes.append(project.parts.shape(part_id))
            except Exception:
                continue
        if not shapes:
            raise ProjectFormatError("no exportable solid geometry in the project")
        self.backend.export_step(self.backend.compound(shapes), path)

    def export_mass_report_csv(self, project: Project, path: str, rows: list[dict]) -> str:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["name", "mass"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path


def recent_files_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "TVCStudio", "recent.json")


def load_recent_files() -> list[str]:
    path = recent_files_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [p for p in data.get("recent", []) if isinstance(p, str) and os.path.exists(p)][:10]
    except Exception:
        return []


def remember_recent_file(path: str) -> None:
    recent = load_recent_files()
    recent = [p for p in recent if os.path.abspath(p) != os.path.abspath(path)]
    recent.insert(0, os.path.abspath(path))
    target = recent_files_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({"recent": recent[:10]}, fh, indent=2)
    except Exception:
        pass


def library_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "TVCStudio", "library.json")


class ComponentLibraryStore:
    """Reusable hardware and subassembly definitions across projects."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or library_path()
        self.entries: list[dict] = []
        self.load()

    def add(self, entry: dict) -> None:
        entry = dict(entry)
        entry.setdefault("id", f"lib_{len(self.entries)+1}")
        self.entries = [e for e in self.entries if e.get("id") != entry["id"]] + [entry]
        self.save()

    def by_kind(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e.get("kind") == kind]

    def get(self, entry_id: str) -> dict | None:
        return next((e for e in self.entries if e.get("id") == entry_id), None)

    def remove(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.get("id") != entry_id]
        if len(self.entries) != before:
            self.save()
            return True
        return False

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.entries = list(data.get("entries", []))
            except Exception:
                self.entries = []

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"entries": self.entries}, fh, indent=2)
        except Exception:
            pass
