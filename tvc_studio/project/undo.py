"""Snapshot-based undo/redo.

Engineering edits are structured (features, constraints, instances), so snapshots of the
serialised project are simple, exact and cheap enough for interactive use. Each entry keeps
the label shown in the Edit menu and the full project state before/after the edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UndoEntry:
    label: str
    before: dict
    after: dict


class UndoStack:
    def __init__(self, limit: int = 60) -> None:
        self.limit = limit
        self.entries: list[UndoEntry] = []
        self.index = 0            # number of applied entries
        self._pending: tuple[str, dict] | None = None

    # -- recording ---------------------------------------------------------
    def begin(self, label: str, project) -> None:
        self._pending = (label, project.to_dict())

    def commit(self, project) -> bool:
        """Finish an edit started with ``begin``; returns False when nothing changed."""
        if self._pending is None:
            return False
        label, before = self._pending
        self._pending = None
        after = project.to_dict()
        if before == after:
            return False
        if self.index < len(self.entries):
            self.entries = self.entries[:self.index]
        self.entries.append(UndoEntry(label, before, after))
        if len(self.entries) > self.limit:
            self.entries = self.entries[-self.limit:]
        self.index = len(self.entries)
        return True

    def abort(self) -> None:
        self._pending = None

    # -- navigation --------------------------------------------------------
    def can_undo(self) -> bool:
        return self.index > 0

    def can_redo(self) -> bool:
        return self.index < len(self.entries)

    def undo_label(self) -> str:
        return self.entries[self.index - 1].label if self.can_undo() else ""

    def redo_label(self) -> str:
        return self.entries[self.index].label if self.can_redo() else ""

    def undo(self) -> dict | None:
        if not self.can_undo():
            return None
        self.index -= 1
        return self.entries[self.index].before

    def redo(self) -> dict | None:
        if not self.can_redo():
            return None
        entry = self.entries[self.index]
        self.index += 1
        return entry.after

    def clear(self) -> None:
        self.entries.clear()
        self.index = 0
        self._pending = None
