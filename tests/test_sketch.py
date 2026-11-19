import math

import numpy as np
import pytest

from tvc_studio.cad.sketch import Sketch, SketchConstraintType, circle_sketch, rectangle_sketch


def test_rectangle_sketch_is_solved_and_closed():
    sketch = rectangle_sketch(0.04, 0.02)
    result = sketch.solve()
    assert result.residual < 1e-8
    loops = sketch.profile_loops()
    assert len(loops) == 1
    assert len(loops[0]) == 4


def test_circle_sketch_profile():
    sketch = circle_sketch(0.012)
    sketch.solve()
    loops = sketch.profile_loops()
    assert len(loops) == 1


def test_sketch_rectangle_dimensions():
    sketch = rectangle_sketch(0.05, 0.03)
    loops = sketch.profile_loops()
    xs = [p[0] for seg in loops[0] for p in (seg.start, seg.end)]
    ys = [p[1] for seg in loops[0] for p in (seg.start, seg.end)]
    assert max(xs) - min(xs) == pytest.approx(0.05, rel=1e-6)
    assert max(ys) - min(ys) == pytest.approx(0.03, rel=1e-6)


def test_sketch_constraint_solver_moves_free_point():
    sketch = Sketch("constrained")
    p1 = sketch.add_point(0.0, 0.0, fixed=True)
    p2 = sketch.add_point(0.1, 0.05)
    line = sketch.add_line(p1, p2)
    sketch.constrain(SketchConstraintType.HORIZONTAL, [line.id])
    sketch.solve()
    assert abs(p2.y - p1.y) < 1e-6


def test_sketch_serialisation_round_trip():
    sketch = rectangle_sketch(0.03, 0.02)
    restored = Sketch.from_dict(sketch.to_dict())
    assert len(restored.lines) == len(sketch.lines)
    restored.solve()
    assert len(restored.profile_loops()) == 1
