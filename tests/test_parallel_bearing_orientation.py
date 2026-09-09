from __future__ import annotations

from pipeline_calculator.core.angles import bearing_orientation_diff


def test_bearing_orientation_diff_treats_antiparallel_as_parallel() -> None:
    assert bearing_orientation_diff(0.0, 180.0) == 0.0
    assert bearing_orientation_diff(10.0, 190.0) == 0.0
    assert bearing_orientation_diff(350.0, 170.0) == 0.0


def test_bearing_orientation_diff_perpendicular_is_90() -> None:
    assert bearing_orientation_diff(0.0, 90.0) == 90.0
    assert bearing_orientation_diff(45.0, 135.0) == 90.0


def test_bearing_orientation_diff_near_parallel_is_small() -> None:
    assert bearing_orientation_diff(0.0, 5.0) == 5.0
    assert bearing_orientation_diff(0.0, 175.0) == 5.0

