from __future__ import annotations

import math

import pytest

import pipeline_calculator_v3 as pc


def test_segment_pipeline_count_and_indices() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 10.0

    coords = [(-100.0, 40.0), (-100.0, 40.001)]  # ~111 m northward
    _, _, distance_m = analyzer.geod.inv(coords[0][0], coords[0][1], coords[1][0], coords[1][1])
    expected = int(math.floor(abs(distance_m) / analyzer.segment_length))

    segments = analyzer.segment_pipeline(coords)

    assert len(segments) == expected
    assert [s["segment_index"] for s in segments] == list(range(len(segments)))
    assert all(abs(float(s["length"]) - analyzer.segment_length) < 1e-6 for s in segments)


def test_segment_pipeline_handles_short_line() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 1000.0
    segments = analyzer.segment_pipeline([(-100.0, 40.0), (-100.0, 40.0001)])
    assert segments == []


class _LinearGeod:
    """Exact metre coordinates isolate sampling policy from PROJ roundoff."""

    def inv(self, x1, y1, x2, y2):
        return (90 if x2 >= x1 else -90), 0, abs(x2-x1)

    def fwd(self, x, y, bearing, distance):
        return x + (distance if bearing >= 0 else -distance), y, 0


@pytest.mark.parametrize('delta', [-5e-7, -3.4e-9, 0, 3.4e-9, 5e-7])
def test_near_integral_terminal_sample_is_stable_and_clamped(delta):
    from pipeline_calculator.core.segmentation import segment_pipeline
    endpoint = 15 + delta
    coords = [(0, 0), (endpoint, 0), (endpoint, 0)]
    segments = segment_pipeline(_LinearGeod(), coords, 5)
    assert len(segments) == 3
    assert segments[-1]['midpoint'][0] == pytest.approx((10+min(endpoint, 15))/2, abs=1e-12)
    assert coords == [(0, 0), (endpoint, 0), (endpoint, 0)]


@pytest.mark.parametrize('endpoint,count', [(15-2e-6, 2), (14.5, 2), (15.25, 3)])
def test_genuine_terminal_remainders_are_not_rounded_to_full_samples(endpoint, count):
    from pipeline_calculator.core.segmentation import segment_pipeline
    segments = segment_pipeline(_LinearGeod(), [(0, 0), (endpoint, 0)], 5)
    assert len(segments) == count


def test_endpoint_allowance_does_not_accumulate_at_redundant_vertices():
    from pipeline_calculator.core.segmentation import segment_pipeline
    coords = [(i*(5-5e-7), 0) for i in range(101)]
    segments = segment_pipeline(_LinearGeod(), coords, 5)
    assert len(segments) == 99
    assert segments[-1]['midpoint'][0] == pytest.approx(492.5, abs=1e-10)


@pytest.mark.parametrize('step', [1e-7, 1e-13])
def test_tiny_steps_cannot_gain_a_whole_sample_from_absolute_allowance(step):
    from pipeline_calculator.core.segmentation import segment_pipeline
    assert segment_pipeline(_LinearGeod(), [(0, 0), (step/2, 0)], step) == []
    segments = segment_pipeline(_LinearGeod(), [(0, 0), (step*2.5, 0)], step)
    assert len(segments) == 2
    assert [s['midpoint'][0]/step for s in segments] == pytest.approx([.5, 1.5])


def test_terminal_rounding_still_obeys_segment_budget():
    from pipeline_calculator.core.segmentation import segment_pipeline
    with pytest.raises(ValueError, match='segment limit'):
        segment_pipeline(_LinearGeod(), [(0, 0), (15-5e-7, 0)], 5, max_segments=2)


def test_corridor_terminal_rounding_clamps_to_source_and_rejects_real_overrun():
    from pipeline_calculator.core.corridor_coverage import qualified_path_runs
    from pipeline_calculator.core.segmentation import segment_pipeline
    geod = _LinearGeod()
    endpoint = 15-5e-7
    pipes = [{'id': i, 'coordinates': [(0, 0), (endpoint, 0)],
              'segments': segment_pipeline(geod, [(0, 0), (endpoint, 0)], 5)} for i in range(2)]
    qualified = dict(pair=(0, 1), paths=(0, 0), segment_ids=([0, 1, 2], [0, 1, 2]))
    runs = qualified_path_runs(pipes, qualified, 5, geod)
    assert all(run.end_m == endpoint and run.coordinates[-1] == (endpoint, 0) for run in runs)
    pipes[0]['coordinates'][-1] = (15-2e-6, 0)
    with pytest.raises(ValueError, match='exceeds its original path'):
        qualified_path_runs(pipes, qualified, 5, geod)

