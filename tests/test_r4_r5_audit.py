"""Adversarial resource, sampling, lifecycle and geometry audit regressions."""
import copy
import itertools
import math
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import KDTree

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.core import overlap, segmentation, coordinates, workload
from pipeline_calculator.export import geometry_validation as geometry
from scripts.validation.common import geographic


class RecordingContext(ExecutionContext):
    def __init__(self):
        super().__init__(interactive=True)
        self.warnings = []

    def confirm_workload(self, message):
        self.warnings.append(message)


def test_periodic_sparse_samples_do_not_hide_massively_dense_input():
    # Equally spaced samples all land on isolated points, missing >98% density.
    points = np.zeros((255*64+1, 3))
    points[::64, 0] = np.arange(1, 257)*1000
    context = RecordingContext()
    workload.check_density_workload(KDTree(points), points, 16, context)
    assert len(context.warnings) == 1


def test_trailing_remainder_does_not_consume_full_segment_budget(monkeypatch):
    monkeypatch.setattr(segmentation, 'MAX_ANALYSIS_SEGMENTS', 1)
    a = PipelineAnalyzer(segment_length=5)
    start = (-100, 40)
    end = a.geod.fwd(*start, 0, 5.5)[:2]
    assert len(a.segment_pipeline([start, end])) == 1


def test_project_budget_is_enforced_before_overallocation(monkeypatch):
    monkeypatch.setattr(overlap, 'MAX_ANALYSIS_SEGMENTS', 4)
    a = PipelineAnalyzer()
    # Three full segments per disjoint path, six in one pipeline.
    pipes = [{'name': 'multi', 'coordinate_paths': [
        [geographic((x, 0)), geographic((x, 15.1))] for x in (0, 20)]}]
    generated = []
    original = segmentation.segment_pipeline
    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        generated.extend(result)
        return result
    monkeypatch.setattr(coordinates, 'segment_pipeline', track)
    with pytest.raises(ValueError, match='segment limit'):
        a.find_parallel_segments(pipes)
    assert len(generated) <= 4


def test_ring_input_budget_precedes_projection_and_sorting(monkeypatch):
    monkeypatch.setattr(geometry, 'MAX_RING_POINTS', 8, raising=False)
    consumed = []
    def many_points():
        for i in range(100):
            consumed.append(i)
            # Distinct valid coordinates; budget must fire before area/topology.
            yield (i*.00001, (i % 2)*.001)
    with pytest.raises(ValueError, match='point limit'):
        geometry.validated_ring(many_points())
    assert len(consumed) <= 9


def test_repeated_fallback_object_is_checked_only_once(monkeypatch):
    calls = []
    def invalid(points):
        calls.append(points)
        raise ValueError('bad ring')
    monkeypatch.setattr(geometry, 'validated_ring', invalid)
    bad = [(0, 0), (0, 0)]
    with pytest.raises(ValueError, match='No usable'):
        geometry.prepare_geometry({'corridor_polygon': bad, 'oriented_polygon': bad})
    assert len(calls) == 1


@pytest.mark.parametrize('cancel_when', ['before', 'publishing', 'waiting', 'after_accept'])
def test_warning_cancellation_races_never_publish_success(cancel_when):
    published = threading.Event()
    class Context(ExecutionContext):
        def report(self, stage, *args, **kwargs):
            result = super().report(stage, *args, **kwargs)
            if stage == 'Waiting for your choice':
                if cancel_when == 'publishing':
                    self.cancel()
                published.set()
            return result
    context = Context(interactive=True)
    if cancel_when == 'before':
        context.cancel()
    outcomes = []
    # A held final check makes accept/cancel ordering deterministic.
    original_check = context.check
    accepted_check = threading.Event()
    release = threading.Event()
    def check():
        if cancel_when == 'after_accept' and context.workload_accepted:
            accepted_check.set()
            assert release.wait(5)
        original_check()
    context.check = check
    def run():
        try:
            context.confirm_workload('warning')
            outcomes.append('success')
        except AnalysisCancelled:
            outcomes.append('cancelled')
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        if cancel_when in ('waiting', 'after_accept'):
            assert published.wait(5)
            # Wait on the lock-protected warning without arbitrary sleeps.
            for _ in range(500):
                if context.workload_warning() is not None:
                    break
                threading.Event().wait(.01)
            assert context.workload_warning() is not None
            if cancel_when == 'after_accept':
                context.accept_workload()
                assert accepted_check.wait(5)
            context.cancel()
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert outcomes == ['cancelled'] and context.workload_warning() is None
    finally:
        context.cancel(); release.set(); thread.join(5)


def test_savings_order_invariance_with_duplicate_names_and_shifted_groups():
    rng = random.Random(1405)
    for _ in range(30):
        a = PipelineAnalyzer(min_parallel_length=20, detection_range=15)
        pipes = []
        for i in range(4):
            x, y, length = rng.choice([0, 4, 8, 12, 16, 20]), rng.choice([0, 2.5, 10, 30]), rng.choice([40, 60, 90])
            pipes.append({'name': 'same', 'coordinates': [geographic((x, y)), geographic((x, y+length))]})
        totals = []
        for order in itertools.islice(itertools.permutations(pipes), 6):
            data = copy.deepcopy(order)
            result = a.calculate_overlap_results(data, a.find_parallel_segments(data))
            totals.append(result['savings_meters'])
        assert max(totals)-min(totals) < 1e-6


def test_extreme_finite_segment_size_cannot_break_source_mileage(tmp_path):
    from scripts.validation.common import write_fixture, parallel
    path = write_fixture(tmp_path/'tiny-step.kml', parallel())
    baseline = PipelineAnalyzer().analyze_complete(path)
    result = PipelineAnalyzer(segment_length=1e-320).analyze_complete(path)
    assert result['total_meters'] == baseline['total_meters']
    assert not result['analysis_complete']
    assert result['diagnostics'][-1]['code'] == 'overlap_analysis_failed'
    assert 'segment limit' in result['diagnostics'][-1]['context']['error']
    zero = write_fixture(tmp_path/'zero.kml', parallel(length=0))
    result = PipelineAnalyzer(segment_length=1e-320).analyze_complete(zero)
    assert result['total_meters'] == 0 and result['analysis_complete']


@pytest.mark.parametrize('size', ['800x400', '400x400'])
@pytest.mark.native_gui
def test_warning_controls_fit_small_windows_with_long_names(size):
    import customtkinter as ctk
    from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
    root = ctk.CTk(); root.withdraw(); root.geometry(size)
    context = ExecutionContext(interactive=True)
    context._workload_warning = workload.warning_text('Estimated work: 999,999,999 nearby checks.')
    job = SimpleNamespace(context=context, done=threading.Event(), state='running',
                          job_id=context.job_id, cancel=context.cancel)
    session = AnalysisSession(root, lambda j: None, controller=SimpleNamespace(start=lambda *a:job))
    try:
        session.start('x'*240+'.kmz', None)
        root.update_idletasks()
        # Tk reports physical pixels; CTk geometry takes logical pixels. The
        # withdrawn host may still report its initial 200px geometry on Windows.
        available_height = root._apply_window_scaling(400*.85)
        available_width = root._apply_window_scaling(int(size.split('x')[0])*.9-20)
        assert session.frame.winfo_reqheight() <= available_height
        assert session.controls.winfo_reqwidth() <= available_width
        assert session.continue_button.winfo_manager() == 'pack'
        # The text body can grow without increasing the reserved controls' height.
        height = session.frame.winfo_reqheight()
        session.label.configure(text=context._workload_warning*5)
        root.update_idletasks()
        assert session.frame.winfo_reqheight() == height
    finally:
        session.close(); root.destroy()


def test_topology_matches_independent_checker_for_seeded_shapes():
    from scripts.validation.geometry import local_points, ring_checks
    rng = random.Random(905)
    accepted = rejected = 0
    for _ in range(60):
        points = [geographic((r*math.cos(i*2*math.pi/12), r*math.sin(i*2*math.pi/12)))
                  for i in range(12) for r in [rng.uniform(30, 100)]]
        if rng.random() < .5:
            rng.shuffle(points)
        rounded = [geometry.coordinate(p) for p in points]
        ring = rounded + [rounded[0]]
        checks = ring_checks(local_points(ring, ring[0]))
        if checks['self_intersections']:
            with pytest.raises(ValueError, match='crosses|area'):
                geometry.validated_ring(points)
            rejected += 1
        else:
            assert geometry.validated_ring(points)[1]
            accepted += 1
    assert accepted and rejected


def test_point_limit_falls_back_without_rejecting_small_valid_shapes(monkeypatch):
    monkeypatch.setattr(geometry, 'MAX_RING_POINTS', 4)
    square = [(0, 0), (.001, 0), (.001, .001), (0, .001)]
    assert geometry.validated_ring(square)[1]
    ring, _, kind, reason = geometry.prepare_geometry({
        'corridor_polygon': square * 3, 'oriented_polygon': square})
    assert kind == 'oriented_rectangle' and 'point limit' in reason
    assert ring[0] == ring[-1]


def test_segment_budget_carries_between_pipelines(monkeypatch):
    monkeypatch.setattr(overlap, 'MAX_ANALYSIS_SEGMENTS', 4)
    a = PipelineAnalyzer()
    pipes = [{'name': str(i), 'coordinates': [geographic((i*2, 0)), geographic((i*2, 15.1))]}
             for i in range(2)]
    called = []
    original = coordinates.segment_pipeline
    def track(*args, **kwargs):
        called.append(kwargs['max_segments'])
        return original(*args, **kwargs)
    monkeypatch.setattr(coordinates, 'segment_pipeline', track)
    with pytest.raises(ValueError, match='segment limit'):
        a.find_parallel_segments(pipes)
    assert called == [4, 1]


def test_density_sample_is_repeatable_bounded_and_skips_accepted_jobs():
    points = np.zeros((17003, 3))
    points[:, 0] = np.arange(len(points))
    samples = []
    def count(point, radius, **kwargs):
        assert kwargs['return_length'] is True
        samples.append(int(point[0]))
        return 1
    context = RecordingContext()
    tree = SimpleNamespace(query_ball_point=count)
    workload.check_density_workload(tree, points, 16, context)
    first = list(samples)
    samples.clear()
    workload.check_density_workload(tree, points, 16, context)
    assert samples == first and len(set(first)) == workload.DENSITY_SAMPLE_SIZE
    for i, index in enumerate(first):
        assert i*len(points)//len(first) <= index < (i+1)*len(points)//len(first)
    context.workload_accepted = True
    samples.clear()
    workload.check_density_workload(tree, points, 16, context)
    assert not samples and not context.warnings
