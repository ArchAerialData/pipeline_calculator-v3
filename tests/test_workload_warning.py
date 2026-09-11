from types import SimpleNamespace
import time

import numpy as np
import pytest
from scipy.spatial import KDTree

from pipeline_calculator.core import workload
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController
from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
from pipeline_calculator.gui.state import AnalysisParameters


def input_file(tmp_path):
    path = tmp_path / 'pair.kml'
    path.write_text('<kml>' + ''.join(
        f'<Placemark><LineString><coordinates>{-100+i*.00002},40 '
        f'{-100+i*.00002},40.003</coordinates></LineString></Placemark>'
        for i in range(2)) + '</kml>')
    return path


def wait_for_warning(job):
    deadline = time.monotonic() + 5
    while job.context.workload_warning() is None:
        assert not job.done.is_set(), job.error
        assert time.monotonic() < deadline
        job.done.wait(.01)


@pytest.mark.parametrize('continue_run', [False, True])
@pytest.mark.parametrize('warning_stage', ['segments', 'density'])
def test_real_job_pauses_before_overlap_and_cancels_or_continues(tmp_path, monkeypatch, continue_run, warning_stage):
    if warning_stage == 'segments':
        monkeypatch.setattr(workload, 'SEGMENT_WARNING_THRESHOLD', 1)
    else:
        monkeypatch.setattr(workload, 'CANDIDATE_WARNING_THRESHOLD', 1)
    path = input_file(tmp_path)
    baseline = PipelineAnalyzer().analyze_complete(path)
    job = AnalysisController().start(str(path), AnalysisParameters())
    try:
        wait_for_warning(job)
        assert job.result is None and not job.done.is_set()
        assert job.context.snapshot().stage == 'Waiting for your choice'
        if continue_run:
            job.context.accept_workload()
        else:
            job.cancel()
        assert job.done.wait(5)
        assert job.error is None
        if continue_run:
            assert job.state == 'completed' and job.result == baseline
        else:
            assert job.state == 'cancelled' and job.result is None
    finally:
        job.cancel()
        job.done.wait(5)


class RecordingContext(ExecutionContext):
    def __init__(self):
        super().__init__(interactive=True)
        self.warnings = []

    def confirm_workload(self, message):
        self.warnings.append(message)


def test_high_thresholds_and_small_or_sparse_workloads_are_quiet():
    context = RecordingContext()
    workload.check_segment_workload(749_999, context)
    points = np.column_stack((np.arange(5000)*100, np.zeros(5000), np.zeros(5000)))
    workload.check_density_workload(KDTree(points), points, 16, context)
    assert not context.warnings
    workload.check_segment_workload(750_000, context)
    assert len(context.warnings) == 1
    assert '750,000' in context.warnings[0]


def test_extreme_density_warns_without_materializing_neighbor_lists():
    context = RecordingContext()
    points = np.zeros((3163, 3))
    workload.check_density_workload(KDTree(points), points, 16, context)
    assert len(context.warnings) == 1
    assert '10,004,569' in context.warnings[0]


def test_density_check_is_bounded_and_cancellable():
    context = RecordingContext()
    calls = []
    def count(*args, **kwargs):
        assert kwargs == {'return_length': True}
        calls.append(1)
        if len(calls) == 3:
            context.cancel()
        return 0
    with pytest.raises(AnalysisCancelled):
        workload.check_density_workload(SimpleNamespace(query_ball_point=count),
                                       np.zeros((4000, 3)), 16, context)
    assert len(calls) == 3


def test_hard_limit_preserves_source_mileage_without_allocating_segments(tmp_path, monkeypatch):
    from pipeline_calculator.core import analyzer
    path = input_file(tmp_path)
    baseline = PipelineAnalyzer().analyze_complete(path)
    monkeypatch.setattr(analyzer, 'MAX_ANALYSIS_SEGMENTS', 1)
    def forbidden(*args, **kwargs):
        pytest.fail('Over-limit input must not allocate analysis segments')
    monkeypatch.setattr(PipelineAnalyzer, 'find_parallel_segments', forbidden)
    result = PipelineAnalyzer().analyze_complete(path)
    assert result['total_meters'] == baseline['total_meters']
    assert not result['analysis_complete'] and result['overlap_analysis'] is None
    assert result['diagnostics'][-1]['code'] == 'overlap_analysis_failed'


@pytest.mark.parametrize('action', ['continue', 'cancel', 'close'])
@pytest.mark.native_gui
def test_native_warning_controls(tmp_path, monkeypatch, action):
    import customtkinter as ctk
    monkeypatch.setattr(workload, 'SEGMENT_WARNING_THRESHOLD', 1)
    root = ctk.CTk()
    root.withdraw()
    session = AnalysisSession(root, lambda job: None)
    try:
        session.start(input_file(tmp_path), AnalysisParameters())
        wait_for_warning(session.job)
        root.update()
        # Let the scheduled Tk poll present the warning without a second poll chain.
        deadline = time.monotonic() + 5
        while not session.warning_visible:
            root.update()
            assert time.monotonic() < deadline
            session.job.done.wait(.01)
        assert 'exceptionally large' in session.label.cget('text')
        assert session.continue_button.winfo_manager() == 'pack'
        if action == 'continue':
            session.continue_button.invoke()
        elif action == 'cancel':
            session.cancel_button.invoke()
        else:
            session.close()
        assert session.job.done.wait(5)
        assert session.job.state == ('completed' if action == 'continue' else 'cancelled')
        root.update()
    finally:
        session.close()
        root.destroy()
