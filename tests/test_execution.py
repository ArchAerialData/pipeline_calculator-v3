from __future__ import annotations

import threading
from types import SimpleNamespace
import time

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.gui.controllers import analysis_controller as controller
from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
from pipeline_calculator.gui.state import AnalysisParameters


def fixture(tmp_path):
    path = tmp_path / 'parallel.kml'
    path.write_text('<kml>' + ''.join(
        f'<Placemark><name>{i}</name><LineString><coordinates>'
        f'{-100+i*0.00002},40 {-100+i*0.00002},40.003'
        '</coordinates></LineString></Placemark>' for i in range(3)) + '</kml>')
    return str(path)


@pytest.mark.parametrize('stage', ['Reading documents', 'Calculating source lengths',
    'Segmenting paths', 'Segmenting path', 'Building spatial index', 'Searching neighbors',
    'Qualifying sections', 'Building corridors', 'Building group graph',
    'Sorting group candidates', 'Calculating savings', 'Finalizing results'])
def test_cancellation_bypasses_all_error_boundaries(tmp_path, stage):
    class CancelAtStage(ExecutionContext):
        def report(self, current, *args):
            if stage == current:
                self.cancel()
            return super().report(current, *args)
    analyzer = PipelineAnalyzer()
    with pytest.raises(AnalysisCancelled):
        analyzer.analyze_complete(fixture(tmp_path), context=CancelAtStage())
    # Per-call context is restored, so a subsequent synchronous analysis can succeed.
    assert analyzer.analyze_complete(fixture(tmp_path))['analysis_complete']


def test_inner_coordinate_loop_cancels(tmp_path):
    path = tmp_path / 'long.kml'
    path.write_text('<kml><Placemark><LineString><coordinates>' + ' '.join(
        f'-100,{40+i/1000000}' for i in range(10000)) + '</coordinates></LineString></Placemark></kml>')
    class CancelInLoop(ExecutionContext):
        def checkpoint(self):
            super().checkpoint()
            if self._ticks == 512:
                self.cancel()
    context = CancelInLoop()
    with pytest.raises(AnalysisCancelled):
        PipelineAnalyzer().analyze_complete(path, context=context)
    assert context._ticks <= 768


def test_progress_throttles_without_queue_and_preserves_stage_changes():
    now = [0.0]
    context = ExecutionContext(clock=lambda: now[0])
    context.report('Reading documents', 0)
    for count in range(10000):
        context.report('Reading documents', count)
    assert context.snapshot().sequence == 1
    context.report('Searching neighbors', 0, 10)
    assert context.snapshot().sequence == 2
    now[0] = 1
    context.report('Searching neighbors', 5, 10)
    assert context.snapshot().completed == 5
    context.cancel()
    with pytest.raises(AnalysisCancelled):
        context.report('Finalizing results')


def test_prestart_and_terminal_cancel(tmp_path):
    job = controller.AnalysisJob(fixture(tmp_path), AnalysisParameters())
    job.cancel(); job.cancel(); job.start()
    assert job.done.is_set() and job.state == 'cancelled' and job._thread is None
    success = controller.AnalysisController().start(fixture(tmp_path), AnalysisParameters())
    assert success.done.wait(10)
    success.cancel()
    assert success.state == 'completed' and success.result is not None
    with pytest.raises(RuntimeError):
        success.start()


@pytest.mark.parametrize('fail', [False, True])
def test_cancel_wins_before_terminal_publication(monkeypatch, fail):
    entered, release = threading.Event(), threading.Event()
    def work(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if fail:
            raise ValueError('work error before publication')
        return {'partial': True}
    monkeypatch.setattr(controller, 'analyze_file', work)
    job = controller.AnalysisController().start('fake.kml', AnalysisParameters())
    assert entered.wait(5)
    job.cancel(); job.cancel()
    assert job.state == 'cancellation_requested' and not job.done.is_set()
    release.set()
    assert job.done.wait(5)
    job._thread.join(5)
    assert not job._thread.is_alive()
    assert job.state == 'cancelled' and job.result is None and job.error is None


def test_thread_start_failure_publishes_done(monkeypatch):
    def fail(*args):
        raise RuntimeError('thread unavailable')
    monkeypatch.setattr(threading.Thread, 'start', fail)
    job = controller.AnalysisJob('fake', AnalysisParameters())
    with pytest.raises(RuntimeError):
        job.start()
    assert job.done.is_set() and job.state == 'failed'


def test_context_preserves_float_callbacks_and_outputs(tmp_path):
    path = fixture(tmp_path)
    events = []
    result = PipelineAnalyzer().analyze_complete(path, events.append, context=ExecutionContext())
    assert events and all(isinstance(value, float) for value in events)
    assert result == PipelineAnalyzer().analyze_complete(path)


def test_session_close_cancels_poll_and_stale_completion(tmp_path):
    removed, callbacks = [], []
    root = SimpleNamespace(after_cancel=removed.append)
    session = AnalysisSession(root, callbacks.append)
    session.job = controller.AnalysisJob(fixture(tmp_path), AnalysisParameters())
    session.poll_id = 'poll-1'
    session.close()
    session._poll(session.job.job_id)
    assert removed == ['poll-1'] and not callbacks
    assert session.job.state == 'cancelled'


@pytest.mark.parametrize('legacy', [False, True])
def test_gui_rejects_stale_job_completion(legacy):
    if legacy:
        from pipeline_calculator_v3 import PipelineCalculatorGUI
    else:
        from pipeline_calculator.gui.main_window import PipelineCalculatorGUI
    gui = PipelineCalculatorGUI.__new__(PipelineCalculatorGUI)
    gui._closing = False
    gui._processing = True
    gui._analysis_session = SimpleNamespace(job=object())
    gui._analysis_done(SimpleNamespace(state='completed', result={'old': True}))
    assert gui._processing


@pytest.mark.native_gui
def test_native_session_controls_and_close(tmp_path):
    import customtkinter as ctk
    root = ctk.CTk()
    root.withdraw()
    session = AnalysisSession(root, lambda job: None)
    try:
        session.start(fixture(tmp_path), AnalysisParameters())
        session.cancel_button.invoke()
        assert session.cancel_button.cget('state') == 'disabled'
        assert 'Cancelling' in session.label.cget('text')
        session.close()
        assert session.job.done.wait(10)
        root.update()
    finally:
        session.close()
        root.destroy()
