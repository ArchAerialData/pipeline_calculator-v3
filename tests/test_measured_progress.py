import threading
import time
from types import SimpleNamespace
import pytest

from pipeline_calculator.core.execution import ExecutionContext
from pipeline_calculator.core.progress import RuntimeProjection, WorkProgress


def test_real_counts_advance_progress_without_reversing_or_timer_animation():
    progress = WorkProgress()
    assert progress.update('Reading documents', 0, None) == 0
    assert progress.update('Calculating source lengths', 5, 10) == pytest.approx(.075)
    assert progress.update('Segmenting path', 500, 1000) == pytest.approx(.225)
    assert progress.update('Segmenting paths', 400, 1000) == pytest.approx(.225)
    assert progress.update('Waiting for your choice', 0, None) == pytest.approx(.225)
    assert progress.update('Searching neighbors', 500, 1000) == pytest.approx(.615)
    assert progress.update('Finalizing results', 0, None) == .99
    assert progress.update('Complete', 1, 1) == 1


def test_runtime_projection_requires_sustained_observed_work_over_60_seconds():
    estimate = RuntimeProjection()
    assert estimate.observe('Searching neighbors', 0, 1000, 0) is None
    assert estimate.observe('Searching neighbors', 30, 1000, 3) is None
    assert estimate.observe('Searching neighbors', 30, 1000, 4) is None  # no new work
    assert estimate.observe('Searching neighbors', 40, 1000, 4) == pytest.approx(100)
    # Exactly a minute is not over the requested trigger.
    estimate = RuntimeProjection()
    estimate.observe('Searching neighbors', 0, 600, 0)
    assert estimate.observe('Searching neighbors', 30, 600, 3) is None
    assert estimate.observe('Searching neighbors', 40, 600, 4) is None
    assert estimate.observe('Building corridors', 0, None, 61) == 61


def test_wwm_sized_fast_run_and_transient_slow_sample_are_quiet():
    estimate = RuntimeProjection()
    estimate.observe('Segmenting paths', 0, 754586, 0)
    assert estimate.observe('Segmenting path', 754586, 754586, 4) is None
    estimate.observe('Searching neighbors', 0, 754586, 5)
    for elapsed in range(6, 22):
        assert estimate.observe('Searching neighbors', (elapsed-5)*754586//16, 754586, elapsed) is None
    estimate = RuntimeProjection()
    estimate.observe('Searching neighbors', 0, 1000, 0)
    assert estimate.observe('Searching neighbors', 1, 1000, 3) is None
    assert estimate.observe('Searching neighbors', 500, 1000, 4) is None


def test_clock_starts_with_worker_and_new_job_resets(monkeypatch):
    from pipeline_calculator.gui.controllers import analysis_controller as controller
    from pipeline_calculator.gui.state import AnalysisParameters
    clock = [0.0]
    seen = []
    def analyze(*args, context, **kwargs):
        seen.append(context.elapsed_seconds())
        clock[0] += 2
        context.report('Searching neighbors', 5, 10)
        return {'analysis_complete': True}
    monkeypatch.setattr(controller, 'analyze_file', analyze)
    for ready_at in (500, 5000):
        job = controller.AnalysisJob('file.kmz', AnalysisParameters())
        job.context = ExecutionContext(clock=lambda: clock[0])
        clock[0] = ready_at
        assert job.context.elapsed_seconds() == 0 and job.context.snapshot() is None
        job.start()
        assert job.done.wait(3) and job.state == 'completed'
        assert job.context.snapshot().fraction == 1
        assert job.context.elapsed_seconds() == 2
        clock[0] += 100
        assert job.context.elapsed_seconds() == 2
    assert seen == [0, 0]


def test_warning_wait_is_excluded_from_processing_clock():
    clock = [0.0]
    context = ExecutionContext(clock=lambda: clock[0], interactive=True)
    context.begin()
    clock[0] = 5
    waiter = threading.Thread(target=context.confirm_workload, args=('test warning',))
    waiter.start()
    until = time.monotonic()+3
    while context.workload_warning() is None:
        assert time.monotonic() < until
        time.sleep(.005)
    clock[0] = 105
    assert context.elapsed_seconds() == 5
    context.accept_workload()
    waiter.join(3)
    assert not waiter.is_alive()
    clock[0] = 107
    assert context.elapsed_seconds() == 7


@pytest.mark.native_gui
def test_green_percentage_bar_cancel_below_and_next_import_reset():
    import customtkinter as ctk
    from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
    from pipeline_calculator.gui.state import AnalysisParameters
    root = ctk.CTk()
    root.geometry('1000x700')
    clock = [0.0]
    class Controller:
        def start(self, *args):
            context = ExecutionContext(clock=lambda: clock[0])
            context.begin()
            context.report('Reading documents')
            self.job = SimpleNamespace(job_id=context.job_id, context=context, state='running',
                                       done=threading.Event(), cancel=lambda: None)
            return self.job
    def settle():
        until = time.monotonic()+.16
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
    controller = Controller()
    session = AnalysisSession(root, lambda job: None, controller)
    try:
        session.start('first.kmz', AnalysisParameters())
        settle()
        assert session.bar.cget('mode') == 'determinate' and session.bar.get() == 0
        assert session.percent_label.cget('text') == '0%'
        clock[0] = 10
        controller.job.context.report('Searching neighbors', 500, 1000)
        settle()
        assert session.bar.get() == pytest.approx(.615)
        assert session.percent_label.cget('text') == '61%'
        assert 'Elapsed: 10 s' in session.label.cget('text')
        assert session.cancel_button.winfo_rooty() > session.bar.winfo_rooty()+session.bar.winfo_height()
        assert session.cancel_button.cget('fg_color') == '#E5B94F'
        assert session.bar.cget('progress_color') == '#40B873'
        session.cancel_button.invoke()
        assert session.cancel_button.cget('state') == 'disabled'
        assert session.bar.get() < 1
        session.close()
        clock[0] = 5000
        session = AnalysisSession(root, lambda job: None, controller)
        session.start('second.kmz', AnalysisParameters())
        settle()
        assert session.bar.get() == 0 and session.percent_label.cget('text') == '0%'
        assert 'Elapsed: 0 s' in session.label.cget('text')
        assert session.cancel_button.cget('state') == 'normal'
        assert session.filename_label.cget('text') == 'second.kmz'
    finally:
        session.close()
        root.destroy()
