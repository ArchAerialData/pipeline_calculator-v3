"""Cooperative execution control. No GUI dependencies or unbounded message queues."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from uuid import uuid4
from pipeline_calculator.core.progress import WorkProgress, RuntimeProjection


class AnalysisCancelled(Exception):
    """Control flow, not a malformed input or unavailable-overlap result."""


@dataclass(frozen=True)
class ProgressSnapshot:
    job_id: str
    sequence: int
    stage: str
    completed: int
    total: int | None
    elapsed_seconds: float
    fraction: float = 0.0


class ExecutionContext:
    def __init__(self, job_id=None, *, clock=time.monotonic, interactive=False):
        self.job_id = job_id or uuid4().hex
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._clock = clock
        self.started_at = None
        self.finished_at = None
        self._paused_at = None
        self._paused_seconds = 0.0
        self.progress = WorkProgress()
        self.runtime_projection = RuntimeProjection()
        self.estimated_segments = None
        self._last_emit = float('-inf')
        self._snapshot = None
        self._ticks = 0
        self.interactive = interactive
        self.workload_accepted = False
        self._workload_warning = None
        self._workload_decision = threading.Event()

    def begin(self):
        if self.started_at is None:
            self.started_at = self._clock()

    def elapsed_seconds(self):
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else (
            self._paused_at if self._paused_at is not None else self._clock())
        return max(0.0, end-self.started_at-self._paused_seconds)

    def finish(self):
        self.report('Complete', 1, 1)
        self.finished_at = self._clock()

    def cancel(self):
        self.cancel_event.set()
        self._workload_decision.set()

    def workload_warning(self):
        with self._lock:
            return self._workload_warning

    def accept_workload(self):
        with self._lock:
            if self._workload_warning is None or self.cancel_event.is_set():
                return
            self.workload_accepted = True
            self._workload_warning = None
            self._workload_decision.set()

    def confirm_workload(self, message):
        """Pause only GUI jobs; the GUI polls and responds on its own thread."""
        self.check()
        if not self.interactive or self.workload_accepted:
            return
        self.report('Waiting for your choice')
        with self._lock:
            self._workload_decision.clear()
            self._workload_warning = message
            self._paused_at = self._clock()
        try:
            while not self._workload_decision.wait(0.1):
                self.check()
            self.check()
        finally:
            with self._lock:
                self._workload_warning = None
                if self._paused_at is not None:
                    self._paused_seconds += self._clock()-self._paused_at
                    self._paused_at = None

    def check(self):
        if self.cancel_event.is_set():
            raise AnalysisCancelled('Analysis cancelled')

    def checkpoint(self):
        """Cheap work-item checkpoint, checking the event every 256 items."""
        self._ticks += 1
        if self._ticks & 255 == 0:
            self.check()

    def report(self, stage, completed=0, total=None, *, fraction=None):
        self.check()
        self.begin()
        now = self._clock()
        elapsed = self.elapsed_seconds()
        with self._lock:
            old = self._snapshot
            if old is not None and old.stage == stage and now - self._last_emit < 0.1:
                return
            if fraction is None:
                progress = self.progress.update(stage, completed, total)
            else:
                progress = max(self.progress.fraction, min(1.0, max(0.0, fraction)))
                self.progress.fraction = progress
            self._snapshot = ProgressSnapshot(self.job_id, 1 if old is None else old.sequence + 1,
                                              stage, completed, total, elapsed, progress)
            self._last_emit = now
        if self.interactive and not self.workload_accepted:
            estimate = self.runtime_projection.observe(stage, completed, total, elapsed)
            if estimate is not None:
                from pipeline_calculator.core.workload import runtime_warning
                self.confirm_workload(runtime_warning(estimate, elapsed))

    def snapshot(self):
        with self._lock:
            return self._snapshot


class ScopedExecutionContext:
    """Map a sequential sub-analysis into one interval of its parent's progress.

    Cancellation, warnings and elapsed time belong to the parent job. A child
    completing never completes the job or resets its clock. This adapter also
    lets geometry stages report work counts without extending the ordinary
    analyzer's fixed progress schedule.
    """

    def __init__(self, parent, start, end, label=''):
        if not 0 <= start <= end <= 1:
            raise ValueError('Invalid progress interval')
        self.parent, self.start, self.end, self.label = parent, start, end, label
        self.progress = WorkProgress()

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def report(self, stage, completed=0, total=None, *, fraction=None):
        from pipeline_calculator.core.progress import STAGES
        if fraction is not None:
            self.progress.fraction = max(self.progress.fraction, min(1.0, max(0.0, fraction)))
        elif stage in STAGES:
            self.progress.update(stage, completed, total)
        elif total and total > 0:
            self.progress.fraction = max(self.progress.fraction, min(1.0, completed / total))
        label = f'{self.label}: {stage}' if self.label else stage
        self.parent.report(label, completed, total,
                           fraction=self.start + (self.end-self.start)*self.progress.fraction)

    def finish(self):
        self.report('Complete', 1, 1)
