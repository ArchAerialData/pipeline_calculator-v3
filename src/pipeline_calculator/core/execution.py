"""Cooperative execution control. No GUI dependencies or unbounded message queues."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from uuid import uuid4


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


class ExecutionContext:
    def __init__(self, job_id=None, *, clock=time.monotonic, interactive=False):
        self.job_id = job_id or uuid4().hex
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._clock = clock
        self.started_at = clock()
        self._last_emit = float('-inf')
        self._snapshot = None
        self._ticks = 0
        self.interactive = interactive
        self.workload_accepted = False
        self._workload_warning = None
        self._workload_decision = threading.Event()

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
        try:
            while not self._workload_decision.wait(0.1):
                self.check()
            self.check()
        finally:
            with self._lock:
                self._workload_warning = None

    def check(self):
        if self.cancel_event.is_set():
            raise AnalysisCancelled('Analysis cancelled')

    def checkpoint(self):
        """Cheap work-item checkpoint, checking the event every 256 items."""
        self._ticks += 1
        if self._ticks & 255 == 0:
            self.check()

    def report(self, stage, completed=0, total=None):
        self.check()
        now = self._clock()
        with self._lock:
            old = self._snapshot
            if old is not None and old.stage == stage and now - self._last_emit < 0.1:
                return
            self._snapshot = ProgressSnapshot(self.job_id, 1 if old is None else old.sequence + 1,
                                              stage, completed, total, now - self.started_at)
            self._last_emit = now

    def snapshot(self):
        with self._lock:
            return self._snapshot
