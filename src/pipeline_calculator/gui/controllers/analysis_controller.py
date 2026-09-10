from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any
from uuid import uuid4

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext


def analyze_file(file_path: str, params: AnalysisParameters, *, context=None) -> dict[str, Any]:
    analyzer = PipelineAnalyzer(
        detection_range=params.detection_range,
        min_parallel_length=params.min_parallel_length,
        segment_length=params.segment_length,
        angular_tolerance=params.angular_tolerance,
    )
    return analyzer.analyze_complete(file_path, context=context)


@dataclass
class AnalysisJob:
    file_path: str
    params: AnalysisParameters
    done: threading.Event = field(default_factory=threading.Event, init=False)
    result: dict[str, Any] | None = field(default=None, init=False)
    error: BaseException | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _job_id: str = field(default_factory=lambda: uuid4().hex, init=False, repr=False)
    _state: str = field(default='pending', init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    context: ExecutionContext = field(init=False, repr=False)

    def __post_init__(self):
        self._request = (self.file_path, self.params)
        self.context = ExecutionContext(self.job_id, interactive=True)

    @property
    def job_id(self):
        return self._job_id

    @property
    def state(self):
        with self._lock:
            return self._state

    def cancel(self):
        with self._lock:
            if self._state in ('completed', 'failed', 'cancelled'):
                return
            self.context.cancel()
            if self._state == 'pending':
                self._state = 'cancelled'
                self.done.set()
            else:
                self._state = 'cancellation_requested'

    def start(self) -> None:
        with self._lock:
            if self._state == 'cancelled' and self._thread is None:
                return
            if self._state != 'pending':
                raise RuntimeError('AnalysisJob can only be started once')
            self._state = 'running'
            self._thread = threading.Thread(target=self._worker, daemon=True)
            try:
                self._thread.start()
            except BaseException as exc:
                self.error = exc
                self._state = 'failed'
                self.done.set()
                raise

    def _worker(self) -> None:
        result = error = None
        try:
            self.context.check()
            result = analyze_file(*self._request, context=self.context)
        except AnalysisCancelled:
            self.context.cancel()
        except BaseException as e:  # noqa: BLE001 - report to UI
            error = e
        finally:
            with self._lock:
                if self.context.cancel_event.is_set():
                    self._state = 'cancelled'
                elif error is not None:
                    self._state, self.error = 'failed', error
                else:
                    self._state, self.result = 'completed', result
                self.done.set()


class AnalysisController:
    """Runs analysis off the UI thread."""

    def start(self, file_path: str, params: AnalysisParameters) -> AnalysisJob:
        job = AnalysisJob(file_path=file_path, params=params)
        job.start()
        return job

