from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any
from uuid import uuid4

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext, ScopedExecutionContext
from pipeline_calculator.core.options import AnalysisOptions


class RepairRequired(Exception):
    """Prepared source awaiting an explicit decision, not an analysis failure."""

    def __init__(self, source_session):
        super().__init__('This source requires verified repair before analysis.')
        self.source_session = source_session


def analyze_file(file_path: str, params: AnalysisParameters, *, context=None,
                 options: AnalysisOptions | None = None, prepare_repair=False,
                 source_session=None, approve_repair=False, on_source=None) -> dict[str, Any]:
    """Library calls stay strict; desktop jobs enable the decision workflow.

    Source acquisition, approval and analysis all run on the worker. A completed
    source snapshot is reported before analysis so later failure/cancellation does
    not discard verified input or make retry reopen changing source files.
    """
    analyzer = PipelineAnalyzer(
        detection_range=params.detection_range,
        min_parallel_length=params.min_parallel_length,
        segment_length=params.segment_length,
        angular_tolerance=params.angular_tolerance,
    )
    kwargs = {'options': options} if options is not None else {}
    if not prepare_repair and not approve_repair and source_session is None:
        return analyzer.analyze_complete(file_path, context=context, **kwargs)
    from pipeline_calculator.parsers.source import prepare_source
    preparation = ScopedExecutionContext(context, 0, .12) if context is not None else None
    source = source_session or prepare_source(file_path, context=preparation)
    if on_source is not None:
        on_source(source)
    if source.requires_repair and not source.verified:
        if not approve_repair:
            raise RepairRequired(source)
        source.approve(context=preparation)
    parsed = source.fresh_parse(context=preparation)
    if context is not None:
        context.report('Source verified' if source.requires_repair else 'Input ready', 1, 1, fraction=.12)
    analysis_context = ScopedExecutionContext(context, .12, .99) if context is not None else None
    result = analyzer.analyze_parsed(parsed, context=analysis_context, **kwargs)
    if source.requires_repair:
        report = source.report
        report['combined_analysis_status'] = 'complete' if result.get('analysis_complete') else 'incomplete'
        report['geography_status'] = (result.get('geography') or {}).get('status', 'not_requested')
        result['input_repair'] = report
    return result


@dataclass
class AnalysisJob:
    file_path: str
    params: AnalysisParameters
    options: AnalysisOptions = field(default_factory=AnalysisOptions)
    source_session: Any = field(default=None, repr=False)
    approve_repair: bool = False
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
        self._options = self.options
        self._source_request = self.source_session
        self._approve_request = bool(self.approve_repair)
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
            if self._state in ('completed', 'failed', 'cancelled', 'repair_required'):
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
        awaiting_repair = False
        try:
            self.context.check()
            self.context.begin()
            result = analyze_file(
                *self._request, context=self.context, options=self._options,
                prepare_repair=True, source_session=self._source_request,
                approve_repair=self._approve_request, on_source=self._retain_source)
            self.context.finish()
        except RepairRequired as decision:
            self.source_session = decision.source_session
            awaiting_repair = True
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
                elif awaiting_repair:
                    self._state = 'repair_required'
                else:
                    self._state, self.result = 'completed', result
                self.done.set()

    def _retain_source(self, source):
        # Immutable in-memory snapshots remain alive while any job/UI owns them.
        with self._lock:
            self.source_session = source


class AnalysisController:
    """Runs analysis off the UI thread."""

    def start(self, file_path: str, params: AnalysisParameters, *,
              options: AnalysisOptions | None = None, source_session=None,
              approve_repair=False) -> AnalysisJob:
        job = AnalysisJob(file_path=file_path, params=params, options=options or AnalysisOptions(),
                          source_session=source_session, approve_repair=approve_repair)
        job.start()
        return job

