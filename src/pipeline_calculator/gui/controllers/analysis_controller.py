from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.gui.state import AnalysisParameters


def analyze_file(file_path: str, params: AnalysisParameters) -> dict[str, Any]:
    analyzer = PipelineAnalyzer(
        detection_range=params.detection_range,
        min_parallel_length=params.min_parallel_length,
        segment_length=params.segment_length,
        angular_tolerance=params.angular_tolerance,
    )
    return analyzer.analyze_complete(file_path)


@dataclass
class AnalysisJob:
    file_path: str
    params: AnalysisParameters
    done: threading.Event = field(default_factory=threading.Event, init=False)
    result: dict[str, Any] | None = field(default=None, init=False)
    error: BaseException | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            self.result = analyze_file(self.file_path, self.params)
        except BaseException as e:  # noqa: BLE001 - report to UI
            self.error = e
        finally:
            self.done.set()


class AnalysisController:
    """Runs analysis off the UI thread."""

    def start(self, file_path: str, params: AnalysisParameters) -> AnalysisJob:
        job = AnalysisJob(file_path=file_path, params=params)
        job.start()
        return job

