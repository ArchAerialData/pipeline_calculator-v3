from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from pipeline_calculator.core.constants import (
    ANGULAR_TOLERANCE,
    DEFAULT_DETECTION_RANGE,
    MIN_PARALLEL_LENGTH,
    SEGMENT_LENGTH,
)


def _format_number(value: float) -> str:
    if isinstance(value, (int,)) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return str(value)


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if raw_str == "":
        return None
    # Accept common formatting like "1,234.5"
    raw_str = raw_str.replace(",", "")
    try:
        value = float(raw_str)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class AnalysisParameters:
    """Validated analysis parameters (no tkinter dependencies)."""

    detection_range: float = float(DEFAULT_DETECTION_RANGE)
    min_parallel_length: float = float(MIN_PARALLEL_LENGTH)
    segment_length: float = float(SEGMENT_LENGTH)
    angular_tolerance: float = float(ANGULAR_TOLERANCE)

    @classmethod
    def from_strings(
        cls,
        detection_range: str | None,
        min_parallel_length: str | None,
        segment_length: str | None,
        angular_tolerance: str | None,
        *,
        defaults: "AnalysisParameters | None" = None,
    ) -> tuple["AnalysisParameters", dict[str, str]]:
        """Parse parameters from UI strings.

        - Invalid/empty values fall back to defaults and are returned in the
          `corrections` dict so the UI may reset the text field.
        - Valid numbers are clamped to the same bounds as the legacy GUI, but
          clamping is not reported as a correction (legacy does not auto-edit
          the input fields when clamping).
        """

        base = defaults or cls()
        corrections: dict[str, str] = {}

        det = _parse_float(detection_range)
        if det is None:
            det = base.detection_range
            corrections["detection_range"] = _format_number(base.detection_range)

        min_par = _parse_float(min_parallel_length)
        if min_par is None:
            min_par = base.min_parallel_length
            corrections["min_parallel_length"] = _format_number(base.min_parallel_length)

        seg = _parse_float(segment_length)
        if seg is None:
            seg = base.segment_length
            corrections["segment_length"] = _format_number(base.segment_length)

        ang = _parse_float(angular_tolerance)
        if ang is None:
            ang = base.angular_tolerance
            corrections["angular_tolerance"] = _format_number(base.angular_tolerance)

        # Clamp to legacy bounds used in `PipelineCalculatorGUI.process_file()`.
        det_clamped = max(1.0, float(det))
        min_par_clamped = max(10.0, float(min_par))
        seg_clamped = max(1.0, float(seg))
        ang_clamped = max(1.0, min(90.0, float(ang)))

        return (
            cls(
                detection_range=det_clamped,
                min_parallel_length=min_par_clamped,
                segment_length=seg_clamped,
                angular_tolerance=ang_clamped,
            ),
            corrections,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "detection_range": float(self.detection_range),
            "min_parallel_length": float(self.min_parallel_length),
            "segment_length": float(self.segment_length),
            "angular_tolerance": float(self.angular_tolerance),
        }


@dataclass
class AppState:
    current_file: str | None = None
    current_results: dict[str, Any] | None = None
    params: AnalysisParameters = field(default_factory=AnalysisParameters)

