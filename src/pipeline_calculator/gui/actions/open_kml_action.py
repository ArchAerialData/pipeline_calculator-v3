from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml


def write_corridor_kml_tempfile(section: dict[str, Any], index: int) -> str:
    kml = build_overlap_corridor_kml(section, index)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=f"_corridor_{index:03d}.kml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(kml)
        return tmp.name


def open_path(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # nosec - user requested open
        return
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=True, capture_output=True, timeout=10)
        return
    subprocess.run(["xdg-open", path], check=True, capture_output=True, timeout=10)


@dataclass(frozen=True)
class LaunchOutcome:
    path: str
    status: str
    error: str | None = None


class CorridorLaunchError(OSError):
    def __init__(self, outcome):
        self.path = outcome.path
        super().__init__(f'KML saved to {self.path}, but opening failed: {outcome.error}')


def launch_saved_corridor(path: str) -> LaunchOutcome:
    try:
        open_path(path)
    except (OSError, subprocess.SubprocessError) as exc:
        details = getattr(exc, 'stderr', None)
        if isinstance(details, bytes):
            details = details.decode('utf-8', errors='replace')
        return LaunchOutcome(path, 'failed', str(details or exc)[-1000:])
    return LaunchOutcome(path, 'requested')


def create_and_launch_corridor(section: dict[str, Any], index: int) -> LaunchOutcome:
    return launch_saved_corridor(write_corridor_kml_tempfile(section, index))


def open_overlap_corridor(section: dict[str, Any], index: int) -> str:
    """Compatibility wrapper: a launch error retains the successfully written path."""
    outcome = create_and_launch_corridor(section, index)
    if outcome.status == 'failed':
        raise CorridorLaunchError(outcome)
    return outcome.path

