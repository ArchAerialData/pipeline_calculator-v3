from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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
        subprocess.run(["open", path], check=False)
        return
    subprocess.run(["xdg-open", path], check=False)


def open_overlap_corridor(section: dict[str, Any], index: int) -> str:
    path = write_corridor_kml_tempfile(section, index)
    try:
        open_path(path)
    except Exception:
        # Opening is best-effort; writing is the main deliverable.
        pass
    return path

