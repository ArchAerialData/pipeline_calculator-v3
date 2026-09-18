"""Atomic, organized export packages for a completed geography result snapshot."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from pipeline_calculator.export.geography_kmz import write_geography_kmz
from pipeline_calculator.export.xlsx import build_analysis_workbook
from pipeline_calculator.core.corridor_geometry import prepare_scope_visualizations


def prepare_export_snapshot(results):
    """Preflight requested map visuals before writing reports, without UI mutation."""
    snapshot = prepare_scope_visualizations(results)
    geography = snapshot['geography'] = dict(results['geography'])
    geography['states'] = [prepare_scope_visualizations(state, state_code=state['state_code'])
                           for state in geography.get('states', [])]
    return snapshot


def _filename(value):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")[:100]
    if not name or name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        name = "analysis_" + name
    return name


def export_analysis_package(results, output_parent, current_file=None, *, include_maps=True, include_json=False):
    """Write every scope together and publish only when all requested files succeed.

    A sibling exclusive lock reserves a collision-safe name across simultaneous
    exports. The final directory appears through one rename on the same filesystem.
    """
    if not isinstance(results.get("geography"), dict):
        raise ValueError("A state breakdown result is required to export a geography package")
    from pipeline_calculator.export.repair_provenance import validate_input_repair
    validate_input_repair(results)
    parent = Path(output_parent).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Choose an existing folder for the export package")
    source = Path(current_file).stem if current_file else "pipeline"
    base_name = f"{_filename(source)}_analysis_{datetime.now():%Y%m%d_%H%M%S}"
    serial = 1
    while True:
        name = base_name if serial == 1 else f"{base_name}_{serial}"
        destination = parent / name
        lock = parent / f".{name}.export.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            serial += 1
            continue
        os.close(descriptor)
        if destination.exists():
            lock.unlink()
            serial += 1
            continue
        break
    staging = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=parent))
        if include_maps:
            results = prepare_export_snapshot(results)
        build_analysis_workbook(results).save(staging / "analysis.xlsx")
        if include_json:
            with (staging / "analysis.json").open("w", encoding="utf-8") as stream:
                json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
        geography = results["geography"]
        fragments = geography.get("fragments", [])
        if include_maps and fragments:
            combined = staging / "Combined"
            combined.mkdir()
            write_geography_kmz(results, combined / "analysis.kmz")
            interior_codes = {code for fragment in fragments if fragment["kind"] == "state"
                              for code in fragment["state_codes"]}
            for state in sorted(geography.get("states", []), key=lambda state: state["state_name"]):
                if state["state_code"] not in interior_codes:
                    continue
                folder = staging / "States" / _filename(state["state_name"])
                folder.mkdir(parents=True)
                write_geography_kmz(results, folder / "analysis.kmz", state["state_code"])
        if destination.exists():
            raise FileExistsError(f"Export destination appeared during export: {destination}")
        staging.rename(destination)
        staging = None
        return destination
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)
