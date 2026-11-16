"""Real supplied ZIPs: verify every source path, not just plausible total mileage."""
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import pytest
import shapefile
from pyproj import Geod

from pipeline_calculator.importing import prepare_import
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from gis_helpers import diagnostic_codes


ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT / "docs/validation/gis-import-samples.json").read_text())


@pytest.mark.parametrize("sample", REPORT["samples"], ids=lambda sample: sample["dataset"])
def test_customer_zip_preserves_all_records_paths_and_reference_lengths(sample):
    source_path = ROOT / sample["fixture"]
    original = source_path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == sample["sha256"]
    session = prepare_import(source_path)
    assert len(session.layers) == 1
    parsed = session.fresh_parse()
    expected = sample["counts"]
    assert len(parsed.pipelines) == expected["nonnull_geometry_records"]
    assert not parsed.placemarks
    assert len({p["id"] for p in parsed.pipelines}) == len(parsed.pipelines)
    assert sum(len(p["coordinate_paths"]) for p in parsed.pipelines) == expected["parts"]
    assert sum(len(path) for p in parsed.pipelines for path in p["coordinate_paths"]) == expected["xy_vertices"]
    assert bool("null_geometry" in diagnostic_codes(parsed)) == bool(expected["null_geometry_records"])
    geod = Geod(ellps="GRS80")
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        stem = sample["dataset"]
        with shapefile.Reader(**{ext: io.BytesIO(archive.read(f"{stem}.{ext}"))
                                 for ext in ("shp", "shx", "dbf")}, encoding="utf-8") as reader:
            shapes = list(reader.iterShapes())
    by_id = {int(p["source_feature_id"]): p for p in parsed.pipelines}
    null_indices = []
    measured = []
    reference = sample["independent_reference"]["per_record_horizontal_length_m"]
    for index, shape in enumerate(shapes):
        if shape.shapeType == shapefile.NULL:
            assert index not in by_id and reference[index] is None
            null_indices.append(index)
            continue
        starts = list(shape.parts)
        source_paths = [[tuple(xy) for xy in shape.points[start:end]]
                        for start, end in zip(starts, starts[1:] + [len(shape.points)])]
        actual_paths = by_id[index]["coordinate_paths"]
        assert actual_paths == source_paths  # Explicit mainland EPSG:1188 is XY no-op.
        length = math.fsum(geod.line_length(*zip(*coords)) for coords in actual_paths)
        assert length == pytest.approx(reference[index], abs=.001, rel=1e-10)
        measured.append(length)
    assert null_indices == sample["null_geometry_record_indices_zero_based"]
    assert math.fsum(measured) == pytest.approx(
        sample["independent_reference"]["total_horizontal_length_m"], abs=.001, rel=1e-10)
    rows, total, _ = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)
    assert len(rows) == expected["nonnull_geometry_records"]
    assert total == pytest.approx(math.fsum(measured), abs=.001, rel=1e-10)
    assert source_path.read_bytes() == original


@pytest.mark.parametrize("sample", REPORT["samples"], ids=lambda sample: sample["dataset"])
def test_customer_zip_and_loose_components_have_same_geometry(tmp_path, sample):
    archive_path = ROOT / sample["fixture"]
    archived = prepare_import(archive_path).fresh_parse()
    stem = sample["dataset"]
    with zipfile.ZipFile(archive_path) as archive:
        for ext in ("shp", "shx", "dbf", "prj", "cpg"):
            (tmp_path / f"{stem}.{ext}").write_bytes(archive.read(f"{stem}.{ext}"))
    loose = prepare_import(tmp_path / f"{stem}.shp").fresh_parse()
    assert [(p["source_feature_id"], p["name"], p["coordinate_paths"]) for p in loose.pipelines] == [
        (p["source_feature_id"], p["name"], p["coordinate_paths"]) for p in archived.pipelines]
    assert diagnostic_codes(loose) == diagnostic_codes(archived)


@pytest.mark.parametrize("sample", REPORT["samples"], ids=lambda sample: sample["dataset"])
@pytest.mark.parametrize("state_breakdown", [False, True], ids=["combined", "states"])
def test_customer_archives_through_desktop_analysis_route(sample, state_breakdown):
    from pipeline_calculator.core.options import AnalysisOptions
    from pipeline_calculator.gui.controllers.analysis_controller import analyze_file
    from pipeline_calculator.gui.state import AnalysisParameters
    retained = []
    # Source mileage is independent of overlap sampling. Keep these import-route
    # tests bounded by preventing qualifying survey corridors; separate core
    # tests cover normal overlap thresholds and buffer generation.
    result = analyze_file(str(ROOT / sample["fixture"]),
                          AnalysisParameters(segment_length=1000, min_parallel_length=1e9),
                          options=AnalysisOptions(state_breakdown=state_breakdown),
                          prepare_repair=True, on_source=retained.append)
    assert len(result["pipelines"]) == sample["counts"]["nonnull_geometry_records"]
    assert result["total_meters"] == pytest.approx(
        sample["independent_reference"]["total_horizontal_length_m"], abs=.001, rel=1e-10)
    assert result["analysis_complete"] is (sample["counts"]["null_geometry_records"] == 0)
    assert retained and retained[0].verified
    assert result["parsed_sources"] and result["import_metadata"]
    assert not result.get("parsed_kml_files")
    if state_breakdown:
        geography = result["geography"]
        assert geography["reconciliation"]["passed"]
        assert {state["state_code"] for state in geography["states"]} == {"TX"}
