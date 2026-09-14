import json
import math

import pytest
from pyproj import Geod
from shapely.geometry import Polygon, MultiPolygon, box

from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor, load_boundaries, partition_pipelines

GEOD = Geod(ellps="GRS80")


def dataset():
    return BoundaryDataset({"AA": box(-2, -2, 0, 2), "BB": box(0, -2, 2, 2)}, {"AA": "Alpha", "BB": "Beta"})


def pipeline(coords=None, paths=None):
    return {"name": "Same name", "objectid": "duplicate", "placemark_id": "same",
            "coordinates": coords or [], "coordinate_paths": paths or []}


def mileage(result, kind, code=None):
    return math.fsum(f["length_meters"] for f in result["fragments"]
                     if f["kind"] == kind and (code is None or code in f["state_codes"]))


def test_exact_two_state_split_conserves_geodesic_and_coordinates():
    result = partition_pipelines([pipeline([(-1, 0), (1, 0)])], GEOD, boundaries=dataset())
    expected = GEOD.inv(-1, 0, 1, 0)[2]
    assert result["reconciliation"]["passed"]
    assert mileage(result, "state", "AA") == pytest.approx(expected/2, abs=0.001)
    assert mileage(result, "state", "BB") == pytest.approx(expected/2, abs=0.001)
    assert result["crossing_count"] == 1
    assert all(f["kind"] == "state" for f in result["fragments"])
    json.dumps(result, allow_nan=False)


def test_shared_border_single_ledger_and_direction_vertex_invariance():
    variants = [[(0, -.01), (0, .01)], [(0, .01), (0, -.01)], [(0, -.01), (0, 0), (0, .01)]]
    expected = GEOD.inv(0, -.01, 0, .01)[2]
    for coords in variants:
        result = partition_pipelines([pipeline(coords)], GEOD, boundaries=dataset())
        assert mileage(result, "shared") == pytest.approx(expected, abs=0.001)
        assert mileage(result, "state") == 0
        assert all(f["state_codes"] == ["AA", "BB"] for f in result["fragments"])
        assert result["reconciliation"]["passed"]


def test_close_parallel_line_is_not_snapped_and_tiny_crossing_survives():
    result = partition_pipelines([pipeline([(-1e-8, -.01), (-1e-8, .01)])], GEOD, boundaries=dataset())
    assert mileage(result, "shared") == 0
    assert mileage(result, "state", "AA") > 0
    result = partition_pipelines([pipeline([(-1e-8, 0), (1e-8, 0)])], GEOD, boundaries=dataset())
    assert mileage(result, "state", "AA") > 0
    assert mileage(result, "state", "BB") > 0
    assert result["reconciliation"]["passed"]


@pytest.mark.parametrize("finish,expected", [((.001, .001), 1), ((-.001, .001), 0)])
def test_shared_run_counts_only_a_change_of_exclusive_state(finish, expected):
    coordinates = [(-.001, -.001), (0, -.001), (0, .001), finish]
    for path in (coordinates, list(reversed(coordinates))):
        result = partition_pipelines([pipeline(path)], GEOD, boundaries=dataset())
        assert mileage(result, "shared") > 0
        assert result["crossing_count"] == expected
        assert result["reconciliation"]["passed"]


def test_crossing_count_resets_across_coverage_exceptions():
    boundaries = BoundaryDataset({"AA": box(-1, -1, -.1, 1), "BB": box(.1, -1, 1, 1)})
    result = partition_pipelines([pipeline([(-.2, 0), (.2, 0)])], GEOD, boundaries=boundaries)
    assert mileage(result, "outside") > 0
    assert result["crossing_count"] == 0


def test_reentry_disconnected_paths_and_duplicate_source_identity():
    records = [pipeline(paths=[[(-.01, 0), (.01, 0), (-.01, .01)], [(1, 0), (1.01, 0)]]),
               pipeline([(-.01, -.01), (-.02, -.01)])]
    result = partition_pipelines(records, GEOD, boundaries=dataset())
    assert {f["source_id"] for f in result["fragments"]} == {0, 1}
    assert result["crossing_count"] == 2
    a_fragments = [f for f in result["fragments"] if f["source_id"] == 0 and f["state_codes"] == ["AA"]]
    assert len(a_fragments) == 2
    assert {f["path_index"] for f in result["fragments"] if f["source_id"] == 0} == {0, 1}
    assert result["reconciliation"]["passed"]


def test_touch_has_no_shared_mileage_and_disconnected_states_are_not_crossings():
    result = partition_pipelines([pipeline([(-.01, -.01), (0, 0), (-.01, .01)])], GEOD, boundaries=dataset())
    assert mileage(result, "state", "BB") == 0
    assert mileage(result, "shared") == 0
    result = partition_pipelines([pipeline(paths=[[(-1, 0), (-.9, 0)], [(1, 0), (1.1, 0)]])], GEOD, boundaries=dataset())
    assert result["crossing_count"] == 0


def test_long_sparse_geodesic_crosses_a_border_above_endpoint_latitude():
    boundaries = BoundaryDataset({"AA": box(-10, 40, 10, 50.01), "BB": box(-10, 50.01, 10, 60)})
    result = partition_pipelines([pipeline([(-5, 50), (5, 50)])], GEOD, boundaries=boundaries)
    assert mileage(result, "state", "BB") > 0
    assert result["crossing_count"] == 2
    assert result["reconciliation"]["passed"]
    for fragment in result["fragments"]:
        if fragment["state_codes"] == ["BB"]:
            assert fragment["coordinates"][0][1] == pytest.approx(50.01, abs=1e-8)
            assert fragment["coordinates"][-1][1] == pytest.approx(50.01, abs=1e-8)


def test_holes_and_islands_are_preserved():
    outer = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, -2)],
                    [[(-.1, -.1), (.1, -.1), (.1, .1), (-.1, .1), (-.1, -.1)]])
    boundaries = BoundaryDataset({"AA": MultiPolygon([outer, box(3, -1, 4, 1)])})
    result = partition_pipelines([pipeline([(-1, 0), (3.5, 0)])], GEOD, boundaries=boundaries)
    assert mileage(result, "outside") == pytest.approx(GEOD.inv(-.1, 0, .1, 0)[2]+GEOD.inv(2, 0, 3, 0)[2], abs=.001)
    assert result["reconciliation"]["passed"]


def test_dateline_does_not_create_world_spanning_geometry():
    boundaries = BoundaryDataset({"AK": Polygon([(179, 50), (-179, 50), (-179, 60), (179, 60), (179, 50)])})
    result = partition_pipelines([pipeline([(179.5, 55), (-179.5, 55)])], GEOD, boundaries=boundaries)
    assert mileage(result, "state", "AK") == pytest.approx(GEOD.inv(179.5, 55, -179.5, 55)[2], abs=.001)
    assert mileage(result, "outside") == 0
    assert result["reconciliation"]["passed"]


def test_positive_area_boundary_overlap_is_unresolved():
    boundaries = BoundaryDataset({"AA": box(-1, -1, .1, 1), "BB": box(-.1, -1, 1, 1)})
    result = partition_pipelines([pipeline([(-.05, 0), (.05, 0)])], GEOD, boundaries=boundaries)
    assert mileage(result, "unresolved") > 0
    assert mileage(result, "shared") == 0
    assert result["diagnostics"][0]["code"] == "ambiguous_state_coverage"


def test_cancellation_does_not_publish_partial_result():
    context = ExecutionContext()
    context.cancel()
    with pytest.raises(AnalysisCancelled):
        partition_pipelines([pipeline([(-1, 0), (1, 0)])], GEOD, boundaries=dataset(), context=context)


def test_corridor_clipping_preserves_holes_and_omits_invalid_fallback():
    state = Polygon([(-2, -2), (0, -2), (0, 2), (-2, 2), (-2, -2)],
                    [[(-1.5, -.2), (-1, -.2), (-1, .2), (-1.5, .2), (-1.5, -.2)]])
    boundaries = BoundaryDataset({"AA": state})
    section = {"corridor_polygon": [[-1.8, -1], [.5, -1], [.5, 1], [-1.8, 1], [-1.8, -1]]}
    clipped = clip_state_corridor(section, "AA", boundaries, GEOD)
    assert clipped["clipped_polygons"]
    assert "clipped_polygons" not in section
    assert len(clipped["clipped_polygons"][0]["holes"]) == 1
    for poly in clipped["clipped_polygons"]:
        assert state.covers(Polygon(poly["outer"], poly["holes"]))
    invalid = clip_state_corridor({"oriented_polygon": [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]}, "AA", boundaries, GEOD)
    assert invalid["clipped_polygons"] == []
    assert invalid["diagnostics"]


def test_bundled_data_is_offline_full_coverage_and_provenanced():
    boundaries = load_boundaries()
    assert len(boundaries.geometries) == 51
    assert boundaries.state_names["DC"] == "District of Columbia"
    assert boundaries.states_at(-157.8583, 21.3069) == ["HI"]
    assert boundaries.states_at(-149.9003, 61.2181) == ["AK"]
    assert boundaries.states_at(-97.7431, 30.2672) == ["TX"]
    assert boundaries.boundary_source["vintage"] == "2025"
    assert len(boundaries.boundary_source["source_sha256"]) == 64
    transforms = boundaries.boundary_source["transformation"]
    assert any(p["operation"] == "EPSG:1252" for p in transforms["components"]["HI"])
    assert any(p["operation"] == "EPSG:1251" for p in transforms["components"]["AK"])
    assert transforms["operations"]["BALLPARK"]["accuracy_meters"] is None
    assert boundaries.boundary_source["approximate_regions"]


def test_missing_boundary_resource_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_boundaries(tmp_path / "missing.zip")


def test_explicit_source_ids_and_contiguous_multichunk_merge():
    coordinates = [(-1.9, -1.4), (-1.5, -.8), (-1.1, -.2), (-.5, .4)]
    record = pipeline(coordinates)
    record["id"] = 42
    result = partition_pipelines([record], GEOD, boundaries=dataset())
    assert len(result["fragments"]) == 1
    fragment = result["fragments"][0]
    assert fragment["source_id"] == 42
    for point in coordinates:
        assert list(point) in fragment["coordinates"]
    assert result["reconciliation"]["passed"]


def test_boundary_free_vertices_do_not_construct_projections(monkeypatch):
    import pipeline_calculator.core.geography.partition as engine
    monkeypatch.setattr(engine, "_local_projection", lambda *args: pytest.fail("Unnecessary projection"))
    result = partition_pipelines([pipeline([(-1, .1), (-.9, .2), (-.8, .3)])], GEOD, boundaries=dataset())
    assert len(result["fragments"]) == 1


def test_work_budget_preserves_unprocessed_source_as_unresolved(monkeypatch):
    import pipeline_calculator.core.geography.partition as engine
    monkeypatch.setattr(engine, "MAX_WORK_ITEMS", 3)
    result = partition_pipelines([pipeline([(-1, 0), (-.9, 0), (-.8, 0), (-.7, 0), (-.6, 0)])], GEOD, boundaries=dataset())
    assert mileage(result, "unresolved") > 0
    assert result["reconciliation"]["passed"]
    assert any(d["code"] == "state_clipping_incomplete" for d in result["diagnostics"])


def test_area_aware_datum_operations_retain_all_vertices():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/data/prepare_state_boundaries.py"
    spec = importlib.util.spec_from_file_location("boundary_preparation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for code, point, operation in [("TX", (-97, 30), "EPSG:1188"),
                                    ("HI", (-157, 21), "EPSG:1252"),
                                    ("AK", (-170, 52), "EPSG:1251"),
                                    ("HI", (-178, 28), "BALLPARK")]:
        lon, lat = point
        source = box(lon, lat, lon+.01, lat+.01)
        result, lineage = module.transform_components(source, code)
        assert lineage[0]["operation"] == operation
        assert len(result.exterior.coords) == len(source.exterior.coords)
        if operation in ("EPSG:1252", "EPSG:1251"):
            assert not result.equals_exact(source, 1e-8)
