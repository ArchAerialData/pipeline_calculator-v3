"""Canonical boundary/source endpoints must not manufacture tiny state visits."""
import itertools
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest
from pyproj import Geod
from shapely import from_wkb
from shapely.geometry import Point, Polygon, box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.geography import BoundaryDataset, load_boundaries, partition_pipelines
from pipeline_calculator.core.geography.boundaries import _unwrap_ring, canonical_geometry
from pipeline_calculator.core.geography.partition import _point_on_boundary_edge
from pipeline_calculator.core.state_analysis import build_state_breakdown


GEOD = Geod(ellps="GRS80")
RESOURCE = Path(__file__).resolve().parents[1] / "src/pipeline_calculator/data/states_2025.zip"
# Exact serialized coordinates from the independently diagnosed two-edge cases.
NATIVE_VERTEX_APPROACH = [(-103.04575640720749, 36.47841093115979), (-103.041703, 36.478411)]
MERIDIAN_APPROACH = [(-103.06468931797826, 32.74962528131012), (-103.064732, 32.7496252813174)]


def source(path):
    return {"id": 42, "name": "Endpoint regression", "coordinate_paths": [path]}


def path_variant(path, reverse, extra_vertex):
    path = list(path)
    if extra_vertex:
        azimuth, _, distance = GEOD.inv(*path[0], *path[1])
        path.insert(1, GEOD.fwd(*path[0], azimuth, distance/2)[:2])
    return path[::-1] if reverse else path


@pytest.mark.parametrize("code", ["NM", "TX"])
def test_boundary_normalization_preserves_native_geometry_and_vertex(code):
    with zipfile.ZipFile(RESOURCE) as archive:
        native = from_wkb(archive.read(f"states/{code}.wkb"))
    normalized = load_boundaries().geometries[code]
    endpoint = Point(NATIVE_VERTEX_APPROACH[-1])
    assert native.covers(endpoint)
    assert normalized.covers(endpoint)
    assert native.symmetric_difference(normalized).is_empty


def test_degree_unwrap_retains_canonical_coordinates_and_moves_only_whole_turns():
    native = Polygon([(-103.041703, 36.478411), (-102.99999999999999, 36.48),
                      (-103.01, 36.6), (-103.041703, 36.478411)])
    assert np.array_equal(_unwrap_ring(native.exterior), np.asarray(native.exterior.coords))
    dateline = Polygon([(179.125, 51.25), (-179.75, 51.25), (-179.75, 52.5),
                        (179.125, 52.5), (179.125, 51.25)],
                       [[(179.75, 51.5), (-179.875, 51.5), (-179.875, 52.0),
                         (179.75, 52.0), (179.75, 51.5)]])
    unwrapped = _unwrap_ring(dateline.exterior)
    delta = unwrapped[:, 0] - np.asarray(dateline.exterior.coords)[:, 0]
    assert set(delta) == {0, 360}
    normalized = canonical_geometry(dateline)
    assert normalized.is_valid
    assert normalized.covers(Point(179.5, 51.75))
    assert not normalized.covers(Point(179.875, 51.75))
    assert not normalized.covers(Point(-179.9375, 51.75))


@pytest.mark.parametrize("reverse,extra_vertex", list(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("coordinates,code", [(NATIVE_VERTEX_APPROACH, "NM"), (MERIDIAN_APPROACH, "TX")])
def test_proven_boundary_endpoint_keeps_one_complete_exclusive_source(coordinates, code, reverse, extra_vertex):
    path = path_variant(coordinates, reverse, extra_vertex)
    analyzer = PipelineAnalyzer()
    records = [source(path)]
    combined = analyzer.analyze_features(records)
    geography = build_state_breakdown(analyzer, records, combined)
    assert geography["status"] == "complete", geography["diagnostics"]
    assert geography["crossing_count"] == 0
    assert geography["reconciliation"]["unresolved_meters"] == 0
    assert geography["reconciliation"]["passed"]
    assert len(geography["states"]) == 1
    assert geography["states"][0]["state_code"] == code
    assert len(geography["fragments"]) == 1
    fragment = geography["fragments"][0]
    assert fragment["kind"] == "state"
    assert fragment["state_codes"] == [code]
    assert fragment["start_m"] == 0
    assert fragment["end_m"] == combined["total_meters"]
    assert fragment["coordinates"][0] == list(path[0])
    assert fragment["coordinates"][-1] == list(path[-1])


@pytest.mark.parametrize("reverse", [False, True])
def test_boundary_vertex_touch_and_crossing_remain_different(reverse):
    west, endpoint = NATIVE_VERTEX_APPROACH
    for end, count, states in ((west, 0, {"NM"}),
                               (GEOD.fwd(*endpoint, 90, 4)[:2], 1, {"NM", "TX"})):
        path = [west, endpoint, end]
        if reverse:
            path.reverse()
        result = partition_pipelines([source(path)], GEOD)
        assert result["diagnostics"] == []
        assert result["crossing_count"] == count
        assert {f["state_codes"][0] for f in result["fragments"]} == states
        assert all(f["kind"] == "state" and f["length_meters"] > 0 for f in result["fragments"])
        assert result["reconciliation"]["passed"]


def test_exact_binary_endpoint_predicate_does_not_promote_a_nearby_point():
    a, b = (-1.0, -1.0), (1.0, 1.0)
    assert _point_on_boundary_edge((.125, .125), a, b)
    assert not _point_on_boundary_edge((math.nextafter(.125, 1), .125), a, b)
    # The fixture's rejected interpolated touch is inside Texas by an exact
    # nonzero determinant. Being extremely close does not establish identity.
    assert not _point_on_boundary_edge((-103.04170777115063, 36.4799999766659),
                                       (-103.041729, 36.48707), (-103.041703, 36.478411))


@pytest.mark.parametrize("reverse", [False, True])
def test_non_axis_aligned_exact_endpoint_uses_canonical_station(reverse):
    boundaries = BoundaryDataset({
        "AA": Polygon([(-1, -1), (1, 1), (-1, 1), (-1, -1)]),
        "BB": Polygon([(-1, -1), (1, -1), (1, 1), (-1, -1)]),
    })
    endpoint = (.125, .125)
    path = [GEOD.fwd(*endpoint, 315, 4)[:2], endpoint]
    if reverse:
        path.reverse()
    result = partition_pipelines([source(path)], GEOD, boundaries=boundaries)
    assert result["diagnostics"] == []
    assert result["crossing_count"] == 0
    assert len(result["fragments"]) == 1
    assert result["fragments"][0]["state_codes"] == ["AA"]
    assert result["fragments"][0]["kind"] == "state"
    assert result["reconciliation"]["passed"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("short_length", [0.1, 1e-5])
def test_genuine_short_crossing_near_an_endpoint_is_never_absorbed(reverse, short_length):
    lon, lat = MERIDIAN_APPROACH[-1]
    boundaries = BoundaryDataset({"AA": box(lon-1, lat-1, lon, lat+1),
                                  "BB": box(lon, lat-1, lon+1, lat+1)})
    path = [GEOD.fwd(lon, lat, 270, 4)[:2], GEOD.fwd(lon, lat, 90, short_length)[:2]]
    if reverse:
        path.reverse()
    result = partition_pipelines([source(path)], GEOD, boundaries=boundaries)
    assert result["diagnostics"] == []
    assert result["crossing_count"] == 1
    assert result["reconciliation"]["passed"]
    assert len(result["fragments"]) == 2
    short = next(f for f in result["fragments"] if f["state_codes"] == ["BB"])
    assert short["kind"] == "state"
    assert short["length_meters"] == pytest.approx(short_length, abs=1e-8)
    assert short["length_meters"] > 0
