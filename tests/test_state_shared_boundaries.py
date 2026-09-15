"""Shared ownership must survive numerical representation and path changes."""
import itertools
import math

import pytest
from pyproj import Geod
from shapely.geometry import Polygon, box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.geography import BoundaryDataset, load_boundaries, partition_pipelines
from pipeline_calculator.core.geography import partition as engine
from pipeline_calculator.core.state_analysis import build_state_breakdown


GEOD = Geod(ellps="GRS80")


def adjacent_states(longitude, latitude, reverse=False):
    items = [("AA", box(longitude-1, latitude-1, longitude, latitude+1)),
             ("BB", box(longitude, latitude-1, longitude+1, latitude+1))]
    return BoundaryDataset(dict(reversed(items) if reverse else items))


def source(source_id, coordinates):
    return {"id": source_id, "name": "Repeated source name", "objectid": "duplicate",
            "placemark_id": "duplicate", "coordinates": coordinates,
            "coordinate_paths": [coordinates]}


def assert_shared_only(result, expected_by_source, codes):
    assert result["diagnostics"] == []
    assert result["reconciliation"]["passed"]
    assert result["crossing_count"] == 0
    assert len(result["fragments"]) == len(expected_by_source)
    for fragment in result["fragments"]:
        assert fragment["kind"] == "shared"
        assert fragment["state_codes"] == sorted(codes)
        expected = expected_by_source[fragment["source_id"]]
        assert fragment["length_meters"] == pytest.approx(expected, abs=1e-8)
        assert fragment["start_m"] == 0
        assert fragment["end_m"] == pytest.approx(expected, abs=1e-8)
        assert all(GEOD.inv(*a, *b)[2] > 0
                   for a, b in zip(fragment["coordinates"], fragment["coordinates"][1:]))


@pytest.mark.parametrize("longitude,latitude,half_span", [
    (-118.21696820486638, 40.5975722857745, .0016409363084426332),
    (-103.0, 30.0, 1e-8),  # A 2.2 mm shared source, including the old reverse failure.
    (-118.21696820486638, 40.5975722857745, .6),  # Multiple 20 km chunks.
])
def test_shared_ownership_preserves_direction_vertices_and_input_order(longitude, latitude, half_span):
    for reverse_path, extra_vertex, reverse_order in itertools.product((False, True), repeat=3):
        boundaries = adjacent_states(longitude, latitude, reverse_order)
        # Use the chosen canonical boundary model, not a separately rounded meridian.
        lon = boundaries.geometries["AA"].bounds[2]
        coordinates = [(lon, latitude-half_span), (lon, latitude+half_span)]
        if extra_vertex:
            coordinates.insert(1, (lon, latitude))
        if reverse_path:
            coordinates.reverse()
        records = [source(42, coordinates), source(7, coordinates)]
        if reverse_order:
            records.reverse()
        expected = math.fsum(GEOD.inv(*a, *b)[2] for a, b in zip(coordinates, coordinates[1:]))
        result = partition_pipelines(records, GEOD, boundaries=boundaries)
        assert_shared_only(result, {42: expected, 7: expected}, ["AA", "BB"])
        for fragment in result["fragments"]:
            assert all(list(point) in fragment["coordinates"] for point in coordinates)


@pytest.mark.parametrize("reverse,extra_vertex", list(itertools.product((False, True), repeat=2)))
def test_real_texas_new_mexico_shared_endpoint_has_no_unresolved_tail(reverse, extra_vertex, monkeypatch):
    coordinates = [(-103.064732, 32.744215), (-103.064732, 32.75427)]
    if extra_vertex:
        coordinates.insert(1, (-103.064732, (32.744215+32.75427)/2))
    if reverse:
        coordinates.reverse()
    # Exercise canonical source endpoints and generated chunk endpoints together.
    monkeypatch.setattr(engine, "MAX_CHUNK_METERS", 400)
    boundaries = load_boundaries()
    expected = math.fsum(GEOD.inv(*a, *b)[2] for a, b in zip(coordinates, coordinates[1:]))
    result = partition_pipelines([source(42, coordinates)], GEOD, boundaries=boundaries)
    assert_shared_only(result, {42: expected}, ["NM", "TX"])


@pytest.mark.parametrize("real_boundaries", [False, True])
def test_shared_reproductions_publish_complete_equal_undiscounted_allocations(real_boundaries):
    if real_boundaries:
        boundaries = load_boundaries()
        coordinates = [(-103.064732, 32.744215), (-103.064732, 32.75427)]
    else:
        boundaries = adjacent_states(-118.21696820486638, 40.5975722857745)
        lon = boundaries.geometries["AA"].bounds[2]
        coordinates = [(lon, 40.595931349466056), (lon, 40.59921322208294)]
    for path in (coordinates, coordinates[::-1]):
        analyzer = PipelineAnalyzer()
        records = [source(42, path)]
        combined = analyzer.analyze_features(records)
        geography = build_state_breakdown(analyzer, records, combined, boundaries=boundaries)
        assert geography["status"] == "complete", geography["diagnostics"]
        assert geography["reconciliation"]["unresolved_meters"] == 0
        assert geography["reconciliation"]["passed"]
        assert len(geography["states"]) == 2
        shared_id = geography["fragments"][0]["id"]
        for state in geography["states"]:
            assert state["analysis_complete"]
            assert state["interior_meters"] == 0
            assert state["shared_allocation_meters"] == pytest.approx(combined["total_meters"]/2, abs=1e-8)
            assert state["adjusted_total_meters"] == state["total_meters"]
            assert state["shared_overlap_status"] == "not_calculated"
            assert state["pipelines"][0]["shared_border_allocations"][0]["fragment_id"] == shared_id


@pytest.mark.parametrize("offset", [1e-8, 1e-10])
def test_nearby_parallel_lines_are_not_shared_and_short_true_crossings_remain(offset):
    boundaries = adjacent_states(-118.21696820486638, 40.5975722857745)
    lon = boundaries.geometries["AA"].bounds[2]
    lat = 40.5975722857745
    for sign, code in ((-1, "AA"), (1, "BB")):
        coordinates = [(lon+sign*offset, lat-.001), (lon+sign*offset, lat+.001)]
        result = partition_pipelines([source(42, coordinates)], GEOD, boundaries=boundaries)
        assert all(f["kind"] == "state" and f["state_codes"] == [code] for f in result["fragments"])
        assert result["diagnostics"] == []
    for points in ([(lon-offset, lat), (lon+offset, lat)],
                   [(lon+offset, lat), (lon-offset, lat)]):
        result = partition_pipelines([source(42, points)], GEOD, boundaries=boundaries)
        assert {tuple(f["state_codes"]) for f in result["fragments"]} == {("AA",), ("BB",)}
        assert all(f["kind"] == "state" and f["length_meters"] > 0 for f in result["fragments"])
        assert result["reconciliation"]["passed"]


def test_shared_endpoint_does_not_absorb_a_distinct_short_outside_interval():
    boundaries = BoundaryDataset({"AA": box(-1, -1, 0, 0), "BB": box(0, -1, 1, 0)})
    coordinates = [(0, -.001), (0, 1e-8)]
    for points in (coordinates, coordinates[::-1]):
        result = partition_pipelines([source(42, points)], GEOD, boundaries=boundaries)
        outside = math.fsum(f["length_meters"] for f in result["fragments"] if f["kind"] == "outside")
        assert outside == pytest.approx(GEOD.inv(0, 0, 0, 1e-8)[2], rel=1e-6)
        assert outside > 0
        assert result["diagnostics"] == []
        assert result["reconciliation"]["passed"]


def test_different_boundary_vertex_lists_share_one_contiguous_source_interval():
    # The two states encode the same boundary with different intermediate vertices.
    boundaries = BoundaryDataset({
        "AA": Polygon([(-1, -1), (0, -1), (0, -.000123456789), (0, 1), (-1, 1), (-1, -1)]),
        "BB": Polygon([(0, -1), (1, -1), (1, 1), (0, 1), (0, .000123456789), (0, -1)]),
    })
    for coordinates in ([(0, -.001), (0, .001)], [(0, .001), (0, -.001)]):
        result = partition_pipelines([source(42, coordinates)], GEOD, boundaries=boundaries)
        assert_shared_only(result, {42: GEOD.inv(*coordinates[0], *coordinates[1])[2]}, ["AA", "BB"])


def test_shared_equatorial_path_is_continuous_across_the_antimeridian():
    boundaries = BoundaryDataset({
        "AA": Polygon([(179, -1), (-179, -1), (-179, 0), (179, 0), (179, -1)]),
        "BB": Polygon([(179, 0), (-179, 0), (-179, 1), (179, 1), (179, 0)]),
    })
    for coordinates in ([(179.9, 0), (-179.9, 0)], [(-179.9, 0), (179.9, 0)]):
        result = partition_pipelines([source(42, coordinates)], GEOD, boundaries=boundaries)
        assert result["diagnostics"] == []
        assert all(f["kind"] == "shared" and f["state_codes"] == ["AA", "BB"] for f in result["fragments"])
        assert math.fsum(f["length_meters"] for f in result["fragments"]) == pytest.approx(
            GEOD.inv(*coordinates[0], *coordinates[1])[2], abs=1e-8)
        assert result["reconciliation"]["passed"]


@pytest.mark.parametrize("origin", [(-118.21696820486636, 40.595931349466056),
                                     (-157.8583123456789, 21.3069123456789),
                                     (-169.8765432109876, 53.1234567890123),
                                     (179.9999, 52.1234567890123),
                                     (-179.9999, 52.1234567890123),
                                     (-153.1234567890123, 72.1234567890123)])
def test_actual_local_projection_preserves_radial_geodesics_across_supported_regions(origin):
    transformer = engine._local_projection(*origin, GEOD)
    for bearing in (0, 37.25, 90, 180, 270):
        angle = math.radians(bearing)
        for distance in (0, .002, 10_000, 20_000):
            point = origin if distance == 0 else GEOD.fwd(*origin, bearing, distance)[:2]
            actual = transformer.transform(*point)
            expected = (distance*math.sin(angle), distance*math.cos(angle))
            assert math.dist(actual, expected) < 1e-6


def test_unreliable_projection_has_a_numerical_diagnostic_and_preserves_source_mileage(monkeypatch):
    class InvalidProjection:
        def transform(self, lon, lat):
            return 100.0, 100.0
    monkeypatch.setattr(engine, "_local_projection", lambda *args: InvalidProjection())
    boundaries = adjacent_states(0, 0)
    result = partition_pipelines([source(42, [(0, -.001), (0, .001)])], GEOD, boundaries=boundaries)
    assert result["diagnostics"][0]["code"] == "state_clipping_numerical_uncertainty"
    assert all(f["kind"] == "unresolved" for f in result["fragments"])
    assert result["reconciliation"]["passed"]
