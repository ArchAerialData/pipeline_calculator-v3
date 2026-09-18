"""Independent adversarial checks for the geographic oracle, not app goldens.

Run directly with Python or via unittest/pytest. Analytic rectangles isolate the
root solver; real-resource controls verify that the same decisions hold against
unchanged native Census coordinates. PROJ supplies independent distance/root
evaluations. No application calculation code is imported.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import pytest

for dependency in ('geographiclib', 'matplotlib', 'psutil'):
    pytest.importorskip(dependency, reason='Install the KMZ suite requirements to run its optional audit tests')

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))

from pipeline_kmz_regression_suite.reference.geometry import (  # noqa: E402
    GeometryReference, SECOND_GEODESIC, _SourceEdge, read_kmz,
)


def analytic_reference(polygons):
    """Use declared planar polygons, so expected topology is hand-checkable."""
    reference = GeometryReference.__new__(GeometryReference)
    reference.polygons = list(polygons.values())
    reference.polygon_states = list(polygons)
    reference.state_names = {code: code for code in polygons}
    reference.tree = STRtree(reference.polygons)
    reference.edge_trees = {}
    return reference


def rectangle(west, east, south=-1, north=1):
    return Polygon([(west, south), (east, south), (east, north), (west, north)])


def source(key, *paths):
    return {"key": key, "name": key, "xml_order": 0, "paths": list(paths)}


def projected_point(coordinate, bearing, meters):
    lon, lat, _ = SECOND_GEODESIC.fwd(*coordinate, bearing, meters)
    return lon, lat


class GeometryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = GeometryReference(SUITE.parents[3] / "src/pipeline_calculator/data/states_2025.zip")

    def assert_conservation(self, result):
        accounted = math.fsum([v["attributed_original_meters"] for v in result["states"].values()]
                              + [result["outside_meters"], result["unresolved_meters"]])
        self.assertAlmostEqual(accounted, result["original_meters"], delta=1e-7)
        for item in result["sources"]:
            rows = [r for r in result["intervals"] if r["source_key"] == item["key"]]
            self.assertAlmostEqual(math.fsum(r["length_meters"] for r in rows), item["original_meters"], delta=1e-7)
            for path in item["paths"]:
                parts = [r for r in rows if r["path_index"] == path["path_index"]]
                self.assertEqual(parts[0]["start_m"], 0)
                self.assertAlmostEqual(parts[-1]["end_m"], path["original_meters"], delta=1e-7)
                for left, right in zip(parts, parts[1:]):
                    self.assertEqual(left["end_m"], right["start_m"])

    def test_native_longitude_is_not_perturbed_by_wrapping(self):
        reference = analytic_reference({"AA": rectangle(10.1, 11)})
        coordinate = (10.1, 0)
        self.assertTrue(reference.polygons[0].covers(Point(coordinate)))
        self.assertNotEqual((coordinate[0] + 180) % 360 - 180, coordinate[0])
        self.assertEqual(reference.states_at(coordinate), ["AA"])

    def test_sparse_geodesic_visits_state_without_endpoint_chord_intersection(self):
        latitude = 40.00002
        reference = analytic_reference({"AA": rectangle(-1, 1, latitude, 41)})
        start, end = (-.1, 40), (.1, 40)
        self.assertLess(start[1], latitude)
        self.assertLess(end[1], latitude)
        result = reference.analyze([source("sparse", [start, end])])
        self.assertEqual([r["kind"] for r in result["intervals"]], ["outside", "interior", "outside"])
        azimuth, _, total = SECOND_GEODESIC.inv(*start, *end)
        low, high = 0, total / 2
        for _ in range(60):
            middle = (low + high) / 2
            if SECOND_GEODESIC.fwd(*start, azimuth, middle)[1] < latitude:
                low = middle
            else:
                high = middle
        first_cut = result["intervals"][0]
        self.assertAlmostEqual(first_cut["end_m"], (low + high) / 2,
                               delta=first_cut["end_error_bound_meters"])
        self.assertAlmostEqual(result["states"]["AA"]["interior_meters"], total - low - high,
                               delta=result["states"]["AA"]["length_error_bound_meters"])
        self.assert_conservation(result)

    def test_shallow_crossing_cannot_claim_a_fixed_micrometer_certificate(self):
        edge = _SourceEdge((-.1, 40), (.1, 40))
        # The geodesic nearly touches this latitude. Double-precision latitude
        # rounding is amplified into station error; the fixed 2 um certificate
        # previously understated a PROJ crosscheck difference by over 25x.
        latitude = edge.point(edge.length / 2)[1] - 1e-8
        with self.assertRaisesRegex(ValueError, "Shallow root"):
            GeometryReference._roots(edge, (-1, latitude), (1, latitude))

    def test_distinct_unresolvable_roots_fail_instead_of_erasing_a_visit(self):
        reference = analytic_reference({"AA": rectangle(-1, 0), "BB": rectangle(0, 1e-12),
                                        "CC": rectangle(1e-12, 1)})
        with self.assertRaisesRegex(ValueError, "intervening interval must not be discarded"):
            reference.analyze([source("submicrometer_visit", [(-.001, 0), (.001, 0)])])

    def test_real_ten_centimeter_crossing_is_retained(self):
        center = (-103.064732, 32.748)
        west, east = projected_point(center, 270, .05), projected_point(center, 90, .05)
        result = self.native.analyze([source("tiny", [west, east])])
        self.assertEqual(result["crossing_count"], 1)
        self.assertEqual([r["state_codes"] for r in result["intervals"]], [["NM"], ["TX"]])
        self.assertEqual(result["shared_meters"], 0)
        self.assertEqual(result["unresolved_meters"], 0)
        for row in result["intervals"]:
            self.assertAlmostEqual(row["length_meters"], .05, delta=.00001)
        self.assert_conservation(result)

    def test_exact_native_shared_interval_and_centimeter_offsets(self):
        start, end = (-103.064732, 32.746), (-103.064732, 32.749)
        result = self.native.analyze([
            source("shared", [start, end]),
            source("east", [projected_point(start, 90, .03), projected_point(end, 90, .03)]),
            source("west", [projected_point(start, 270, .03), projected_point(end, 270, .03)]),
        ])
        rows = {r["source_key"]: r for r in result["intervals"]}
        self.assertEqual((rows["shared"]["kind"], rows["shared"]["state_codes"]), ("shared", ["NM", "TX"]))
        self.assertEqual((rows["east"]["kind"], rows["east"]["state_codes"]), ("interior", ["TX"]))
        self.assertEqual((rows["west"]["kind"], rows["west"]["state_codes"]), ("interior", ["NM"]))
        independent_length = SECOND_GEODESIC.inv(*start, *end)[2]
        self.assertAlmostEqual(result["shared_meters"], independent_length, delta=1e-7)
        self.assertEqual(len(result["shared_allocations"]), 2)
        for allocation in result["shared_allocations"]:
            self.assertEqual(allocation["interval_id"], rows["shared"]["id"])
            self.assertAlmostEqual(allocation["allocated_meters"], independent_length / 2, delta=1e-7)
        self.assert_conservation(result)

    def test_native_endpoint_touch_has_no_positive_foreign_length(self):
        endpoint = (-103.064732, 32.748)
        east = projected_point(endpoint, 90, 4)
        for path in ([east, endpoint], [endpoint, east]):
            with self.subTest(path=path):
                result = self.native.analyze([source("touch", path)])
                self.assertEqual(result["represented_states"], ["TX"])
                self.assertEqual(result["crossing_count"], 0)
                self.assertEqual(result["unresolved_meters"], 0)
                self.assertEqual(len(result["endpoint_touches"]), 1)
                self.assertEqual(result["endpoint_touches"][0]["touched_states"], ["NM"])
                self.assertEqual(result["endpoint_touches"][0]["positive_length_in_touched_states_meters"], 0)

    def test_collinear_endpoint_touch_is_not_positive_coincidence(self):
        edge = _SourceEdge((10.1, 0), (10.1, .5))
        roots = GeometryReference._roots(edge, (10.1, -.5), (10.1, 0))
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["station"], 0)
        self.assertEqual(GeometryReference._roots(edge, (10.1, -.5), (10.1, -.1)), [])

    def test_latitude_endpoint_chord_is_never_assumed_shared(self):
        edge = _SourceEdge((-.1, 40), (.1, 40))
        self.assertGreater(edge.point(edge.length / 2)[1], 40)
        self.assertIsNone(GeometryReference._shared(edge, (-.2, 40), (.2, 40)))
        with self.assertRaisesRegex(ValueError, "Unproved positive coincidence"):
            GeometryReference._roots(edge, (-.2, 40), (.2, 40))

    def test_dateline_root_uses_unwrapped_geodesic_and_periodic_ownership(self):
        reference = analytic_reference({"AA": rectangle(179.8, 180), "BB": rectangle(180, 180.2)})
        result = reference.analyze([source("dateline", [(179.9, 0), (-179.9, 0)])])
        self.assertEqual(result["crossing_count"], 1)
        self.assertEqual([r["state_codes"] for r in result["intervals"]], [["AA"], ["BB"]])
        self.assertEqual(result["outside_meters"], 0)
        self.assertEqual(reference.states_at((-179.95, 0)), ["BB"])
        self.assertAlmostEqual(result["states"]["AA"]["interior_meters"], result["original_meters"] / 2, delta=.00001)
        self.assert_conservation(result)

    def test_hole_and_disconnected_paths_preserve_gaps(self):
        polygon = Polygon([(-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01)],
                          [[(-.001, -.001), (.001, -.001), (.001, .001), (-.001, .001)]])
        reference = analytic_reference({"AA": polygon})
        result = reference.analyze([source("hole", [(-.005, 0), (.005, 0)],
                                          [(-.006, .005), (-.004, .005)])])
        self.assertEqual([r["kind"] for r in result["intervals"]], ["interior", "outside", "interior", "interior"])
        fragments = result["state_inputs"]["AA"][0]["paths"]
        self.assertEqual(len(fragments), 3)
        self.assertLess(fragments[0][-1][0], fragments[1][0][0])
        self.assert_conservation(result)

    def test_reversal_and_geodesic_vertex_preserve_ownership_and_original(self):
        reference = analytic_reference({"AA": rectangle(-1, 0), "BB": rectangle(0, 1)})
        start, end = (-.002, .01), (.003, .01)
        azimuth, _, length = SECOND_GEODESIC.inv(*start, *end)
        middle = projected_point(start, azimuth, length * .3)
        variants = [[start, end], [end, start], [start, middle, end]]
        results = [reference.analyze([source("variant", path)]) for path in variants]
        for result in results:
            self.assertEqual(result["crossing_count"], 1)
            self.assertAlmostEqual(result["original_meters"], length, delta=1e-7)
            for code in ("AA", "BB"):
                self.assertAlmostEqual(result["states"][code]["interior_meters"],
                                       results[0]["states"][code]["interior_meters"], delta=.00001)
            self.assert_conservation(result)

    def test_domain_limits_fail_before_certification(self):
        for start, end in [((0, 0), (2, 0)), ((0, 84.99), (0, 85))]:
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "certification domain"):
                _SourceEdge(start, end)

    def test_altitude_is_ignored_when_reading_final_kmz(self):
        xml = '''<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
        <Placemark id="fixture"><name>fixture</name><ExtendedData>
        <Data name="fixture_key"><value>altitude</value></Data></ExtendedData>
        <LineString><coordinates>10.2,0,1000000000 10.3,0,-1000000000</coordinates></LineString>
        </Placemark></Document></kml>'''
        with tempfile.TemporaryDirectory(prefix="kmz-geometry-audit-") as folder:
            path = Path(folder) / "altitude.kmz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("doc.kml", xml)
            sources = read_kmz(path)
        self.assertEqual(sources[0]["paths"], [[(10.2, 0), (10.3, 0)]])
        result = analytic_reference({"AA": rectangle(10, 11)}).analyze(sources)
        self.assertAlmostEqual(result["original_meters"], SECOND_GEODESIC.inv(10.2, 0, 10.3, 0)[2], delta=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
