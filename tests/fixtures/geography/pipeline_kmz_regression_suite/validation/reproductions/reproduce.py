"""Reproduce independently diagnosed clipping behavior, without changing goldens.

Run from the repository: python tests/fixtures/geography/pipeline_kmz_regression_suite/validation/reproductions/reproduce.py
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[1]
REPO = SUITE.parents[3]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.reference.geometry import GeometryReference, read_kmz

KML = "http://www.opengis.net/kml/2.2"
NS = {"k": KML}
ET.register_namespace("", KML)


def extract_minimum(reverse=False, case="04"):
    """Keep exactly two serialized coordinate tokens from one source edge."""
    fixture = "04_shared_border_and_near_border" if case == "04" else "02_three_state_transverse_crossings"
    source_key = "04_shared_qualification_route" if case == "04" else "02_endpoint_touch"
    original = SUITE / "fixtures" / f"{fixture}.kmz"
    with zipfile.ZipFile(original) as archive:
        root = ET.fromstring(archive.read("doc.kml"))
    feature = next(p for p in root.findall(".//k:Placemark", NS)
                   if p.findtext("k:ExtendedData/k:Data[@name='fixture_key']/k:value", namespaces=NS)
                   == source_key)
    feature = deepcopy(feature)
    node = feature.find(".//k:LineString/k:coordinates", NS)
    points = node.text.split()[1:3] if case == "04" else node.text.split()
    if reverse:
        points.reverse()
    node.text = " ".join(points)
    kml = ET.Element(f"{{{KML}}}kml")
    document = ET.SubElement(kml, f"{{{KML}}}Document")
    document.append(feature)
    stem = "04_endpoint_roundoff" if case == "04" else "02_native_vertex_shift"
    path = HERE / (stem + ("_reversed" if reverse else "") + ".kmz")
    info = zipfile.ZipInfo("doc.kml", date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(info, ET.tostring(kml, encoding="utf-8", xml_declaration=True))
    return path


def rejected_construction(reference):
    """This rejected interpolation is retained as numbers, not a false golden."""
    start = (-103.04576126111623, 36.47999990782171)
    endpoint = (-103.04170777115063, 36.4799999766659)
    a, b = (-103.041729, 36.48707), (-103.041703, 36.478411)
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float
        determinant = ((d(b[0]) - d(a[0])) * (d(endpoint[1]) - d(a[1]))
                       - (d(b[1]) - d(a[1])) * (d(endpoint[0]) - d(a[0])))
    rejected = False
    try:
        reference.analyze([{"key": "rejected_interpolated_touch", "name": "Rejected construction",
                            "xml_order": 0, "paths": [[start, endpoint]]}])
    except ValueError as error:
        rejected = True
        failure = str(error)
    assert rejected, "Uncertifiable endpoint sliver must fail closed"
    return {"source_key": "02_endpoint_touch", "rejected_coordinate": endpoint,
            "canonical_edge": [a, b], "exact_binary_determinant_degrees_squared": str(determinant),
            "point_memberships": reference.states_at(endpoint),
            "correction": "Use exact native boundary vertex (-103.041703,36.478411), then regenerate from final saved XML",
            "reference_rejects_uncertifiable_endpoint": rejected, "rejection": failure,
            "initial_reference_defect": "Overlapping root bracket incorrectly consumed the canonical source endpoint and classified a truncated path as an endpoint touch",
            "reference_correction": "Preserve canonical source endpoints, use 80-digit exact-binary endpoint predicates, and reject a root bracket overlapping an unproved endpoint"}


def main():
    reference = GeometryReference(REPO / "src/pipeline_calculator/data/states_2025.zip")
    # Establish each reference before importing/running application algorithms.
    fixed = [(p, reference.analyze(read_kmz(p))) for p in (
        extract_minimum(), extract_minimum(True), extract_minimum(case="02"), extract_minimum(True, case="02"))]
    construction = rejected_construction(reference)
    sys.path.insert(0, str(REPO / "src"))
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core.geography.boundaries import load_boundaries
    from pipeline_calculator.core.options import AnalysisOptions
    from shapely import from_wkb
    from shapely.geometry import Point
    native_endpoint = (-103.041703, 36.478411)
    shifted_vertex_evidence = []
    with zipfile.ZipFile(REPO / "src/pipeline_calculator/data/states_2025.zip") as archive:
        for code in ("NM", "TX"):
            native = from_wkb(archive.read(f"states/{code}.wkb"))
            actual_geometry = load_boundaries().geometries[code]
            def nearby_vertices(geometry):
                parts = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
                return [list(coordinate) for part in parts for coordinate in part.exterior.coords
                        if abs(coordinate[0] - native_endpoint[0]) < 1e-6
                        and abs(coordinate[1] - native_endpoint[1]) < 1e-6]
            shifted_vertex_evidence.append({"state_code": code,
                "native_vertices": nearby_vertices(native), "application_vertices": nearby_vertices(actual_geometry),
                "native_covers_original_endpoint": native.covers(Point(native_endpoint)),
                "application_covers_original_endpoint": actual_geometry.covers(Point(native_endpoint)),
                "symmetric_difference_area_degrees_squared": native.symmetric_difference(actual_geometry).area})
    rows = []
    for path, expected in fixed:
        actual = PipelineAnalyzer().analyze_complete(path, options=AnalysisOptions(True))
        geo = actual["geography"]
        rows.append({"archive": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "source_key": expected["sources"][0]["key"], "original_path_index": 0,
                     "original_vertex_indices": [1, 2] if path.name.startswith("04") else [0, 1],
                     "source_count": 1, "path_count": 1, "vertex_count": 2,
                     "expected": {"status": "complete", "original_meters": expected["original_meters"],
                                  "unresolved_meters": 0.0, "crossing_count": expected["crossing_count"],
                                  "intervals": expected["intervals"], "endpoint_touches": expected["endpoint_touches"]},
                     "observed": {"status": geo["status"], "original_meters": actual["total_meters"],
                                  "unresolved_meters": geo["reconciliation"]["unresolved_meters"],
                                  "crossing_count": geo["crossing_count"], "fragments": geo["fragments"],
                                  "diagnostics": geo["diagnostics"]}})
    report = {"schema_version": "1.0.0", "expectations_established_before_application_run": True,
              "baseline_commit": "71da499d5756648ae395660f0a241ea00edbea4f",
              "boundary_sha256": reference.provenance["resource_sha256"],
              "application_defect": "A boundary crossing root immediately before an already exact source endpoint creates a spurious unresolved positive-length interval and incomplete geography",
              "native_vertex_shift_defect": {"native_endpoint": native_endpoint,
                  "cause": "Application _unwrap_ring applies degrees(unwrap(radians(longitude))) to already canonical continental coordinates, moving this native vertex one ULP west",
                  "boundary_evidence": shifted_vertex_evidence},
              "rejected_fixture_construction": construction, "reproductions": rows}
    (HERE / "findings.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in rows:
        print(row["archive"], row["observed"]["status"], row["observed"]["unresolved_meters"])


if __name__ == "__main__":
    main()
