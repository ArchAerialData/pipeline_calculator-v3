from __future__ import annotations

import xml.etree.ElementTree as ET

from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml


def _kml_ns(tag: str) -> str:
    return f"{{http://www.opengis.net/kml/2.2}}{tag}"


def test_build_overlap_corridor_kml_uses_polygon_and_is_well_formed() -> None:
    section = {
        "pipeline_1": "A & B",
        "pipeline_2": "C",
        "bundled_length_miles": 1.234,
        "average_separation": 12.3,
        "center_lon": -100.0,
        "center_lat": 40.0,
        "oriented_width_m": 25.0,
        "oriented_polygon": [
            (-100.0, 40.0),
            (-100.1, 40.0),
            (-100.1, 40.1),
            (-100.0, 40.1),
            (-100.0, 40.0),
        ],
    }

    kml = build_overlap_corridor_kml(section, 1)
    assert "A &amp; B" in kml  # XML escaping

    root = ET.fromstring(kml)
    assert root.tag == _kml_ns("kml")

    placemarks = root.findall(f".//{_kml_ns('Placemark')}")
    assert len(placemarks) == 2

    coords = root.find(f".//{_kml_ns('Polygon')}//{_kml_ns('coordinates')}")
    assert coords is not None
    txt = (coords.text or "").strip()
    assert "-100.0000000,40.0000000,0" in txt
    assert "-100.1000000,40.1000000,0" in txt


def test_build_overlap_corridor_kml_falls_back_to_bbox() -> None:
    section = {
        "pipeline_1": "A",
        "pipeline_2": "B",
        "bundled_length_miles": 0.1,
        "average_separation": 5.0,
        "center_lon": -100.0,
        "center_lat": 40.0,
        "bbox": {"min_lon": -100.0, "max_lon": -99.0, "min_lat": 40.0, "max_lat": 41.0},
    }

    kml = build_overlap_corridor_kml(section, 2)
    root = ET.fromstring(kml)
    coords = root.find(f".//{_kml_ns('Polygon')}//{_kml_ns('coordinates')}")
    assert coords is not None
    txt = (coords.text or "").strip()
    assert "-100.0000000,40.0000000,0" in txt
    assert "-99.0000000,41.0000000,0" in txt
