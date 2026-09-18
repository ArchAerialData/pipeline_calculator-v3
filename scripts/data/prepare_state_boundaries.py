"""Reproducibly bundle full-detail Census 2025 state polygons.

Run with requirements-dev installed:
  python scripts/data/prepare_state_boundaries.py --archive /path/tl_2025_us_state.zip

An explicit offline archive is required; runtime analysis never fetches data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pyproj
import shapefile
from shapely.geometry import shape, MultiPolygon
from shapely.ops import transform
from shapely import to_wkb

SOURCE_URL = "https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip"
SUPPORTED = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())

# EPSG operation definitions, explicitly fixed rather than selected using
# whichever optional transformation grids happen to exist on the build host.
_HELMERT = "proj=pipeline step proj=unitconvert xy_in=deg xy_out=rad step proj=push v_3 step proj=cart ellps=GRS80 step proj=helmert {translation} step inv proj=cart ellps=WGS84 step proj=pop v_3 step proj=unitconvert xy_in=rad xy_out=deg"
OPERATIONS = {
    "EPSG:1188": {"name": "NAD83 to WGS 84 (1)", "area": "Mainland North America",
                  "bounds": [-172.54, 23.81, -47.74, 86.46], "accuracy_meters": 4.0,
                  "pipeline": "+proj=noop"},
    "EPSG:1252": {"name": "NAD83 to WGS 84 (3)", "area": "Hawaii main islands",
                  "bounds": [-163.74, 15.56, -151.27, 25.58], "accuracy_meters": 4.0,
                  "pipeline": _HELMERT.format(translation="x=1 y=1 z=-1")},
    "EPSG:1251": {"name": "NAD83 to WGS 84 (2)", "area": "Aleutian Islands",
                  "bounds": [172.42, 51.3, -164.84, 54.34], "accuracy_meters": 8.0,
                  "pipeline": _HELMERT.format(translation="x=-2 y=0 z=4")},
    "BALLPARK": {"name": "Explicit approximate NAD83/WGS84 coordinate equivalence",
                 "area": "Components outside available published operation areas", "bounds": [-180, -90, 180, 90],
                 "accuracy_meters": None, "pipeline": "+proj=noop",
                 "notes": "Source coordinates retained under an approximate datum-equivalence convention. Accuracy is unknown, not four meters. No region-specific transformation is claimed."},
}


def _in_area(part, bounds):
    west, south, east, north = bounds
    for ring in [part.exterior, *part.interiors]:
        for lon, lat in ring.coords:
            longitude_ok = west <= lon <= east if west <= east else lon >= west or lon <= east
            if not longitude_ok or not south <= lat <= north:
                return False
    return True


def transform_components(geometry, code):
    transformed, lineage = [], []
    for index, part in enumerate(getattr(geometry, "geoms", [geometry])):
        # Prefer the region-specific operation where areas overlap. Apply one
        # operation to the complete component, preserving topology and holes.
        candidates = (["EPSG:1252"] if code == "HI" else ["EPSG:1251", "EPSG:1188"] if code == "AK" else ["EPSG:1188"])
        operation_id = next((op for op in candidates if _in_area(part, OPERATIONS[op]["bounds"])), "BALLPARK")
        operation = OPERATIONS[operation_id]
        transformed.append(transform(pyproj.Transformer.from_pipeline(operation["pipeline"]).transform, part))
        lineage.append({"component_index": index, "source_bounds": list(part.bounds), "operation": operation_id})
    result = transformed[0] if len(transformed) == 1 else MultiPolygon(transformed)
    if not result.is_valid:
        raise ValueError(f"Datum transformation produced invalid topology: {code}")
    return result, lineage


def prepare(archive_path, output):
    raw_archive = Path(archive_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as source:
        prefix = "tl_2025_us_state"
        prj = source.read(prefix + ".prj").decode("ascii").strip()
        crs = pyproj.CRS.from_wkt(prj)
        if crs.to_epsg() != 4269:
            raise ValueError("Expected EPSG:4269 NAD83 source coordinates")
        reader = shapefile.Reader(shp=io.BytesIO(source.read(prefix + ".shp")),
                                  shx=io.BytesIO(source.read(prefix + ".shx")),
                                  dbf=io.BytesIO(source.read(prefix + ".dbf")))
        records = []
        blobs = {}
        component_operations = {}
        for record in reader.iterShapeRecords():
            attributes = record.record.as_dict()
            code = attributes["STUSPS"]
            if code not in SUPPORTED:
                continue
            geometry = shape(record.shape.__geo_interface__)
            if not geometry.is_valid:
                raise ValueError(f"Invalid source state topology: {code}")
            geometry, component_operations[code] = transform_components(geometry, code)
            raw = to_wkb(geometry, byte_order=1, output_dimension=2)
            filename = f"states/{code}.wkb"
            blobs[filename] = raw
            records.append({"code": code, "name": attributes["NAME"], "fips": attributes["STATEFP"],
                            "file": filename, "sha256": hashlib.sha256(raw).hexdigest()})
    if {r["code"] for r in records} != SUPPORTED:
        raise ValueError("Missing supported states")
    manifest = {
        "schema_version": 1, "name": "U.S. Census Bureau TIGER/Line States", "vintage": "2025",
        "source_url": SOURCE_URL, "source_sha256": hashlib.sha256(raw_archive).hexdigest(),
        "source_crs": "EPSG:4269", "source_crs_wkt": prj, "coordinate_crs": "EPSG:4326",
        "transformation": {"operations": OPERATIONS, "components": component_operations,
                           "notes": "Area-aware component operations fixed at build time without optional grids. Published operation accuracy is not a surveyed boundary guarantee. BALLPARK components have unknown datum accuracy."},
        "approximate_regions": [{"state_code": code, **entry} for code, entries in sorted(component_operations.items())
                                for entry in entries if entry["operation"] == "BALLPARK"],
        "boundary_edge_model": "Coordinate-linear edges through transformed source vertices in longitude/latitude; original vertex count, islands, holes and water retained; no simplification.",
        "attribution": "U.S. Census Bureau, 2025 TIGER/Line Shapefiles. Public domain U.S. government data.",
        "coverage": "50 United States and Washington, DC; no territories",
        "states": sorted(records, key=lambda r: r["code"]),
    }
    blobs["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for filename, raw in sorted(blobs.items()):
            info = zipfile.ZipInfo(filename, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, raw, compresslevel=9)
    print(f"Bundled {len(records)} jurisdictions: {output} ({output.stat().st_size:,} bytes)")
    print(f"SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[2] / "src/pipeline_calculator/data/states_2025.zip"))
    args = parser.parse_args()
    prepare(args.archive, args.output)
