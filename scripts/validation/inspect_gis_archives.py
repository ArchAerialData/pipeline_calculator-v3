"""Reproduce planning evidence for two hash-pinned customer shapefile ZIPs.

This is an independent fixture inspector, not the application's GIS importer.
It never extracts archive members, parses metadata XML, or prints attribute values.
Run with the repository development environment; pyshp and pyproj are required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import struct
import zipfile

import pyproj
import shapefile


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/gis/customer_zip"
SAMPLES = {
    "RF_Lehman_Pipes.zip": "10b57d3b76c2e23304233fa559d9971c2956052e5e55a56ee327b11cd7fee1e4",
    "Verdun_Pipes.zip": "3dfea6159b5e7ef9bec27118dde484d3ac2c04f9b460384304c01a25359af44d",
}
DIGEST_DESCRIPTION = (
    "SHA-256 of binary stream starting with ASCII pipeline-xy-paths-v1 followed by NUL. "
    "For each SHP record in physical order, append ASCII R, big-endian uint64 zero-based "
    "record index, uint32 shape type, uint32 part count. For each part append ASCII P, "
    "uint32 zero-based part index, uint64 vertex count, then each XY coordinate as two "
    "big-endian IEEE-754 float64 values in source order, then ASCII e. End each record "
    "with ASCII E. Null records have zero parts and are included. Z/M and DBF values "
    "are excluded. The explicit record/part boundaries prevent accidental joining."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dimension_summary(values: list) -> dict:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "slots": len(values),
        "missing": sum(value is None for value in values),
        "nonfinite": sum(value is not None and not math.isfinite(value) for value in values),
        "finite": len(finite),
        "zero": sum(value == 0 for value in finite),
        "nonzero": sum(value != 0 for value in finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def identifier_summary(records: list, fields: list) -> dict:
    result = {}
    for index, field in enumerate(fields):
        if field[0].lower() not in {"pline_id", "globalid", "objectid", "fid", "id"}:
            continue
        values = [record[index] for record in records if record is not None]
        present = [value for value in values if value is not None and str(value).strip()]
        counts = Counter(present)
        result[field[0]] = {
            "active_records": len(values),
            "nonempty": len(present),
            "empty": len(values) - len(present),
            "distinct_nonempty_values": len(counts),
            "duplicated_distinct_values": sum(count > 1 for count in counts.values()),
            "records_in_duplicate_groups": sum(count for count in counts.values() if count > 1),
            "unique_and_nonempty_for_all_active_records": len(counts) == len(values) == len(present),
        }
    return result


def inspect_archive(path: Path) -> dict:
    data = path.read_bytes()
    original_hash = sha256(data)
    require(path.name in SAMPLES, f"Unrecognized fixture name: {path.name}")
    require(original_hash == SAMPLES[path.name], f"Fixture hash mismatch: {path.name}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        require(len(entries) <= 32, "Unexpected archive entry count")
        require(sum(entry.file_size for entry in entries) <= 8 * 1024 * 1024, "Archive exceeds fixture budget")
        require(len({entry.filename for entry in entries}) == len(entries), "Duplicate archive member")
        require(not any(entry.flag_bits & 1 for entry in entries), "Encrypted fixture")
        bad_crc = archive.testzip()
        require(bad_crc is None, f"ZIP CRC failure: {bad_crc}")
        shp_names = [entry.filename for entry in entries if entry.filename.lower().endswith(".shp")]
        require(len(shp_names) == 1, "Inspector expects exactly one shapefile dataset")
        stem = str(PurePosixPath(shp_names[0]).with_suffix(""))
        names = {entry.filename.lower(): entry.filename for entry in entries}
        payload = {}
        for extension in ("shp", "shx", "dbf", "prj", "cpg"):
            key = f"{stem}.{extension}".lower()
            require(key in names, f"Missing fixture component: {key}")
            payload[extension] = archive.read(names[key])
        members = [{
            "name": entry.filename,
            "uncompressed_bytes": entry.file_size,
            "compressed_bytes": entry.compress_size,
            "crc32": f"{entry.CRC:08x}",
            "role": "opaque metadata; not parsed" if entry.filename.lower().endswith(".shp.xml") else
                    "spatial index; not used" if PurePosixPath(entry.filename).suffix.lower() in {".sbn", ".sbx"} else
                    "shapefile component",
        } for entry in entries]

    encoding = payload["cpg"].decode("ascii").strip()
    source_crs = pyproj.CRS.from_wkt(payload["prj"].decode("utf-8"))
    require(source_crs.to_authority() == ("EPSG", "4269"), "This sample-specific reference requires NAD83 EPSG:4269")
    # Pin the operation instead of accepting PROJ's host-dependent default selection.
    # It is suitable only after checking these geographic sample extents below.
    operation = pyproj.Transformer.from_pipeline("EPSG:1188")
    require(operation.to_proj4() == "+proj=noop", "Expected EPSG:1188 horizontal noop representation")
    require(operation.accuracy == 4.0, "Unexpected EPSG:1188 operation accuracy")
    area = operation.area_of_use
    require(area is not None, "Missing operation area of use")
    geod = pyproj.Geod(ellps="GRS80")

    dbf = payload["dbf"]
    require(len(dbf) >= 32, "Truncated DBF header")
    dbf_count = struct.unpack_from("<I", dbf, 4)[0]
    header_bytes, row_bytes = struct.unpack_from("<HH", dbf, 8)
    require(row_bytes > 0 and header_bytes + dbf_count * row_bytes <= len(dbf), "Truncated DBF records")
    flags = [dbf[header_bytes + index * row_bytes] for index in range(dbf_count)]
    require(all(flag in (0x20, 0x2A) for flag in flags), "Unexpected DBF deletion marker")
    deleted = [index for index, flag in enumerate(flags) if flag == 0x2A]
    shx_count = (len(payload["shx"]) - 100) // 8
    require(len(payload["shx"]) >= 100 and (len(payload["shx"]) - 100) % 8 == 0, "Invalid SHX size")

    with shapefile.Reader(shp=io.BytesIO(payload["shp"]), shx=io.BytesIO(payload["shx"]),
                          dbf=io.BytesIO(dbf), encoding=encoding, encodingErrors="strict") as reader:
        shapes = list(reader.iterShapes())
        # Read each physical row independently: iterShapeRecords can omit deleted
        # rows and must not be used to establish SHP/DBF alignment or null counts.
        records = [reader.record(index) for index in range(dbf_count)]
        fields = list(reader.fields[1:])
        require(len(shapes) == dbf_count == shx_count == reader.numRecords, "SHP/SHX/DBF row count mismatch")
        require([index for index, row in enumerate(records) if row is None] == deleted, "DBF deletion flag disagreement")
        require(reader.shapeType == shapefile.POLYLINEZ, "Unexpected fixture header geometry type")
        header_bounds = list(reader.bbox)

    all_xy = [point for shape in shapes for point in shape.points]
    require(all_xy and all(math.isfinite(x) and math.isfinite(y) for x, y in all_xy), "Invalid or absent XY coordinates")
    bounds = [min(x for x, _ in all_xy), min(y for _, y in all_xy),
              max(x for x, _ in all_xy), max(y for _, y in all_xy)]
    require(bounds == header_bounds, "SHP header bounds differ from actual vertices")
    whole_extent_contained = area.west <= bounds[0] <= bounds[2] <= area.east and area.south <= bounds[1] <= bounds[3] <= area.north
    require(whole_extent_contained, "Sample extent exceeds EPSG:1188 area of use")
    digest = hashlib.sha256(b"pipeline-xy-paths-v1\x00")
    part_lengths = []
    record_lengths = []
    parts_count = 0
    multipart_count = 0
    null_indices = []
    transformed_vertices = 0
    for record_index, shape in enumerate(shapes):
        record_part_lengths = []
        require(shape.shapeType in (shapefile.NULL, shapefile.POLYLINEZ), "Unexpected per-record geometry type")
        require(shape.oid == record_index, "Physical SHP record identity changed")
        starts = list(shape.parts) if shape.shapeType != shapefile.NULL else []
        if shape.shapeType == shapefile.NULL:
            null_indices.append(record_index)
            require(not shape.points, "Null geometry contains points")
        else:
            require(starts and starts[0] == 0 and starts == sorted(set(starts)), "Invalid part offsets")
        multipart_count += len(starts) > 1
        parts_count += len(starts)
        digest.update(b"R" + struct.pack(">QII", record_index, shape.shapeType, len(starts)))
        ends = starts[1:] + [len(shape.points)]
        for part_index, (start, end) in enumerate(zip(starts, ends)):
            require(0 <= start < end <= len(shape.points), "Invalid part range")
            points = shape.points[start:end]
            require(len(points) >= 2, "Degenerate line part")
            xs, ys = zip(*points)
            tx, ty = operation.transform(xs, ys, errcheck=True)
            # EPSG:1188 is explicitly no-op here, so no axis/order or datum shift
            # can silently enter the sample reference. General imports need their
            # own axis/operation selection policy; this is not such a policy.
            require(list(zip(tx, ty)) == list(map(tuple, points)), "Unexpected horizontal coordinate change")
            transformed_vertices += len(points)
            digest.update(b"P" + struct.pack(">IQ", part_index, len(points)))
            for x, y in zip(tx, ty):
                digest.update(struct.pack(">dd", x, y))
            digest.update(b"e")
            part_length = geod.line_length(tx, ty)
            part_lengths.append(part_length)
            record_part_lengths.append(part_length)
        digest.update(b"E")
        record_lengths.append(None if shape.shapeType == shapefile.NULL else math.fsum(record_part_lengths))
    require(transformed_vertices == len(all_xy), "Vertex coverage mismatch")
    require(sha256(path.read_bytes()) == original_hash, "Inspection changed the fixture")

    return {
        "fixture": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
        "sha256": original_hash,
        "hash_matches_pinned_original": True,
        "archive_bytes": len(data),
        "archive_crc_check": "passed",
        "archive_entries": members,
        "dataset": stem,
        "encoding": encoding,
        "counts": {
            "shp_records": len(shapes), "shx_records": shx_count, "dbf_records": dbf_count,
            "active_attribute_records": dbf_count - len(deleted), "deleted_attribute_records": len(deleted),
            "nonnull_geometry_records": len(shapes) - len(null_indices), "null_geometry_records": len(null_indices),
            "multipart_geometry_records": multipart_count, "parts": parts_count, "xy_vertices": len(all_xy),
            "line_edges": len(all_xy) - parts_count,
        },
        "null_geometry_record_indices_zero_based": null_indices,
        "null_geometry_record_ordinals_one_based": [index + 1 for index in null_indices],
        "null_geometry_identifier_note": "Indices are physical SHP/DBF zero-based record positions (pyshp oid), not PLINE_ID or GlobalID values.",
        "deleted_attribute_record_indices_zero_based": deleted,
        "shape_type_counts": dict(sorted(Counter(shapefile.SHAPETYPE_LOOKUP[shape.shapeType] for shape in shapes).items())),
        "header_shape_type": "POLYLINEZ (13)",
        "bounds_xy": bounds,
        "bounds_match_header": True,
        "dimensions": {
            "z": dimension_summary([value for shape in shapes for value in getattr(shape, "z", [])]),
            "m": dimension_summary([value for shape in shapes for value in getattr(shape, "m", [])]),
            "measurement_policy": "Horizontal XY only; Z/M retained in the original archive and summarized without interpreting their units or semantics.",
        },
        "fields": [{"name": f[0], "dbf_type": f[1], "width": f[2], "decimal_places": f[3]} for f in fields],
        "identifier_summaries": identifier_summary(records, fields),
        "crs": {
            "authority": list(source_crs.to_authority()), "name": source_crs.name,
            "source_wkt": payload["prj"].decode("utf-8"), "is_geographic": source_crs.is_geographic,
            "coordinate_storage_order": "shapefile X=longitude, Y=latitude",
            "operation": "EPSG:1188", "operation_name": operation.description,
            "operation_proj_pipeline": operation.to_proj4(), "operation_accuracy_m": operation.accuracy,
            "accuracy_note": "4 m is the coordinate operation's stated accuracy, not the source survey accuracy or a mileage error bound.",
            "operation_area_bounds_west_south_east_north": [area.west, area.south, area.east, area.north],
            "whole_vertex_extent_contained_in_operation_area": whole_extent_contained,
            "output_crs": "EPSG:4326", "horizontal_coordinates_identical": True,
        },
        "xy_path_digest_sha256": digest.hexdigest(),
        "independent_reference": {
            "method": "pyproj.Geod(ellps='GRS80').line_length per actual SHP part; math.fsum across parts; no connection between records/parts",
            "total_horizontal_length_m": math.fsum(part_lengths),
            "total_horizontal_length_us_survey_miles": math.fsum(part_lengths) / (5280 * 1200 / 3937),
            "per_record_horizontal_length_m": record_lengths,
            "per_record_index_note": "List index is the zero-based physical SHP record index; ordinal = index + 1. Each line value is math.fsum of its own part lengths; null geometry is null, not zero.",
            "scope": "Independent of application parser/analyzer; geodesic library is pyproj, also used by the application, so not an independent geodesic implementation.",
        },
        "fixture_hash_unchanged_after_inspection": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-directory", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/gis-import-samples.json")
    parser.add_argument("--verify-originals", type=Path, nargs=2, metavar=("RF_ORIGINAL", "VERDUN_ORIGINAL"),
                        help="Also hash-check both user originals; never modify them.")
    args = parser.parse_args()
    input_paths = [(args.fixture_directory / name).resolve() for name in SAMPLES]
    input_paths.extend(path.resolve() for path in (args.verify_originals or []))
    require(args.output.resolve() not in input_paths, "Report output must not overwrite any inspected archive")
    originals = {}
    if args.verify_originals:
        for name, path in zip(SAMPLES, args.verify_originals):
            original_hash = sha256(path.read_bytes())
            require(original_hash == SAMPLES[name], f"Original hash mismatch: {name}")
            originals[name] = {"path": path, "before": original_hash}
    report = {
        "schema_version": 1,
        "purpose": "Planning evidence and future GIS importer regression fixture inventory; no app implementation",
        "environment": {"python": platform.python_version(), "pyshp": shapefile.__version__,
                        "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str,
                        "epsg_database": pyproj.database.get_database_metadata("EPSG.VERSION")},
        "xy_path_digest_specification": DIGEST_DESCRIPTION,
        "samples": [inspect_archive((args.fixture_directory / name).resolve()) for name in SAMPLES],
        "limitations": [
            "Only these exact hash-pinned customer archives are accepted; not a general archive validator or production importer.",
            "The XML metadata files are opaque archive bytes. Their contents are not parsed, executed, or treated as instructions.",
            "CRC verifies archive consistency, not correctness or source survey accuracy.",
            "Both samples are NAD83 geographic PolyLineZ; they do not establish GPKG, projected CRS, Point, MultiPoint, Polygon, or mixed-layer support.",
            "EPSG:1188 is explicitly pinned after whole-extent area-of-use containment for these samples only; not a global transformation fallback.",
            "No application import, overlap analysis, state analysis, packaged execution, or automated regression test is run by this inspector.",
            "Identifier summaries expose uniqueness counts, not raw customer attribute values. Source Z/M semantics and accuracy are unknown.",
        ],
    }
    if originals:
        report["original_copy_verification"] = {}
        for name, evidence in originals.items():
            after = sha256(evidence["path"].read_bytes())
            require(after == evidence["before"] == SAMPLES[name], f"Original changed during inspection: {name}")
            report["original_copy_verification"][name] = {"original_sha256_before": evidence["before"],
                "original_sha256_after": after, "fixture_sha256": SAMPLES[name], "all_equal": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "samples": [{"name": sample["dataset"],
        "counts": sample["counts"], "xy_path_digest_sha256": sample["xy_path_digest_sha256"],
        "length_m": sample["independent_reference"]["total_horizontal_length_m"]} for sample in report["samples"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
