"""Probe planned repair mechanisms using small, authored synthetic KML only.

These fixed, fixture-specific patches are investigation evidence, not a repair
detector, production verifier, or acceptance tests for an implemented feature.
No customer files or application code are modified. Temporary inputs are removed.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics

BOM = b"\xef\xbb\xbf"
SPACE = b" \t\r\n"
XSI = b' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
GEOMETRY = b"""<MultiGeometry>
<LineString><coordinates>-100,34,4 -100.01,34,5 -100.01,34.01,6</coordinates></LineString>
<LineString><coordinates>-99.9,34,7 -99.91,34.01,8</coordinates></LineString>
<Point><coordinates>-100,34,9</coordinates></Point>
<Polygon><outerBoundaryIs><LinearRing><coordinates>-100,34,1 -100.1,34,2 -100.1,34.1,3 -100,34,1</coordinates></LinearRing></outerBoundaryIs>
<innerBoundaryIs><LinearRing><coordinates>-100.05,34.01,1 -100.06,34.01,2 -100.06,34.02,3 -100.05,34.01,1</coordinates></LinearRing></innerBoundaryIs></Polygon>
</MultiGeometry>"""
TRACK = b"""<gx:Track><altitudeMode>absolute</altitudeMode>
<when>2023-08-01T00:00:00Z</when><when>2023-08-01T00:01:00Z</when>
<gx:coord>-99.5 34 12</gx:coord><gx:coord>-99.51 34.01 13</gx:coord></gx:Track>"""
GOLDEN = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<kml xmlns="http://www.opengis.net/kml/2.2"'
    b' xmlns:gx="http://www.google.com/kml/ext/2.2"' + XSI + b'>\n'
    b'<Document xsi:schemaLocation="http://www.opengis.net/kml/2.2 schema.xsd">\n'
    b'<Placemark id="branch"><name>Caf\xc3\xa9 A &amp; B</name>'
    b'<description>North &amp; South &lt; 10</description><Snippet>Survey &amp;</Snippet>'
    + GEOMETRY + b'</Placemark>\n'
    b'<Placemark id="track"><name>Track</name>'
    b'<description><![CDATA[Leave A & B <i>alone</i>]]></description>'
    + TRACK + b'</Placemark>\n'
    b'<!-- Preserve literal & and < here -->\n</Document></kml>\n'
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_bytes(data):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("doc.kml", date_time=(2023, 8, 1, 0, 0, 0))
        archive.writestr(info, data)
        archive.writestr(zipfile.ZipInfo("notes.txt", date_time=(2023, 8, 1, 0, 0, 0)), b"Untouched resource")
    return output.getvalue()


def projection(path):
    parsed = parse_kml_kmz_with_diagnostics(path)
    # The one explicit standalone-root relocation mapping. Other fields remain.
    records = [dict(pipeline) for pipeline in parsed.pipelines]
    for pipeline in records:
        if pipeline["source_kml"] == str(path.resolve()):
            pipeline["source_kml"] = "<standalone-root>"
    _, meters, miles = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)
    return {
        "pipelines": records,
        "placemarks": parsed.placemarks,
        "diagnostic_codes": [d["code"] for d in parsed.diagnostics],
        "meters": meters,
        "US_survey_miles": miles,
    }


def expect_failure(path):
    try:
        projection(path)
    except ValueError as error:
        # Do not export private temporary directory names in portable evidence.
        return str(error).replace(str(path.resolve()), "<source>")
    raise AssertionError(f"Expected rejection: {path.name}")


def fixed_patches(original, *, preamble=False, xsi=False, metadata=()):
    """Known fixture locations only; deliberately NOT a generic XML recognizer."""
    patches = []
    if preamble:
        offset = len(BOM) if original.startswith(BOM) else 0
        assert original[offset:offset + len(SPACE)] == SPACE
        patches.append((offset, SPACE, b"", "leading_xml_whitespace_v1"))
    if xsi:
        marker = b'<kml xmlns="http://www.opengis.net/kml/2.2"'
        offset = original.index(marker) + len(marker)
        patches.append((offset, b"", XSI, "missing_xsi_schema_namespace_v1"))
    for old, new in metadata:
        assert original.count(old) == 1
        offset = original.index(old) + old.index(b"&")
        assert new == old.replace(b"&", b"&amp;", 1)
        patches.append((offset, b"&", b"&amp;", "literal_metadata_ampersand_v1"))
    patches.sort()
    result, end = bytearray(), 0
    inverse = []
    for offset, before, after, rule in patches:
        assert offset >= end and original[offset:offset + len(before)] == before
        result.extend(original[end:offset])
        inverse.append((len(result), after, before))
        result.extend(after)
        end = offset + len(before)
    result.extend(original[end:])
    candidate = bytes(result)
    restored = candidate
    for offset, after, before in reversed(inverse):
        assert restored[offset:offset + len(after)] == after
        restored = restored[:offset] + before + restored[offset + len(after):]
    assert restored == original
    return candidate, [
        {"rule": rule, "original_byte_offset": offset,
         "before": before.decode("utf-8"), "after": after.decode("utf-8")}
        for offset, before, after, rule in patches
    ]


def inspect():
    edits = [
        (b"Caf\xc3\xa9 A & B", b"Caf\xc3\xa9 A &amp; B"),
        (b"North & South", b"North &amp; South"),
        (b"Survey &</Snippet>", b"Survey &amp;</Snippet>"),
    ]
    cases = [("leading_whitespace", SPACE + GOLDEN, {"preamble": True}),
             ("bom_then_whitespace", BOM + SPACE + GOLDEN, {"preamble": True})]
    for name, edit in zip(("name_ampersand", "description_ampersand", "snippet_ampersand"), edits):
        cases.append((name, GOLDEN.replace(edit[1], edit[0]), {"metadata": [edit]}))
    combined = SPACE + GOLDEN.replace(XSI, b"")
    for old, new in edits:
        combined = combined.replace(new, old)
    cases.append(("combined_allowlisted_defects", combined,
                  {"preamble": True, "xsi": True, "metadata": edits}))
    report = {
        "scope": "Synthetic mechanism evidence only; no production repair detector or verifier exists.",
        "limits": "No prevalence estimate, schema validation, overlap/state analysis, archive fuzzing, or GUI test.",
        "parser_sha256": digest((ROOT / "src/pipeline_calculator/parsers/kml_kmz.py").read_bytes()),
        "golden_sha256": digest(GOLDEN),
        "geometry_sha256": digest(GEOMETRY + TRACK),
        "repair_probes": [], "format_probes": [], "already_valid_probes": [],
        "refused_defect_observations": [],
    }
    with tempfile.TemporaryDirectory(prefix="repair-catalog-probe-") as directory:
        base = Path(directory)
        baselines = {}
        for suffix in (".kml", ".kmz"):
            path = base / ("golden" + suffix)
            path.write_bytes(GOLDEN if suffix == ".kml" else archive_bytes(GOLDEN))
            baselines[suffix] = projection(path)
        report["baseline"] = {
            "pipelines": len(baselines[".kml"]["pipelines"]),
            "paths": sum(len(p["coordinate_paths"]) for p in baselines[".kml"]["pipelines"]),
            "vertices": sum(len(c) for p in baselines[".kml"]["pipelines"] for c in p["coordinate_paths"]),
            "meters": baselines[".kml"]["meters"],
            "US_survey_miles": baselines[".kml"]["US_survey_miles"],
        }
        for label, original, kwargs in cases:
            candidate, patches = fixed_patches(original, **kwargs)
            ET.fromstring(candidate)
            # Golden root attributes may differ only in order for the xsi case.
            expected = (BOM if original.startswith(BOM) else b"") + GOLDEN
            if kwargs.get("xsi"):
                expected = expected.replace(XSI, b"").replace(
                    b'<kml xmlns="http://www.opengis.net/kml/2.2"',
                    b'<kml xmlns="http://www.opengis.net/kml/2.2"' + XSI, 1)
            assert candidate == expected
            assert all(original.count(chunk) == candidate.count(chunk) == 1 for chunk in (GEOMETRY, TRACK))
            for suffix in (".kml", ".kmz"):
                source = base / (label + suffix)
                source.write_bytes(original if suffix == ".kml" else archive_bytes(original))
                before = source.read_bytes()
                error = expect_failure(source)
                target = base / (label + "_repaired" + suffix)
                target.write_bytes(candidate if suffix == ".kml" else archive_bytes(candidate))
                assert projection(target) == baselines[suffix]
                assert source.read_bytes() == before
                report["repair_probes"].append({
                    "case": label, "container": suffix, "original_error": error,
                    "patches": patches, "inverse_exact": True, "source_unchanged": True,
                    "geometry_bytes_identical": True, "golden_projection_and_mileage_equal": True,
                })
        for actual, wrong in ((".kml", ".kmz"), (".kmz", ".kml")):
            payload = GOLDEN if actual == ".kml" else archive_bytes(GOLDEN)
            source = base / ("misnamed_" + actual[1:] + wrong)
            source.write_bytes(payload)
            error = expect_failure(source)
            target = base / ("corrected_" + actual[1:] + actual)
            target.write_bytes(payload)
            assert projection(target) == baselines[actual]
            assert source.read_bytes() == target.read_bytes() == payload
            report["format_probes"].append({
                "named_as": wrong, "actual_format": actual, "original_error": error,
                "source_and_saved_content_byte_identical": True,
                "golden_projection_and_mileage_equal": True,
            })
        for label, body in (
            ("single_bom", BOM + GOLDEN), ("trailing_xml_whitespace", GOLDEN + SPACE),
            ("leading_whitespace_without_declaration", SPACE + GOLDEN.split(b"\n", 1)[1]),
            ("valid_entities_cdata_comments", GOLDEN),
            ("utf16_declared_with_bom", GOLDEN.decode().replace('encoding="UTF-8"', 'encoding="UTF-16"').encode("utf-16")),
        ):
            source = base / (label + ".kml")
            source.write_bytes(body)
            assert projection(source) == baselines[".kml"]
            report["already_valid_probes"].append({"case": label, "ordinary_parser_accepts_without_edit": True})
        refused = {
            "truncated_document": GOLDEN[:-12],
            "undefined_named_entity": GOLDEN.replace(b"A &amp; B", b"A &custom; B"),
            "unfinished_entity": GOLDEN.replace(b"A &amp; B", b"A &amp B"),
            "second_root": GOLDEN + b"<kml/>",
            "missing_geometry_namespace": GOLDEN.replace(b' xmlns:gx="http://www.google.com/kml/ext/2.2"', b""),
            "invalid_coordinate": GOLDEN.replace(b"-100.01,34,5", b"NaN,34,5"),
            "wrong_geometry_tag_case": GOLDEN.replace(b"LineString", b"linestring"),
        }
        for label, body in refused.items():
            path = base / (label + ".kml")
            path.write_bytes(body)
            try:
                result = projection(path)
            except ValueError as error:
                observation = {"error": str(error).replace(str(path.resolve()), "<source>")}
            else:
                observation = {"diagnostic_codes": result["diagnostic_codes"],
                               "projection_matches_golden": result == baselines[".kml"]}
                assert result != baselines[".kml"]
            report["refused_defect_observations"].append({"case": label, "current_parser": observation,
                                                        "repair_attempted": False})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(result['repair_probes'])} fixed-patch probes, "
          f"{len(result['format_probes'])} byte-identical format corrections, "
          f"{len(result['already_valid_probes'])} already-valid controls, and "
          f"{len(result['refused_defect_observations'])} refusal observations.")
