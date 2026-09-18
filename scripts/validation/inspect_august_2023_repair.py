"""Reproduce one supplied KMZ defect; inspect a temporary, byte-preserving fix.

This is a hash-pinned investigation, not the proposed general repair engine.
The supplied archive and application code are never modified.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics

EXPECTED_SHA256 = '92bea274a0e588e2611e4a46b96b8a8f48d3b26f9089533a1b9dfbbd89f1eb4c'


def inspect(source):
    archive_bytes = source.read_bytes()
    assert hashlib.sha256(archive_bytes).hexdigest() == EXPECTED_SHA256, 'Source differs from inspected sample'
    with zipfile.ZipFile(source) as archive:
        assert archive.namelist() == ['doc.kml']
        assert archive.testzip() is None
        original = archive.read('doc.kml')
    try:
        parse_kml_kmz_with_diagnostics(source)
    except ValueError as error:
        original_error = str(error)
    else:
        raise AssertionError('Expected the original input to fail')
    assert 'unbound prefix: line 6, column 1' in original_error
    # Deliberately specific to the inspected, hash-pinned UTF-8 document.
    offset = original.index(b'>', original.index(b'<kml '))
    addition = b' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
    assert offset == 211 and b'xmlns:xsi' not in original
    repaired = original[:offset] + addition + original[offset:]
    assert repaired[:offset] + repaired[offset + len(addition):] == original
    root = ET.fromstring(repaired)
    counts = Counter(element.tag.rsplit('}', 1)[-1] for element in root.iter())
    xsi_uses = [(element.tag.rsplit('}', 1)[-1], key.rsplit('}', 1)[-1])
                for element in root.iter() for key in element.attrib
                if key.startswith('{http://www.w3.org/2001/XMLSchema-instance}')]
    assert xsi_uses == [('Document', 'schemaLocation')] * 5
    # Supplemental lexical evidence only. The exact inverse patch above is the
    # stronger preservation proof; this regex is not a production XML scanner.
    pattern = rb'<coordinates\b[^>]*>(.*?)</coordinates\s*>'
    before = re.findall(pattern, original, re.S)
    after = re.findall(pattern, repaired, re.S)
    assert before == after and len(before) == counts['coordinates'] == 44
    raw_paths = [[tuple(map(float, token.split(b',')[:2])) for token in block.split()] for block in before]
    with tempfile.TemporaryDirectory(prefix='pipeline-repair-inspection-') as temporary:
        candidate = Path(temporary) / source.name
        with zipfile.ZipFile(candidate, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('doc.kml', repaired)
        parsed = parse_kml_kmz_with_diagnostics(candidate)
        paths = [path for pipeline in parsed.pipelines for path in pipeline['coordinate_paths']]
        assert paths == raw_paths
        assert not [d for d in parsed.diagnostics if d['level'] in ('warning', 'error')]
        _, meters, miles = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)
    assert source.read_bytes() == archive_bytes
    return {
        'source_filename': source.name, 'source_sha256': EXPECTED_SHA256,
        'archive_bytes': len(archive_bytes), 'kml_bytes': len(original),
        'original_error': original_error, 'patch_offset_zero_based': offset,
        'patch': addition.decode(), 'patch_bytes': len(addition),
        'xsi_schemaLocation_attributes': len(xsi_uses),
        'strict_xml_parse_passed': True, 'inverse_patch_recovers_exact_source': True,
        'source_archive_unchanged': True, 'coordinate_blocks_byte_identical': len(before),
        'raw_path_sequence_equals_application_paths': True,
        'pipelines': len(parsed.pipelines), 'paths': len(paths),
        'vertices': sum(map(len, paths)), 'point_placemarks': len(parsed.placemarks),
        'total_meters': meters, 'total_US_survey_miles': miles,
        'diagnostics': parsed.diagnostics,
        'limits': 'Parsing and original mileage only; overlap, state analysis, and full KML XSD validation were not run.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
