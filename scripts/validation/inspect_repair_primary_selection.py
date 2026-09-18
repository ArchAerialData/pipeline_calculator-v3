"""Reproduce the saved-KMZ primary-selection hazard with synthetic input only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_calculator.parsers.kml_kmz import _select_primary_kml, parse_kml_kmz_with_diagnostics


def inspect():
    namespace = 'http://www.opengis.net/kml/2.2'
    line = ('<Placemark><name>{}</name><LineString><coordinates>'
            '-100,30 -100.01,30</coordinates></LineString></Placemark>')
    parent = (f'<kml xmlns="{namespace}"><Document>{line.format("Root line")}'
              '<NetworkLink><Link><href>child.kml</href></Link></NetworkLink></Document></kml>')
    child = (f'<kml xmlns="{namespace}"><Document xsi:schemaLocation="{namespace} schema.xsd">'
             f'{line.format("Child line")}</Document></kml>')
    child = child.replace('</kml>', '<!--' + 'x' * len(parent) + '--></kml>')
    parent = parent.replace('</kml>', '<!--' + 'x' * (len(child) + 10 - len(parent) - 7) + '--></kml>')
    addition = ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
    repaired = child.replace('<kml ', '<kml' + addition + ' ', 1)
    report = {
        'finding': 'primary_document_can_change_after_repair',
        'parser_sha256': hashlib.sha256((ROOT / 'src/pipeline_calculator/parsers/kml_kmz.py').read_bytes()).hexdigest(),
        'root_bytes': len(parent), 'child_before_bytes': len(child), 'child_after_bytes': len(repaired),
        'patch_bytes': len(addition), 'inputs': {'root.kml': parent, 'child.kml': child},
    }
    with tempfile.TemporaryDirectory(prefix='repair-selection-audit-') as temporary:
        for label, body in [('original', child), ('repaired', repaired)]:
            path = Path(temporary) / (label + '.kmz')
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('root.kml', parent)
                archive.writestr('child.kml', body)
            with zipfile.ZipFile(path) as archive:
                report[label + '_primary'] = _select_primary_kml(archive.infolist()).filename
            parsed = parse_kml_kmz_with_diagnostics(path)
            report[label + '_documents'] = parsed.parsed_kml_files
            report[label + '_pipelines'] = [pipeline['name'] for pipeline in parsed.pipelines]
            report[label + '_diagnostic_codes'] = [diagnostic['code'] for diagnostic in parsed.diagnostics]
    assert report['original_primary'] == 'root.kml'
    assert report['repaired_primary'] == 'child.kml'
    assert report['repaired_pipelines'] == ['Child line']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = inspect()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Reproduced: a 54-byte child repair changes the selected primary and pipeline set.')
