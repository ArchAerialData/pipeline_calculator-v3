"""Render complete source/corridor galleries with independent radius checks.

Overlays intentionally contain source LineStrings and are review artifacts, not
reimport fixtures. The separate preview KMLs contain polygons only.
"""
from pathlib import Path
import argparse
from contextlib import nullcontext
import html
import json
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation.common import environment, gallery_specs, save_json, write_fixture
from scripts.validation.corridor_audit import inspect_document, section_runs
from scripts.validation.audit_dataset import CapturedAnalyzer
from pipeline_calculator.core import overlap
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml

NS = {'k': 'http://www.opengis.net/kml/2.2'}


def render_svg(path, parts, paths, title):
    west = min(part.bounds[0] for part in parts); south = min(part.bounds[1] for part in parts)
    east = max(part.bounds[2] for part in parts); north = max(part.bounds[3] for part in parts)
    pad = max(10, (east-west)*.06, (north-south)*.06)
    strokes = max(east-west, north-south)/500
    drawings = []
    for part in parts:
        commands = []
        for ring in [part.exterior, *part.interiors]:
            points = list(ring.coords)
            commands.append(f'M {points[0][0]} {-points[0][1]} ' +
                            ' '.join(f'L {x} {-y}' for x, y in points[1:]) + ' Z')
        drawings.append(f'<path d="{" ".join(commands)}" fill="#6bc7ba" fill-rule="evenodd" fill-opacity=".65" '
                        f'stroke="#147b70" stroke-width="{strokes}"/>')
    for line in paths:
        drawings.append('<polyline points="' + ' '.join(f'{x},{-y}' for x, y in line) +
                        f'" fill="none" stroke="#252d44" stroke-width="{strokes}"/>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="700" height="500" '
                    f'viewBox="{west-pad} {-north-pad} {east-west+2*pad} {north-south+2*pad}">'
                    f'<title>{html.escape(title)}</title>' + ''.join(drawings) + '</svg>', encoding='utf-8')


def extra_specs():
    shapes = {
        'U_shape': [(0, 0), (0, 200), (150, 200), (150, 0)],
        'S_shape': [(0, 0), (0, 100), (70, 180), (0, 260), (0, 400)],
    }
    result = [{'id': name, 'paths': [[path], [[(x+2, y) for x, y in path]]], 'minimum': 10}
              for name, path in shapes.items()]
    result.append({'id': 'branch_rejoin', 'paths': [
        [[(0, 0), (0, 250), (0, 450), (0, 700)]],
        [[(2, 0), (2, 250), (102, 350), (2, 450), (2, 700)]],
    ], 'minimum': 10})
    result.append({'id': 'separate_buffer_parts', 'paths': [[[(0, 0), (0, 302)]], [[(12, 0), (12, 302)]]], 'minimum': 10})
    result.append({'id': 'unavailable_map', 'paths': [[[(0, 0), (0, 302)]], [[(2, 0), (2, 302)]]],
                   'minimum': 10, 'inject_failure': True})
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in [*gallery_specs(), *extra_specs()]:
        origin = spec.get('origin', (-100, 40)); bearing = spec.get('bearing', 0)
        source_path = write_fixture(output / f"{spec['id']}-source.kml", spec['paths'], origin=origin, bearing=bearing)
        analyzer = CapturedAnalyzer(min_parallel_length=spec['minimum'])
        failure = (patch.object(overlap, 'build_buffered_corridor', side_effect=RuntimeError('gallery injected map failure'))
                   if spec.get('inject_failure') and hasattr(overlap, 'build_buffered_corridor') else nullcontext())
        with failure:
            result = analyzer.analyze_complete(source_path)
        save_json(output / f"{spec['id']}-result.json", result)
        sections = (result.get('overlap_analysis') or {}).get('bundled_sections', [])
        qualified = sorted(analyzer.audit_sections, key=lambda s: s['length'], reverse=True)
        if len(qualified) != len(sections) or not result['analysis_complete']:
            rows.append({'fixture': spec['id'], 'status': 'failed', 'reason': 'Numerical sections unavailable or incomplete'})
            continue
        if not sections:
            rows.append({'fixture': spec['id'], 'status': 'failed', 'reason': 'No qualifying control section'})
        for index, (section, source) in enumerate(zip(sections, qualified), 1):
            row = {'fixture': spec['id'], 'section': index, 'viewer': 'pending', 'status': 'ready',
                   'savings_m': result['overlap_analysis']['savings_meters'], 'analysis_complete': result['analysis_complete']}
            rows.append(row)
            if section.get('visualization_status') == 'omitted':
                try:
                    build_overlap_corridor_kml(section, index)
                except ValueError:
                    row.update(status='unavailable', diagnostics=section.get('diagnostics', []),
                               part_count=0, hole_count=0, expected_omission=bool(spec.get('inject_failure') or spec['id'] == 'pole'))
                    if not row['diagnostics'] or not row['expected_omission']:
                        row['status'] = 'failed'
                    continue
                row.update(status='failed', reason='Omitted canonical map was resurrected')
                continue
            if spec.get('inject_failure'):
                row.update(status='failed', reason='Injected construction failure did not omit map')
                continue
            try:
                text = build_overlap_corridor_kml(section, index)
                metadata = section.get('visualization_metadata', {})
                paths, ranges = section_runs(analyzer.audit_pipelines, source, analyzer.segment_length)
                padding = metadata.get('padding_m') if metadata.get('policy') == 'qualified_path_buffer_v1' else None
                verified, parts, local_paths = inspect_document(text, paths, padding_m=padding)
                row.update(verified, qualified_runs=ranges, geometry_policy=metadata.get('policy', 'legacy'))
                if not verified['passed']:
                    row['status'] = 'failed'
                preview = output / f"{spec['id']}-{index:02d}-preview.kml"
                preview.write_text(text, encoding='utf-8')
                row['preview'] = preview.name
                document = ET.fromstring(text)
                source_document = ET.parse(source_path).getroot().find('k:Document', NS)
                target = document.find('k:Document', NS)
                for placemark in source_document:
                    target.append(placemark)
                overlay = output / f"{spec['id']}-{index:02d}-overlay.kml"
                ET.ElementTree(document).write(overlay, encoding='utf-8', xml_declaration=True)
                image = output / f"{spec['id']}-{index:02d}.svg"
                render_svg(image, parts, local_paths, f"{spec['id']} — {row['part_count']} parts, {row['hole_count']} holes")
                row['image'] = image.name
            except (ValueError, KeyError) as error:
                row.update(status='failed', reason=str(error))
    report = {'environment': environment(), 'cases': rows,
              'passed': all(row['status'] in ('ready', 'unavailable') for row in rows),
              'visual_review': 'pending', 'reference': 'Independent 2 m GRS80 paths; 128-quadrant buffer; all exterior/interior rings'}
    save_json(output / 'report.json', report)
    cards = []
    for row in rows:
        title = html.escape(f"{row['fixture']} / {row.get('section', '')}")
        image = f'<img src="{html.escape(row["image"])}" alt="{title}">' if row.get('image') else ''
        details = html.escape(json.dumps({k: row[k] for k in ('status', 'part_count', 'hole_count', 'padding_m',
                                      'boundary_hausdorff_m', 'reason', 'diagnostics') if k in row}, indent=2))
        cards.append(f'<article><h2>{title}</h2>{image}<pre>{details}</pre></article>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Corridor review gallery</title>'
        '<style>body{font:15px system-ui;background:#f4f6f9;color:#243047;margin:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}'
        'article{background:white;padding:18px;border-radius:12px}img{width:100%}h2{font-size:18px}pre{white-space:pre-wrap;font-size:12px}</style>'
        '<h1>Corridor review gallery</h1><p>Dark lines are independently extracted qualified source runs. '
        'Green polygons include every component and hole. Review overlays contain extra source lines; use preview KML for polygon-only inspection.</p><main>'
        + ''.join(cards) + '</main>', encoding='utf-8')
    (output / 'report.md').write_text('# Corridor gallery\n\nAll components and holes are inspected. Visual review remains pending.\n\n'
        '| Fixture | Section | Parts | Holes | Status |\n| --- | --- | --- | --- | --- |\n' +
        '\n'.join(f"| {row['fixture']} | {row.get('section', '')} | {row.get('part_count', '')} | {row.get('hole_count', '')} | {row['status']} |" for row in rows) + '\n', encoding='utf-8')
    print(f'{len(rows)} gallery rows; failures: {sum(row["status"] == "failed" for row in rows)}')
    return int(not report['passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
