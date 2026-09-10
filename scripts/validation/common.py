"""Deterministic local fixtures and evidence metadata (never shipped in the app)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    source = hashlib.sha256()
    for p in sorted((ROOT / 'src').rglob('*.py')):
        source.update(p.relative_to(ROOT).as_posix().encode())
        source.update(p.read_bytes())
    return {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        'source_sha256': source.hexdigest(),
        'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
        'packages': {p: importlib.metadata.version(p) for p in
                     ('numpy', 'scipy', 'pyproj', 'pytest', 'customtkinter', 'pyinstaller')},
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def geographic(point, origin=(-100, 40), bearing=0):
    """GRS80 azimuthal equidistant placement of metric fixture coordinates."""
    from pyproj import Geod
    x, y = point
    return Geod(ellps='GRS80').fwd(*origin, math.degrees(math.atan2(x, y)) + bearing,
                                 math.hypot(x, y))[:2]


def parallel(offsets=(0, 2), length=300, shift=0):
    return [[[(x, shift if i else 0), (x, length + (shift if i else 0))]]
            for i, x in enumerate(offsets)]


def gallery_specs():
    shapes = {
        'straight': [(0, 0), (0, 300)],
        'curve': [(0, 0), (0, 100), (20, 200), (60, 300)],
        'right_angle': [(0, 0), (0, 150), (150, 150)],
        'hairpin': [(0, 0), (0, 200), (8, 200), (8, 0)],
        'loop': [(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)],
        'short': [(0, 0), (0, 20)],
    }
    specs = []
    for name, shape in shapes.items():
        specs.append({'id': name, 'paths': [[shape], [[(x + 2, y) for x, y in shape]]],
                      'minimum': 10})
    specs.extend([
        {'id': 'multipart', 'paths': [[[(0, 0), (0, 150)], [(0, 400), (0, 550)]],
                                      [[(2, 0), (2, 150)], [(2, 400), (2, 550)]]], 'minimum': 10},
        {'id': 'reversed', 'paths': [[[(0, 0), (0, 300)]], [[(2, 300), (2, 0)]]], 'minimum': 10},
        {'id': 'dateline', 'paths': parallel(), 'origin': (179.999, 60), 'bearing': 90, 'minimum': 10},
        {'id': 'pole', 'paths': parallel(), 'origin': (0, 89.99), 'bearing': 90, 'minimum': 10},
    ])
    return specs


def kml_text(paths, origin=(-100, 40), bearing=0, track=False):
    placemarks = []
    for i, parts in enumerate(paths):
        geometries = []
        for part in parts:
            coords = [geographic(p, origin, bearing) for p in part]
            if track:
                geometries.append('<gx:Track>' + ''.join(
                    f'<gx:coord>{x:.12f} {y:.12f} 0</gx:coord>' for x, y in coords) + '</gx:Track>')
            else:
                text = ' '.join(f'{x:.12f},{y:.12f},0' for x, y in coords)
                geometries.append(f'<LineString><coordinates>{text}</coordinates></LineString>')
        placemarks.append(f'<Placemark><name>{escape(str(i))}</name><MultiGeometry>' +
                          ''.join(geometries) + '</MultiGeometry></Placemark>')
    return ('<kml xmlns="http://www.opengis.net/kml/2.2" '
            'xmlns:gx="http://www.google.com/kml/ext/2.2"><Document>' +
            ''.join(placemarks) + '</Document></kml>')


def write_fixture(path, paths, *, linked=0, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = kml_text(paths, **kwargs)
    if linked:
        # Fixed metadata makes the KMZ byte-for-byte reproducible.
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            root = '<kml><Document>' + ''.join(
                f'<NetworkLink><Link><href>part{i}.kml</href></Link></NetworkLink>' for i in range(linked)) + '</Document></kml>'
            for name, content in [('doc.kml', root)] + [(f'part{i}.kml', text) for i in range(linked)]:
                info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content.encode('utf-8'))
    else:
        path.write_text(text, encoding='utf-8')
    return path


def metrics(result):
    overlap = result.get('overlap_analysis') or {}
    return {'complete': result['analysis_complete'], 'pipelines': len(result['pipelines']),
            'original_meters': result['total_meters'], 'savings_meters': overlap.get('savings_meters'),
            'sections': len(overlap.get('bundled_sections', [])),
            'diagnostics': sorted(d['code'] for d in result['diagnostics'])}
