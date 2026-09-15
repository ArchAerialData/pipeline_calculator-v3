"""Isolate the observed floating-point polygon containment residual, offline."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import zipfile

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.suite import BOUNDARIES, dump, digest
from pipeline_kmz_regression_suite.validation.compare_application import PipelineAnalyzer, AnalysisOptions, write_geography_kmz, exported_geometry
from pipeline_kmz_regression_suite.validation.application_contract import polygon_containment
from pipeline_kmz_regression_suite.reference.geometry import read_kmz, GeometryReference
from pipeline_kmz_regression_suite.generator.generate import write_kmz
from shapely import from_wkb
from pyproj import Geod


def main():
    original = SUITE / 'fixtures/03_parallel_corridors_crossing_borders.kmz'
    sources = read_kmz(original)
    with zipfile.ZipFile(BOUNDARIES) as z:
        boundary = from_wkb(z.read('states/TX.wkb'))
    found = []
    output = SUITE / 'validation/reproductions'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='polygon-residual-') as tmp:
        for motif in sorted({s['motif'] for s in sources}):
            group = [s for s in sources if s['motif'] == motif]
            input_path = Path(tmp) / 'input.kmz'
            write_kmz(input_path, group, motif)
            result = PipelineAnalyzer().analyze_complete(input_path, options=AnalysisOptions(True))
            if not any(s['state_code'] == 'TX' and s['interior_meters'] > 0 for s in result['geography']['states']):
                continue
            export_path = Path(tmp) / 'export.kmz'
            write_geography_kmz(result, export_path, 'TX')
            _, polygons, _, _ = exported_geometry(export_path)
            residuals = [p.difference(boundary) for p in polygons]
            total = sum(r.area for r in residuals)
            if total > 1e-18:
                destination = output / f'03_{motif}_polygon_residual.kmz'
                write_kmz(destination, group, motif)
                geo = GeometryReference(BOUNDARIES).analyze(read_kmz(destination))
                areas = [abs(Geod(ellps='GRS80').geometry_area_perimeter(r)[0]) for r in residuals if not r.is_empty]
                found.append({'motif': motif, 'file': destination.relative_to(SUITE).as_posix(),
                    'source_keys': [s['key'] for s in group], 'sha256': digest(destination),
                    'independent_original_meters': geo['original_meters'],
                    'independent_unresolved_meters': geo['unresolved_meters'],
                    'polygon_count': len(polygons), 'outside_area_degrees_squared': total,
                    'outside_area_square_meters_approximate': sum(areas),
                    'classification': 'Strict floating-point clipping diagnostic; evaluate acceptance with the local numerical boundary strip.',
                    'bounded_containment_checks': polygon_containment(polygons, boundary),
                    'bounded_containment_passed': all(row['passed'] for row in polygon_containment(polygons, boundary)),
                    'residual_bounds': [list(r.bounds) for r in residuals if not r.is_empty],
                    'strict_containment_assertion_passed': False})
    dump(output / 'polygon_residual.json', {'source_fixture_sha256': digest(original), 'cases': found})
    print(json.dumps(found, indent=2))


if __name__ == '__main__':
    main()
