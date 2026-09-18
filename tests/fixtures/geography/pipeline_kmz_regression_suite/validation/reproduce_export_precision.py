"""Isolate export endpoints too close to a native border to certify strictly."""
from pathlib import Path
import sys
import tempfile

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.suite import BOUNDARIES, dump, digest
from pipeline_kmz_regression_suite.reference.geometry import GeometryReference, read_kmz
from pipeline_kmz_regression_suite.generator.generate import write_kmz


def main():
    source = next(s for s in read_kmz(SUITE / 'fixtures/03_parallel_corridors_crossing_borders.kmz')
                  if s['key'] == '03_short_a')
    destination = SUITE / 'validation/reproductions/03_export_endpoint_precision.kmz'
    write_kmz(destination, [source], 'Single-source transverse crossing: export precision')
    reference = GeometryReference(BOUNDARIES)
    fixed = reference.analyze(read_kmz(destination))
    # Application code is used only after fixing the independent input evidence.
    from pipeline_kmz_regression_suite.validation.compare_application import PipelineAnalyzer, AnalysisOptions, write_geography_kmz, exported_geometry
    result = PipelineAnalyzer().analyze_complete(destination, options=AnalysisOptions(True))
    evidence = []
    with tempfile.TemporaryDirectory(prefix='export-precision-') as tmp:
        for code in ('NM', 'TX'):
            exported = Path(tmp) / f'{code}.kmz'
            write_geography_kmz(result, exported, code)
            sources, _, _, meters = exported_geometry(exported)
            try:
                geometry = reference.analyze(sources)
                row = {'state': code, 'certified': True, 'intervals': geometry['intervals']}
            except ValueError as error:
                row = {'state': code, 'certified': False, 'reason': str(error)}
            row.update(exported_meters=meters, exported_sources=sources)
            evidence.append(row)
    dump(SUITE / 'validation/reproductions/export_precision.json', {
        'input': destination.relative_to(SUITE).as_posix(), 'sha256': digest(destination),
        'independent_input_original_meters': fixed['original_meters'],
        'independent_input_crossings': fixed['crossings'], 'export_checks': evidence,
        'interpretation': 'Historical strict-certificate diagnostic, not the export acceptance test. A failed strict ownership certificate is inconclusive about nanometer-scale incursion. The audited comparator instead checks each exported line against independently owned original source spans, the 1 cm cut target, 10 micrometer source-geodesic membership, and separate mileage conservation.'})
    print([(r['state'], r['certified']) for r in evidence])


if __name__ == '__main__':
    main()
