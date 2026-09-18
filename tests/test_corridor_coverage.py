import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.corridor_coverage import MeasuredPath, covers_paths
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml
from scripts.validation.geometry import local_points, inside


def test_measured_path_retains_bends_duplicates_and_section_endpoints():
    geod = PipelineAnalyzer().geod
    a = (-100, 40)
    b = geod.fwd(*a, 0, 10)[:2]
    c = geod.fwd(*b, 90, 10)[:2]
    path = MeasuredPath(geod, [a, a, b, c])
    span = path.span(5, 15)
    assert b in span and a not in span and c not in span
    assert geod.inv(*a, *span[0])[2] == pytest.approx(5, abs=1e-6)
    assert geod.inv(*b, *span[-1])[2] == pytest.approx(5, abs=1e-6)
    assert path.span(0, 100)[-1] == c


def test_containment_checks_edges_not_just_vertices_and_bounds_work(monkeypatch):
    from pipeline_calculator.core import corridor_coverage
    # A notch: endpoints lie inside, but the segment between leaves the polygon.
    ring = [(0, 0), (10, 0), (10, 10), (6, 10), (6, 4), (4, 4), (4, 10), (0, 10), (0, 0)]
    assert covers_paths(ring, [[(1, 1), (9, 1)]])
    assert not covers_paths(ring, [[(2, 8), (8, 8)]])
    assert not covers_paths(ring, [[(20, 8)]])
    # Check both sweep orientations and the bounded fallback decision.
    assert covers_paths([(y, x*3) for x, y in ring], [[(1, 3), (1, 27)]])
    monkeypatch.setattr(corridor_coverage, 'MAX_COVERAGE_INSPECTIONS', 0)
    assert not covers_paths(ring, [[(1, 1), (9, 1)]])


@pytest.mark.parametrize('step', [4, 5, 10, 20, 50])
@pytest.mark.parametrize('reverse', [False, True])
def test_shifted_partner_endpoint_is_inside_serialized_corridor(step, reverse):
    analyzer = PipelineAnalyzer(segment_length=step, min_parallel_length=100)
    geod = analyzer.geod
    start = (-100, 40)
    other = geod.fwd(*start, 90, 2)[:2]
    other = geod.fwd(*other, 0, .5)[:2]
    a = [start, geod.fwd(*start, 0, 300)[:2]]
    b = [other, geod.fwd(*other, 0, 300+step)[:2]]
    pipelines = [{'name': 'A', 'coordinates': a}, {'name': 'B', 'coordinates': b[::-1] if reverse else b}]
    matches = analyzer.find_parallel_segments(pipelines)
    results = analyzer.calculate_overlap_results(pipelines, matches)
    from pipeline_calculator.core.bundling import qualifying_sections
    qualified = qualifying_sections(pipelines, matches, step, 100)[0]
    section = results['bundled_sections'][0]
    doc = ET.fromstring(build_overlap_corridor_kml(section, 1))
    points = doc.find('.//{*}Polygon//{*}coordinates').text
    ring = [tuple(map(float, token.split(',')[:2])) for token in points.split()]
    xy = local_points(ring, start)
    for p, ids in zip(pipelines, qualified['segment_ids']):
        path = MeasuredPath(geod, p['coordinates'])
        for point in path.span(min(ids)*step, (max(ids)+1)*step):
            assert inside(local_points([point], start)[0], xy)
    # Every sample size uses the same qualified-path buffer contract.
    assert section['visualization_kind'] == 'qualified_path_buffer'
    assert section['visualization_metadata']['padding_m'] == 5
    assert '5 m padding' in ' '.join(node.text or '' for node in doc.findall('.//{*}description'))


def test_parser_can_be_imported_first_in_fresh_interpreter():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c',
        'from pipeline_calculator.parsers.kml_kmz import extract_features_from_file; '
        'from pipeline_calculator.core import PipelineAnalyzer; assert PipelineAnalyzer().segment_length == 5'],
        env=dict(os.environ, PYTHONPATH=str(root/'src')), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.native_gui
def test_native_settings_feed_matching_backend_parameters():
    from pipeline_calculator.gui.main_window import PipelineCalculatorGUI
    from pipeline_calculator.gui.state import AnalysisParameters
    app = PipelineCalculatorGUI()
    try:
        app.root.update()
        assert app._get_parameters() == AnalysisParameters()
        for values in [(10, 100, 4, 5), (25, 500, 20, 30), (30, 100, 10, 30)]:
            for variable, value in zip((app.detection_range_var, app.min_parallel_var, app.segment_length_var, app.angular_tolerance_var), values):
                variable.set(str(value))
            params = app._get_parameters()
            assert tuple(params.as_dict().values()) == values
        for variable in (app.detection_range_var, app.min_parallel_var, app.segment_length_var, app.angular_tolerance_var):
            variable.set('0')
        assert tuple(app._get_parameters().as_dict().values()) == (1, 10, 1, 1)
        assert [v.get() for v in (app.detection_range_var, app.min_parallel_var, app.segment_length_var, app.angular_tolerance_var)] == ['1', '10', '1', '1']
    finally:
        app.close()
