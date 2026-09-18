"""Measure immutable full-state certification separately from map construction."""
from pathlib import Path
import json
import math
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import shapely
from shapely import get_num_coordinates
from shapely.geometry import Polygon
from pyproj import Geod
from pipeline_calculator.core.geography.boundaries import load_boundaries, polygon_parts
from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.options import AnalysisOptions

dataset = load_boundaries()
geod = Geod(ellps='GRS80')
budget = CorridorGeometryBudget()
rows = []


def inspect(code, shape, name):
    boundary = dataset.geometries[code]
    count = int(get_num_coordinates(boundary))
    assert int(get_num_coordinates(shape)) <= budget.max_native_vertices
    covers, differences = [], []
    for _ in range(3):
        budget.boundary_query(count)
        start = time.perf_counter()
        covered = boundary.covers(shape)
        covers.append(time.perf_counter() - start)
        # Run the strict difference even when covers succeeds. Production can
        # skip it; this benchmark must exercise the more costly fallback too.
        budget.boundary_query(count)
        start = time.perf_counter()
        difference = shape.difference(boundary)
        differences.append(time.perf_counter() - start)
    rows.append(dict(state=code, case=name, state_vertices=count,
                     candidate_vertices=int(get_num_coordinates(shape)), covered=bool(covered),
                     difference_empty=bool(difference.is_empty),
                     covers_seconds=covers, strict_difference_seconds=differences))


for code, boundary in sorted(dataset.geometries.items()):
    largest = max(polygon_parts(boundary), key=lambda polygon: polygon.area)
    vertex = tuple(largest.exterior.coords[0])
    interior = tuple(largest.representative_point().coords[0])
    for case, origin in (('native_boundary_vertex', vertex), ('interior', interior)):
        # 4095 vertices approaches the 4096 candidate cap. Longitude unwrapping
        # keeps a near-dateline ring local; benchmark canonical-zone candidate.
        points = [geod.fwd(*origin, i * 360 / 4094, 5)[:2] for i in range(4094)]
        points = [(lon + 360 * round((origin[0] - lon) / 360), lat) for lon, lat in points]
        shape = Polygon(points)
        assert shape.is_valid
        inspect(code, shape, case)

suite = ROOT / 'tests/fixtures/geography/pipeline_kmz_regression_suite/fixtures'
for stem in ('03_parallel_corridors_crossing_borders', '04_shared_border_and_near_border'):
    result = PipelineAnalyzer().analyze_complete(suite / f'{stem}.kmz', options=AnalysisOptions(state_breakdown=True))
    for state in result['geography']['states']:
        for index, section in enumerate((state.get('overlap_analysis') or {}).get('bundled_sections', [])):
            assert section['visualization_status'] == 'ready', section.get('diagnostics')
            for part, spec in enumerate(section['visualization_polygons']):
                shape = Polygon(spec['outer'], spec['holes'])
                inspect(state['state_code'], shape, f'{stem}/section-{index}/part-{part}')
                assert rows[-1]['difference_empty'], rows[-1]

maximum = max(value for row in rows for key in ('covers_seconds', 'strict_difference_seconds') for value in row[key])
assert maximum < 1., maximum
report = dict(python=platform.python_version(), shapely=shapely.__version__,
              boundary_source=dataset.boundary_source, state_count=len(dataset.geometries),
              boundary_vertex_ceiling=budget.max_boundary_vertices,
              query_ceiling=budget.max_boundary_queries, measured_queries=budget.boundary_queries,
              max_predicate_seconds=maximum, cases=rows,
              interpretation='Every measured call finished in less than 1 s; this is observed latency on the bundled resource and bounded candidates, not a hard OS scheduling guarantee.')
output = ROOT / '.validation-output/corridor-buffer/boundary-gates.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(dict(output=str(output), cases=len(rows), queries=budget.boundary_queries,
                     largest_state_vertices=max(row['state_vertices'] for row in rows), max_seconds=maximum)))
