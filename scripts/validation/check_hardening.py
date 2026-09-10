"""Compare mileage/group decisions with saved benchmarks and record GUI warnings."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.common import digest, environment, save_json
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import ExecutionContext

# These visual fields intentionally changed. All other fields, including every
# source total, savings value, section length and group membership, must match.
VISUAL_FIELDS = {'bbox', 'center_lon', 'center_lat', 'oriented_polygon',
                 'corridor_polygon', 'oriented_width_m', 'corridor_geometry_kind',
                 'corridor_approximation'}


def numerical(value, key=None):
    if key == 'parsed_kml_files':
        return [Path(p).name for p in value]
    if isinstance(value, dict):
        return {k: numerical(v, k) for k, v in value.items() if k not in VISUAL_FIELDS}
    if isinstance(value, list):
        return [numerical(v) for v in value]
    return value


class RecordingContext(ExecutionContext):
    def __init__(self):
        super().__init__(interactive=True)
        self.warnings = []

    def confirm_workload(self, message):
        self.warnings.append(message)
        self.workload_accepted = True


def run(baseline, output):
    old_report = json.loads((baseline / 'report.json').read_text())
    rows = []
    for case in old_report['cases']:
        name = case['id']
        source = baseline / {'short_paths': 'short.kml', 'linked': 'linked.kmz'}.get(name, name+'.kml')
        saved = json.loads((baseline / (name+'-1.json')).read_text())
        context = RecordingContext()
        start = time.perf_counter()
        result = PipelineAnalyzer(**saved['parameters']).analyze_complete(source, context=context)
        previously_limited = any('limit exceeded' in str(d).lower()
                                 for d in saved['result']['diagnostics'])
        row = {'fixture': name, 'same_input_hash': digest(source) == case['sha256'],
               'numerical_output_equal': numerical(result) == numerical(saved['result']),
               'previously_hit_hard_limit': previously_limited,
               'warning_matches_expectation': bool(context.warnings) == previously_limited,
               'warnings': context.warnings, 'single_run_seconds': time.perf_counter()-start}
        rows.append(row)
        print(name, row['numerical_output_equal'], len(row['warnings']), flush=True)
    save_json(output, {'environment': environment(), 'baseline_environment': old_report['environment'],
                       'excluded_visual_fields': sorted(VISUAL_FIELDS),
                       'normalization': 'parsed KML paths compared by filename', 'cases': rows})
    return int(any(not r['same_input_hash'] or not r['numerical_output_equal']
                   or not r['warning_matches_expectation'] for r in rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(run(args.baseline, args.output))
