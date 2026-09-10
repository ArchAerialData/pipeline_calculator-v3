"""Reproducible bounded subprocess benchmarks; profiles are separate from timings."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.common import digest, environment, metrics, parallel, save_json, write_fixture
from scripts.validation.processes import run_bounded


def peak_memory():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {'bytes': counters.PeakWorkingSetSize, 'collector': 'GetProcessMemoryInfo.PeakWorkingSetSize'}
        return {'bytes': None, 'collector': 'unavailable'}
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'bytes': value if sys.platform == 'darwin' else value * 1024, 'collector': 'getrusage.ru_maxrss'}


def worker(args):
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core import overlap
    params = json.loads(args.parameters)
    analyzer = PipelineAnalyzer(**params)
    counts = {}
    stages = {}
    if args.profile:
        # Count candidates only in profiled runs, never pollute unprofiled timing.
        original_tree = overlap.KDTree
        class CountingTree:
            def __init__(self, points):
                counts['segments'] = len(points)
                self.tree = original_tree(points)
            def query_ball_point(self, *a, **kw):
                result = self.tree.query_ball_point(*a, **kw)
                counts['candidate_inspections'] = counts.get('candidate_inspections', 0) + len(result)
                return result
        overlap.KDTree = CountingTree
        for name in ('calculate_pipeline_lengths', 'find_parallel_segments', 'calculate_overlap_results'):
            original = getattr(analyzer, name)
            def timed(*a, _method=original, _name=name, **kw):
                start = time.perf_counter()
                result = _method(*a, **kw)
                stages[_name] = time.perf_counter() - start
                if _name == 'find_parallel_segments':
                    counts['accepted_unique_matches'] = sum(map(len, result.values()))
                return result
            setattr(analyzer, name, timed)
    kwargs = {}
    if args.context:
        from pipeline_calculator.core.execution import ExecutionContext
        kwargs['context'] = ExecutionContext()
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    start = time.perf_counter()
    result = analyzer.analyze_complete(args.input, **kwargs)
    elapsed = time.perf_counter() - start
    if profiler:
        profiler.disable()
        profiler.dump_stats(args.profile)
    save_json(args.output, {'seconds': elapsed, 'memory': peak_memory(), 'metrics': metrics(result),
                           'counts': counts, 'combined_stage_seconds': stages,
                           'result': result, 'parameters': params})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['smoke', 'medium', 'stress'], default='smoke')
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--seed', type=int, default=0, help='Recorded fixture identity; geometry is deterministic')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=str)
    p.add_argument('--parameters', default='{}')
    p.add_argument('--context', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--profile', type=str, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return 0
    if args.repeat < 1 or args.timeout <= 0:
        p.error('repeat and timeout must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    segments = {'smoke': 1000, 'medium': 20000, 'stress': 100000}[args.suite]
    length = segments * 5 / 2
    cases = [('custom', Path(args.input))] if args.input else [
        ('sparse', write_fixture(args.output / 'sparse.kml', parallel((0, 100), length))),
        ('dense', write_fixture(args.output / 'dense.kml', parallel((0, 2, 4, 6), length / 2))),
        ('chain', write_fixture(args.output / 'chain.kml', parallel((0, 10, 20), length / 1.5))),
        ('short_paths', write_fixture(args.output / 'short.kml', [
            [[(i * 100, 0), (i * 100, 50)]] for i in range(min(segments // 10, 1000))])),
        ('linked', write_fixture(args.output / 'linked.kmz', parallel(length=100), linked=min(segments // 40, 200))),
        ('curved', write_fixture(args.output / 'curved.kml', [
            [[(x + offset, y) for x, y in [(0, 0), (0, length / 3), (100, 2 * length / 3), (50, length)]]]
            for offset in (0, 2)])),
    ]
    report = {'environment': environment(), 'suite': args.suite, 'seed': args.seed,
              'context_enabled': args.context, 'cases': []}
    for name, path in cases:
        row = {'id': name, 'sha256': digest(path), 'runs': [], 'status': 'complete'}
        report['cases'].append(row)
        for i in range(args.repeat + 2):
            dest = args.output / f'{name}-{i}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--input', str(path.resolve()),
                       '--output', str(dest.resolve()), '--parameters', args.parameters]
            if args.context:
                command.append('--context')
            if i == args.repeat + 1:
                command += ['--profile', str((args.output / f'{name}.prof').resolve())]
            try:
                process = run_bounded(command, timeout=args.timeout)
                if process.returncode:
                    row.update(status='failed', error=process.stderr[-2000:])
                    break
            except subprocess.TimeoutExpired:
                row.update(status='timeout')
                break
            data = json.loads(dest.read_text())
            if 0 < i <= args.repeat:
                row['runs'].append({k: data[k] for k in ('seconds', 'memory', 'metrics')})
            elif i == args.repeat + 1:
                row['profile_counts'] = data['counts']
                row['profile_stages'] = data['combined_stage_seconds']
        if row['runs']:
            times = [r['seconds'] for r in row['runs']]
            row.update(median_seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times))
        print(name, row['status'], row.get('median_seconds'), flush=True)
        save_json(args.output / 'report.json', report)
    (args.output / 'report.md').write_text('# Benchmark results\n\n' + '\n'.join(
        f"- {r['id']}: {r['status']}; median {r.get('median_seconds', 'N/A')} s" for r in report['cases']) +
        '\n\nTimes exclude imports and profiler instrumentation. Peak RSS includes the worker runtime.\n', encoding='utf-8')
    return int(any(r['status'] != 'complete' for r in report['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
