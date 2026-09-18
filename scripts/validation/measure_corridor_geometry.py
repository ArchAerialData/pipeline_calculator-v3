"""Local reproducible producer accuracy/resource/cancellation evidence."""
from pathlib import Path
import collections
import ctypes
import json
import math
import platform
import random
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_corridor_buffer import GEOD, geographic, run, projected
from pipeline_calculator.core import corridor_buffer as implementation
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from shapely.geometry.base import BaseGeometry
import shapely

native = collections.defaultdict(list)


def timed(name, operation):
    def call(*args, **kwargs):
        start = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        finally:
            native[name].append(time.perf_counter() - start)
    return call


for name in ('buffer', 'difference', 'intersection', 'covers'):
    setattr(BaseGeometry, name, timed(name, getattr(BaseGeometry, name)))
validity = BaseGeometry.is_valid
BaseGeometry.is_valid = property(timed('is_valid', lambda shape: validity.fget(shape)))
implementation.unary_union = timed('unary_union', implementation.unary_union)

cases = [
    ('capsule', geographic([(0, 0), (0, 300)])),
    ('L', geographic([(0, 0), (0, 300), (300, 300)])),
    ('U', geographic([(0, 0), (0, 300), (300, 300), (300, 0)])),
    ('closed_loop', geographic([(0, 0), (300, 0), (300, 300), (0, 300), (0, 0)])),
    ('sparse_100km', geographic([(0, 0), (0, 100000)])),
    ('dense_redundant_5m', geographic([(0, p) for p in range(0, 1001, 5)])),
    ('dateline', geographic([(-300, 0), (300, 0)], (180, 55))),
]
for count in (20, 50, 100):
    points = []
    for index in range(count):
        points.extend([(0 if index % 2 == 0 else 1000, index * 20),
                       (1000 if index % 2 == 0 else 0, index * 20)])
    cases.append((f'serpentine_{count}', geographic(points)))
for seed in range(40):
    rng = random.Random(seed)
    points = [(-100, 40)]
    for _ in range(9):
        points.append(GEOD.fwd(*points[-1], rng.uniform(0, 360), rng.uniform(10, 300))[:2])
    cases.append((f'spider_{seed}', points))

results = []
for name, coordinates in cases:
    budget = implementation.CorridorGeometryBudget()
    started = time.perf_counter()
    candidate = implementation.build_buffered_corridor([run(coordinates)], geod=GEOD, budget=budget)
    row = dict(name=name, seconds=time.perf_counter() - started, status=candidate['visualization_status'],
               counts=candidate['visualization_metadata'], budget=vars(budget), diagnostics=candidate['diagnostics'])
    row['counts'].pop('source_runs')
    if name == 'capsule' and candidate['visualization_status'] == 'ready':
        shape = projected(candidate)
        row['analytic_area_error_m2'] = abs(shape.area - (3000 + math.pi * 25))
        row['analytic_bounds_max_error_m'] = max(abs(a - b) for a, b in zip(shape.bounds, (-5, -5, 5, 305)))
    results.append(row)

cancellations = []
for _ in range(3):
    context = ExecutionContext()
    event_time = []
    def cancel():
        event_time.append(time.perf_counter())
        context.cancel()
    timer = threading.Timer(.03, cancel)
    timer.start()
    try:
        implementation.build_buffered_corridor([run(geographic([(0, 0), (0, 100000)]))], geod=GEOD, context=context)
        raise AssertionError('Cancellation did not interrupt the build')
    except AnalysisCancelled:
        cancellations.append(time.perf_counter() - event_time[0])
    finally:
        timer.cancel()

class Memory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                *[(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                   'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                   'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]

memory = Memory()
memory.cb = ctypes.sizeof(memory)
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory), ctypes.c_ulong]
assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(memory), memory.cb)
report = dict(python=platform.python_version(), shapely=shapely.__version__, cases=results,
              native_calls={name: dict(count=len(values), max_seconds=max(values),
                                      median_seconds=statistics.median(values)) for name, values in native.items()},
              cancellation_seconds=cancellations, peak_working_set_bytes=memory.PeakWorkingSetSize,
              notes=['Measurements are for the isolated corridor producer, not total application memory or end-to-end performance.',
                     'Conservative native complexity limits may omit stress shapes; normal reference cases must remain ready.'])
assert max(cancellations) < 1
assert max(max(values) for values in native.values()) < 1
assert all(row['status'] == 'ready' for row in results if not row['name'].startswith('serpentine'))
output = ROOT / '.validation-output/corridor-buffer/geometry-gates.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(dict(output=str(output), statuses=dict(collections.Counter(row['status'] for row in results)),
                     max_native_seconds=max(max(values) for values in native.values()),
                     max_cancel_seconds=max(cancellations), peak_working_set_bytes=memory.PeakWorkingSetSize)))
