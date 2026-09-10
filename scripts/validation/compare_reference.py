"""Compare sampled production output with explicitly bounded independent models."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.common import ROOT, environment, gallery_specs, geographic, parallel, save_json, write_fixture
sys.path.insert(0, str(ROOT / 'tests'))
from reference.intervals import common_axis_savings, pair_coverage, unique_pair_length
from pipeline_calculator.core.analyzer import PipelineAnalyzer


def distortion(paths, origin, bearing):
    from pyproj import Geod
    geod = Geod(ellps='GRS80')
    vertices = [p for parts in paths for part in parts for p in part]
    if any(math.hypot(*p) > 10000 for p in vertices):
        raise ValueError('Fixture outside the 10 km reference model extent')
    error, acceptable = 0.0, True
    for a,b in itertools.combinations(vertices,2):
        planar = math.dist(a,b)
        actual = geod.inv(*geographic(a, origin, bearing), *geographic(b, origin, bearing))[2]
        delta = abs(planar-actual)
        error = max(error,delta)
        acceptable &= delta <= 0.001+1e-5*planar
    return error, acceptable


def run(output):
    specs = gallery_specs() + [
        {'id': f'offset_{offset}', 'paths': parallel(shift=offset), 'minimum': 200}
        for offset in (0, 0.5, 1.25, 2.5, 50, 100.001, 300)]
    specs += [{'id': f'chain_{d}', 'paths': parallel((0,10,20)), 'minimum':200, 'distance':d}
              for d in (9.999,10,10.001,15,20)]
    specs += [{'id': f'minimum_{m}', 'paths':parallel(shift=50), 'minimum':m} for m in (249.999,250,250.001)]
    specs += [{'id':f'angle_{a}', 'paths':[[[(0,0),(0,300)]], [[(2,0),(2+300*math.sin(math.radians(a)),300*math.cos(math.radians(a)))]]],
               'minimum':10,'angle':15} for a in (14.999,15,15.001)]
    rows=[]
    for spec in specs:
        for reversed_paths, bearing in [(False, spec.get('bearing',0)), (True, spec.get('bearing',0)+37)]:
            paths = [[list(reversed(part)) if reversed_paths else part for part in parts] for parts in spec['paths']]
            origin=spec.get('origin',(-100,40)); distance=spec.get('distance',15); minimum=spec['minimum']
            max_distortion, supported = distortion(paths,origin,bearing)
            pair_sections=[pair_coverage(paths[i],paths[j],distance,minimum,spec.get('angle',15))
                           for i,j in itertools.combinations(range(len(paths)),2)]
            pair_length=sum(s['common_length'] for sections in pair_sections for s in sections)
            unique_length=sum(unique_pair_length(sections) for sections in pair_sections)
            straight = all(len(parts)==1 and len(parts[0])==2 and parts[0][0][0]==parts[0][1][0] for parts in paths)
            optimum = common_axis_savings([(p[0][0][0],p[0][0][1],p[0][1][1]) for p in paths], distance,minimum) if straight else None
            for step in (1,2,5,10,25):
                path=write_fixture(output / 'fixture.kml',paths,origin=origin,bearing=bearing)
                result=PipelineAnalyzer(segment_length=step,min_parallel_length=minimum,detection_range=distance,
                                        angular_tolerance=spec.get('angle',15)).analyze_complete(path)
                overlap=result.get('overlap_analysis') or {}
                measured=overlap.get('total_bundled_length',0)
                delta=measured-pair_length
                rows.append({'fixture':spec['id'],'segment_length':step,'reversed':reversed_paths,'bearing':bearing,
                    'origin':origin,'reference_pair_meters':pair_length,'production_pair_meters':measured,
                    'reference_unique_pair_meters':unique_length,
                    'pair_report_minus_unique_reference_m':measured-unique_length,
                    'signed_pair_error_m':delta,'absolute_pair_error_m':abs(delta),
                    'relative_pair_error':delta/pair_length if pair_length else None,
                    'max_projection_distortion_m':max_distortion,'projection_check':supported,
                    'reference_optimal_savings_m':optimum,'production_savings_m':overlap.get('savings_meters'),
                    'combined_savings_gap_m':None if optimum is None else optimum-overlap['savings_meters'],
                    'group_model':'common-axis exhaustive' if straight else 'pairwise only; no branch/group optimum',
                    'analysis_complete':result['analysis_complete']})
    save_json(output / 'report.json',{'environment':environment(),'cases':rows})
    worst=sorted(rows,key=lambda r:r['absolute_pair_error_m'],reverse=True)[:20]
    (output/'report.md').write_text('# Independent reference comparison\n\n'
        'This measures a declared planar model, not operational accuracy. Savings gap includes sampling and grouping; '
        'it is not solely heuristic loss. Pairwise lengths are not project savings.\n\n'
        '| Fixture | Step m | Pair error m | Projection error m |\n| --- | --- | --- | --- |\n'+
        '\n'.join(f"| {r['fixture']} | {r['segment_length']} | {r['signed_pair_error_m']:.6f} | {r['max_projection_distortion_m']:.6f} |" for r in worst)+'\n',encoding='utf-8')
    print(f'{len(rows)} comparisons; projection failures: {sum(not r["projection_check"] for r in rows)}')
    return int(any(not r['projection_check'] or not r['analysis_complete'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raise SystemExit(run(args.output))
