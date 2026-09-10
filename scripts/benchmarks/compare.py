"""Compare measured runs and full numerical output, excluding stated new metadata."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.validation.common import save_json

ADDITIVE={'corridor_geometry_kind','corridor_approximation','source_path_indices'}


def calculation(value,key=None):
    if key=='parsed_kml_files':
        return [Path(path).name for path in value]
    if isinstance(value,dict):
        return {k:calculation(v,k) for k,v in value.items() if k not in ADDITIVE}
    if isinstance(value,list):
        return [calculation(v) for v in value]
    return value


def run(before,after,disabled,output):
    a=json.loads((before/'report.json').read_text());b=json.loads((after/'report.json').read_text())
    original={c['id']:c for c in a['cases']};rows=[]
    for current in b['cases']:
        name=current['id'];base=original[name]
        old=json.loads((before/f'{name}-1.json').read_text())['result']
        new=json.loads((after/f'{name}-1.json').read_text())['result']
        rows.append({'fixture':name,'same_hash':base['sha256']==current['sha256'],
            'calculation_equal':calculation(old)==calculation(new),
            'baseline_seconds':base['median_seconds'],'final_seconds':current['median_seconds'],
            'change_percent':100*(current['median_seconds']/base['median_seconds']-1),
            'baseline_peak_rss_bytes':max(r['memory']['bytes'] for r in base['runs']),
            'final_peak_rss_bytes':max(r['memory']['bytes'] for r in current['runs'])})
    disabled_report=json.loads((disabled/'report.json').read_text())['cases'][0]
    enabled=next(c for c in b['cases'] if c['id']=='dense')
    assert enabled['sha256']==disabled_report['sha256']
    overhead=100*(enabled['median_seconds']/disabled_report['median_seconds']-1)
    report={'cases':rows,'dense_context_overhead_percent':overhead,
            'excluded_additive_fields':sorted(ADDITIVE),'normalization':'parsed KML source paths compared by filename',
            'baseline_source_sha256':a['environment']['source_sha256'],
            'measured_source_sha256':b['environment']['source_sha256']}
    save_json(output/'comparison.json',report)
    (output/'comparison.md').write_text('# Performance and output comparison\n\n'
        '| Fixture | Before s | After s | Change | Peak RSS before/after MiB | Output equal |\n| --- | --- | --- | --- | --- | --- |\n'+
        '\n'.join(f"| {r['fixture']} | {r['baseline_seconds']:.3f} | {r['final_seconds']:.3f} | {r['change_percent']:+.1f}% | {r['baseline_peak_rss_bytes']/2**20:.1f} / {r['final_peak_rss_bytes']/2**20:.1f} | {r['calculation_equal']} |" for r in rows)+
        f'\n\nDense workload context/progress overhead vs the final disabled mode: **{overhead:.2f}%**. '
        'Five timed runs per workload after warm-up; profiler runs excluded. Subsecond changes are noisy. '
        'Output comparisons exclude only three additive corridor metadata fields and normalize input directory names.\n',encoding='utf-8')
    print(f'{len(rows)} comparisons; equal: {sum(r["calculation_equal"] for r in rows)}; context overhead {overhead:.2f}%')
    return int(any(not r['same_hash'] or not r['calculation_equal'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','disabled','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raise SystemExit(run(args.before,args.after,args.disabled,args.output))
