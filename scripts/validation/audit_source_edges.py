"""Exploratory source-edge cross-check, independent of analysis segmentation.

Local tangent/flat-end overlap of original geodesic edges is a comparison model,
not survey ground truth or an alternative multi-pipeline savings calculation.
Comparison cells connect when they touch along either path; gaps along the other
path can remain. Investigate apparent omissions before calling them defects.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np
from pyproj import Geod, Transformer
from pipeline_calculator.parsers.kml_kmz import extract_features_from_file_with_diagnostics


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]+.02:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def length(intervals):
    return sum(b-a for a, b in union(intervals))


def reference(path, output, audit_output):
    import shapely
    geod = Geod(ellps='GRS80')
    pipes = extract_features_from_file_with_diagnostics(path).pipelines
    project = Transformer.from_crs('EPSG:4326', '+proj=aeqd +lat_0=31 +lon_0=-100 +ellps=GRS80 +units=m', always_xy=True)
    endpoints, meta = [], []
    for p, pipe in enumerate(pipes):
        for part, coords in enumerate(pipe['coordinate_paths']):
            chainage = 0
            valid_index = 0
            for i, (a, b) in enumerate(zip(coords, coords[1:])):
                az, _, meters = geod.inv(*a, *b)
                if meters > 1e-7:
                    endpoints.append((a, b))
                    meta.append((p, part, valid_index, chainage, meters, az))
                    valid_index += 1
                chainage += meters
    endpoints = np.asarray(endpoints)
    meta = np.asarray(meta)
    x, y = project.transform(endpoints[:, :, 0], endpoints[:, :, 1])
    lines = shapely.linestrings(np.stack((x, y), axis=-1))
    tree = shapely.STRtree(lines)
    candidates = []
    for first in range(0, len(lines), 10000):
        a, b = tree.query(lines[first:first+10000], predicate='dwithin', distance=32)
        a = a+first
        use = (a < b) & (meta[a, 0] != meta[b, 0])
        candidates.extend(zip(a[use].tolist(), b[use].tolist()))
    a, b = np.array(candidates).T
    # Project each candidate's endpoints about its first edge's own origin and
    # direction; the broad project-wide projection is only a padded prefilter.
    az0, _, d0 = geod.inv(endpoints[a, 0, 0], endpoints[a, 0, 1], endpoints[b, 0, 0], endpoints[b, 0, 1])
    az1, _, d1 = geod.inv(endpoints[a, 0, 0], endpoints[a, 0, 1], endpoints[b, 1, 0], endpoints[b, 1, 1])
    t0, s0 = d0*np.cos(np.radians(az0-meta[a, 5])), d0*np.sin(np.radians(az0-meta[a, 5]))
    t1, s1 = d1*np.cos(np.radians(az1-meta[a, 5])), d1*np.sin(np.radians(az1-meta[a, 5]))
    dx, dy = t1-t0, s1-s0
    squared = dx*dx+dy*dy
    angles = np.abs((np.degrees(np.arctan2(dy, dx))+90) % 180-90)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, radius, minimum, angle in [('default', 15, 200, 15), ('range10', 10, 200, 15), ('range25', 25, 200, 15), ('minimum100', 15, 100, 15), ('minimum500', 15, 500, 15), ('angle5', 15, 200, 5), ('angle30', 15, 200, 30)]:
        groups = defaultdict(dict)
        for k in np.flatnonzero(angles <= angle):
            ia, ib = a[k], b[k]
            ma, mb = meta[ia], meta[ib]
            if abs(dx[k]) < 1e-10:
                continue
            ends = sorted((t0[k]+s0[k]*dy[k]/dx[k], t0[k]+(squared[k]+s0[k]*dy[k])/dx[k]))
            low, high = max(0, ends[0]), min(ma[4], ends[1])
            if abs(dy[k]) < 1e-10:
                if abs(s0[k]) > radius:
                    continue
            else:
                center = t0[k]-dx[k]*s0[k]/dy[k]
                extent = radius*math.sqrt(squared[k])/abs(dy[k])
                low, high = max(low, center-extent), min(high, center+extent)
            if high-low <= 1e-5:
                continue
            u0 = ((low-t0[k])*dx[k]-s0[k]*dy[k])/squared[k]
            u1 = ((high-t0[k])*dx[k]-s0[k]*dy[k])/squared[k]
            aa = (ma[3]+low, ma[3]+high)
            bb = (mb[3]+min(u0, u1)*mb[4], mb[3]+max(u0, u1)*mb[4])
            groups[(int(ma[0]), int(mb[0]), int(ma[1]), int(mb[1]))][(int(ma[2]), int(mb[2]))] = (aa, bb)
        qualified = []
        for pair, cells in groups.items():
            remaining = set(cells)
            while remaining:
                seed = remaining.pop()
                stack, component = [seed], []
                while stack:
                    cell = stack.pop()
                    component.append(cells[cell])
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            neighbor = (cell[0]+da, cell[1]+db)
                            if neighbor in remaining and any(max(x[0], y[0]) <= min(x[1], y[1])+.02 for x, y in zip(cells[cell], cells[neighbor])):
                                remaining.remove(neighbor)
                                stack.append(neighbor)
                spans = [union([part[i] for part in component]) for i in range(2)]
                meters = min(length(spans[0]), length(spans[1]))
                if meters >= minimum:
                    qualified.append({'pair_indices': list(pair), 'pair': [pipes[pair[0]]['name'], pipes[pair[1]]['name']], 'length_m': meters, 'spans': spans})
        # Identify comparison groups absent from the app's qualified ranges.
        # These can bridge a gap on one path; they require further investigation.
        actual = json.loads((audit_output/case/'corridor-checks.json').read_text(encoding='utf-8'))
        missed = []
        for section in qualified:
            overlaps = []
            for row in actual:
                if row['pair'] == section['pair']:
                    for ref_span in section['spans'][0]:
                        overlaps.append((max(ref_span[0], row['ranges'][0]['start_m']), min(ref_span[1], row['ranges'][0]['end_m'])))
            covered = length([(lo, hi) for lo, hi in overlaps if hi > lo])
            if covered < .01:
                missed.append(section)
        row = {'case': case, 'original_edges': len(lines), 'candidate_edge_pairs': len(candidates),
               'continuous_sections': len(qualified), 'continuous_pairwise_miles': sum(s['length_m'] for s in qualified)/1609.347218694,
               'entire_reference_sections_absent': missed,
               'model': 'Original-edge local tangent geometry, flat endpoints; adjacent cells connect within 2 cm on either path and may bridge a gap on the other. Investigate omissions. Pairwise comparison, not unique mileage removed.'}
        (output/f'{case}.json').write_text(json.dumps({'summary': row, 'sections': qualified}, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
        rows.append(row)
    (output/'matrix.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    reference(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
