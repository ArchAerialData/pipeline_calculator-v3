"""Independent sample search/connectivity and bipartite upper-bound checks."""
import math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching
from pyproj import Transformer


def check_samples(analyzer):
    # Separate per-pipeline indexes, pyproj's coordinate transform, vectorized
    # geodesic predicates, and SciPy graph connectivity replace production's
    # global index, scalar search, and custom flood fill in this reference.
    pipes = analyzer.audit_pipelines
    convert = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
    caches = []
    for pipe in pipes:
        segments = pipe['segments']
        points = np.asarray([s['midpoint'] for s in segments]).reshape((-1, 2))
        x, y, z = convert.transform(points[:, 0], points[:, 1], np.zeros(len(points)))
        xyz = np.column_stack((x, y, z))
        caches.append((points, np.asarray([s['bearing'] for s in segments]), xyz, cKDTree(xyz)))
    expected, compared = {}, 0
    radius = math.hypot(analyzer.detection_range, analyzer.segment_length)+.001
    for p, (pa, ba, xa, ta) in enumerate(caches):
        if not len(pa):
            continue
        for q in range(p+1, len(caches)):
            pb, bb, xb, tb = caches[q]
            if not len(pb) or np.any(xa.min(axis=0)-xb.max(axis=0) > radius) or np.any(xb.min(axis=0)-xa.max(axis=0) > radius):
                continue
            compared += 1
            neighbors = ta.query_ball_tree(tb, radius)
            ia = np.repeat(np.arange(len(pa)), [len(ns) for ns in neighbors])
            ib = np.fromiter((j for ns in neighbors for j in ns), dtype=int)
            if not len(ia):
                continue
            angle = np.abs((ba[ia]-bb[ib]+90) % 180-90)
            ia, ib = ia[angle <= analyzer.angular_tolerance], ib[angle <= analyzer.angular_tolerance]
            if not len(ia):
                continue
            az, back, distance = analyzer.geod.inv(pa[ia, 0], pa[ia, 1], pb[ib, 0], pb[ib, 1])
            alpha, beta = np.radians(az-ba[ia]), np.radians(back-bb[ib])
            lateral = distance*np.maximum(abs(np.sin(alpha)), abs(np.sin(beta)))
            along = distance*np.maximum(abs(np.cos(alpha)), abs(np.cos(beta)))
            use = (lateral <= analyzer.detection_range) & (along < analyzer.segment_length-1e-8)
            found = set(zip(ia[use].tolist(), ib[use].tolist()))
            if found:
                expected[(p, q)] = found
    actual = {pair: {(m['pipeline_1_segment'], m['pipeline_2_segment']) for m in matches}
              for pair, matches in analyzer.audit_groups.items()}
    differences = []
    reference_sections = []
    for pair in expected.keys() | actual.keys():
        missing = expected.get(pair, set())-actual.get(pair, set())
        extra = actual.get(pair, set())-expected.get(pair, set())
        if missing or extra:
            differences.append({'pair': pair, 'missing': len(missing), 'extra': len(extra)})
        cells = sorted(expected.get(pair, set()))
        positions = {cell: i for i, cell in enumerate(cells)}
        rows, columns = [], []
        for (a, b), i in positions.items():
            sa, sb = pipes[pair[0]]['segments'][a], pipes[pair[1]]['segments'][b]
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    other = (a+da, b+db)
                    j = positions.get(other)
                    if j is not None and pipes[pair[0]]['segments'][other[0]]['path_index'] == sa['path_index'] and pipes[pair[1]]['segments'][other[1]]['path_index'] == sb['path_index']:
                        rows.append(i); columns.append(j)
        graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(cells), len(cells))).tocsr()
        count, labels = connected_components(graph, directed=False)
        groups = defaultdict(list)
        for cell, label in zip(cells, labels):
            groups[int(label)].append(cell)
        for group in groups.values():
            ids1, ids2 = {a for a, b in group}, {b for a, b in group}
            length = min(len(ids1), len(ids2))*analyzer.segment_length
            if length+1e-8 >= analyzer.min_parallel_length:
                reference_sections.append((pair, frozenset(ids1), frozenset(ids2)))
    actual_sections = {(s['pair'], frozenset(s['segment_ids'][0]), frozenset(s['segment_ids'][1])) for s in analyzer.audit_sections}
    assert not differences, differences
    assert set(reference_sections) == actual_sections
    # Upper bound: optimize each qualified pipeline pair independently. This
    # may double-count shared paths and is NOT an alternative savings total.
    edges = defaultdict(set)
    for section in analyzer.audit_sections:
        edges[section['pair']].update((m['pipeline_1_segment'], m['pipeline_2_segment']) for m in section['matches'])
    upper = 0
    for pair, group in edges.items():
        first = {value: i for i, value in enumerate(sorted({a for a, b in group}))}
        second = {value: i for i, value in enumerate(sorted({b for a, b in group}))}
        matrix = coo_matrix((np.ones(len(group)), ([first[a] for a, b in group], [second[b] for a, b in group])), shape=(len(first), len(second))).tocsr()
        upper += np.count_nonzero(maximum_bipartite_matching(matrix, perm_type='column') >= 0)*analyzer.segment_length
    # Independent, explicit partitions implement the documented nearest-first
    # clique policy without production union-find or its savings routine.
    distances = {}
    for section in analyzer.audit_sections:
        a, b = section['pair']
        for match in section['matches']:
            key = ((a, match['pipeline_1_segment']), (b, match['pipeline_2_segment']))
            distances[key] = min(distances.get(key, float('inf')), match['midpoint_distance'])
    def geometry(node):
        p, s = node
        segment = pipes[p]['segments'][s]
        return (*segment['midpoint'], segment['bearing'] % 180, pipes[p]['name'], segment['path_index'], s)
    ordered = sorted(distances, key=lambda edge: (distances[edge], sorted(geometry(n) for n in edge)))
    groups = {}
    for a, b in ordered:
        left, right = groups.get(a, frozenset([a])), groups.get(b, frozenset([b]))
        if left == right or {p for p, _ in left} & {p for p, _ in right}:
            continue
        if all(tuple(sorted((x, y))) in distances for x in left for y in right):
            merged = left | right
            for node in merged:
                groups[node] = merged
    partitions = set(groups.values())
    savings = sum((len(group)-1)*analyzer.segment_length for group in partitions)
    return {'pipeline_pairs_searched': compared, 'matching_sample_pairs': sum(map(len, expected.values())),
            'missing_pairs': 0, 'extra_pairs': 0, 'qualified_sections_exact': True,
            'independent_clique_savings_meters': savings, 'disjoint_survey_groups': len(partitions),
            'independent_pairwise_matching_upper_meters': float(upper)}
