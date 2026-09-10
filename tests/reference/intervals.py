"""Bounded planar reference. Only the Python standard library is used here."""
from __future__ import annotations

from functools import lru_cache
import itertools
import math

EPS = 1e-7


def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi - lo <= EPS:
            continue
        if result and lo <= result[-1][1] + EPS:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def length(intervals):
    return sum(hi - lo for lo, hi in union(intervals))


def intersection(left, right):
    return union((max(a, c), min(b, d)) for a, b in left for c, d in right)


def subtract(intervals, removed):
    result = union(intervals)
    for lo, hi in union(removed):
        updated = []
        for a, b in result:
            if hi <= a or lo >= b:
                updated.append((a, b))
            else:
                if lo > a:
                    updated.append((a, lo))
                if hi < b:
                    updated.append((hi, b))
        result = updated
    return result


def edges(path):
    position = 0.0
    for a, b in zip(path, path[1:]):
        distance = math.dist(a, b)
        if distance > EPS:
            yield a, ((b[0]-a[0])/distance, (b[1]-a[1])/distance), distance, position
            position += distance


def clip_linear(interval, value, slope, minimum, maximum):
    lo, hi = interval
    if abs(slope) < 1e-12:
        return (lo, hi) if minimum-EPS <= value <= maximum+EPS else None
    start, end = sorted(((minimum-value)/slope, (maximum-value)/slope))
    lo, hi = max(lo, start), min(hi, end)
    return (lo, hi) if hi-lo > EPS else None


def covered_on_edge(first, second, distance):
    a, u, size, position = first
    b, v, other_size, _ = second
    dx, dy = a[0]-b[0], a[1]-b[1]
    interval = clip_linear((0, size), dx*v[0]+dy*v[1], u[0]*v[0]+u[1]*v[1], 0, other_size)
    if interval is not None:
        interval = clip_linear(interval, dx*v[1]-dy*v[0], u[0]*v[1]-u[1]*v[0], -distance, distance)
    return None if interval is None else (position+interval[0], position+interval[1])


def pair_coverage(parts_a, parts_b, distance=15, minimum=200, angle=15):
    """Continuous bilateral edge-strip coverage, isolated by original part identity.

    Correspondences connect only if their intervals touch on BOTH original paths.
    This declared planar model is not a geodesic flight-route optimization.
    """
    sections = []
    for ai, a in enumerate(parts_a):
        for bi, b in enumerate(parts_b):
            cells = []
            for edge_a, edge_b in itertools.product(list(edges(a)), list(edges(b))):
                u, v = edge_a[1], edge_b[1]
                dot = min(1, max(-1, abs(u[0]*v[0]+u[1]*v[1])))
                if math.degrees(math.acos(dot)) > angle+1e-9:
                    continue
                ia, ib = covered_on_edge(edge_a, edge_b, distance), covered_on_edge(edge_b, edge_a, distance)
                if ia is not None and ib is not None:
                    cells.append((ia, ib))
            if len(cells) > 4096:
                raise ValueError('Reference pair exceeds the small-fixture correspondence budget')
            remaining = set(range(len(cells)))
            while remaining:
                stack = [remaining.pop()]
                component = []
                while stack:
                    index = stack.pop()
                    component.append(cells[index])
                    for other in sorted(remaining):
                        if all(max(cells[index][axis][0], cells[other][axis][0]) <=
                               min(cells[index][axis][1], cells[other][axis][1])+EPS for axis in (0, 1)):
                            remaining.remove(other)
                            stack.append(other)
                left, right = union(c[0] for c in component), union(c[1] for c in component)
                la, lb = length(left), length(right)
                if min(la, lb)+EPS >= minimum:
                    sections.append({'parts': (ai, bi), 'intervals_a': left, 'intervals_b': right,
                                     'length_a': la, 'length_b': lb, 'common_length': min(la, lb)})
    return sections


def common_axis_savings(lines, distance=15, minimum=200):
    """Exhaustive disjoint clique partitions on y-slabs for <=6 straight lines.

    Each line is (x, start_y, end_y). The objective minimizes passes per slab.
    """
    if len(lines) > 6:
        raise ValueError('Exhaustive reference is limited to six pipelines')
    normalized = [(x, min(a, b), max(a, b)) for x, a, b in lines]
    pairs = {}
    for i, j in itertools.combinations(range(len(lines)), 2):
        x, a, b = normalized[i]; y, c, d = normalized[j]
        lo, hi = max(a, c), min(b, d)
        if abs(x-y) <= distance+EPS and hi-lo > EPS and hi-lo+EPS >= minimum:
            pairs[i, j] = (lo, hi)
    boundaries = sorted({p for _, a, b in normalized for p in (a, b)})
    savings = 0.0
    for a, b in zip(boundaries, boundaries[1:]):
        mid = (a+b)/2
        active = tuple(i for i, (_, lo, hi) in enumerate(normalized) if lo < mid < hi)
        @lru_cache(None)
        def passes(nodes):
            if not nodes:
                return 0
            first, rest = nodes[0], nodes[1:]
            best = len(nodes)
            for size in range(len(rest)+1):
                for tail in itertools.combinations(rest, size):
                    group = (first,)+tail
                    if all((i,j) in pairs and pairs[i,j][0] <= mid <= pairs[i,j][1]
                           for i,j in itertools.combinations(sorted(group),2)):
                        best = min(best, 1+passes(tuple(i for i in rest if i not in tail)))
            return best
        savings += (len(active)-passes(active))*(b-a)
    return savings


def unique_pair_length(sections):
    """Union coverage across qualified sections too, avoiding repeated branch matches."""
    left, right = {}, {}
    for section in sections:
        a,b=section['parts']
        left.setdefault(a,[]).extend(section['intervals_a'])
        right.setdefault(b,[]).extend(section['intervals_b'])
    return min(sum(length(parts) for parts in left.values()), sum(length(parts) for parts in right.values()))
