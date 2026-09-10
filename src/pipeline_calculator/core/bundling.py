"""Qualify continuous segment coverage once for both reporting and savings."""
from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

from collections import defaultdict


def qualifying_sections(pipelines, parallel_groups, segment_length, min_parallel_length, *, context=None):
    """Return connected matches, measuring each covered segment only once.

    Matches are cells in the two paths' segment-index grid. Adjacent cells belong
    to one section, regardless of neighbor enumeration or digitization direction.
    Missing segment rows/columns and distinct coordinate paths split sections.
    """
    if context is not None:
        context.report("Qualifying sections", 0, len(parallel_groups))
    qualified = []
    for pair_index, (pair, matches) in enumerate(sorted(parallel_groups.items())):
        if context is not None and pair_index % 256 == 0:
            context.check()
        if context is not None:
            context.report("Qualifying sections", pair_index, len(parallel_groups))
        p1, p2 = pair
        by_paths = defaultdict(dict)
        for match_position, match in enumerate(matches):
            if context is not None and match_position % 256 == 0:
                context.check()
            i, j = match["pipeline_1_segment"], match["pipeline_2_segment"]
            s1, s2 = pipelines[p1]["segments"][i], pipelines[p2]["segments"][j]
            paths = (s1.get("path_index", 0), s2.get("path_index", 0))
            cell = (s1.get("path_segment_index", i), s2.get("path_segment_index", j))
            old = by_paths[paths].get(cell)
            if old is None or match["distance"] < old["distance"]:
                by_paths[paths][cell] = match
        for path_pair_position, (paths, cells) in enumerate(sorted(by_paths.items())):
            if context is not None and path_pair_position % 256 == 0:
                context.check()
            remaining = set(cells)
            for seed_position, seed in enumerate(sorted(cells)):
                if context is not None and seed_position % 256 == 0:
                    context.check()
                if seed not in remaining:
                    continue
                remaining.remove(seed)
                stack, component = [seed], []
                while stack:
                    if context is not None:
                        context.checkpoint()
                    i, j = stack.pop()
                    component.append(cells[(i, j)])
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            neighbor = (i + di, j + dj)
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                stack.append(neighbor)
                ids1 = {m["pipeline_1_segment"] for m in component}
                ids2 = {m["pipeline_2_segment"] for m in component}
                length1 = sum(pipelines[p1]["segments"][i].get("length", segment_length) for i in ids1)
                length2 = sum(pipelines[p2]["segments"][i].get("length", segment_length) for i in ids2)
                length = min(length1, length2)
                if length + 1e-8 < min_parallel_length:
                    continue
                # One representative per first-path segment keeps corridor
                # geometry from zigzagging through every nearby segment pair.
                nearest = {}
                for match_position, match in enumerate(component):
                    if context is not None and match_position % 256 == 0:
                        context.check()
                    i = match["pipeline_1_segment"]
                    key = (match["distance"], match["pipeline_2_segment"])
                    old = nearest.get(i)
                    if old is None or key < (old["distance"], old["pipeline_2_segment"]):
                        nearest[i] = match
                qualified.append({
                    "pair": pair,
                    "paths": paths,
                    "matches": component,
                    "representatives": [nearest[i] for i in sorted(nearest)],
                    "segment_ids": (ids1, ids2),
                    "lengths": (length1, length2),
                    "length": length,
                })
    return qualified


def savings_from_sections(pipelines, sections, segment_length, *, context=None):
    """Build disjoint, mutually compatible survey groups of sampled segments.

    A group contains at most one segment from each pipeline and every pair must
    belong to a qualified overlap section. Merge nearest candidates first, with
    geometry-based tie breaking rather than file order. This is a conservative
    deterministic grouping heuristic, not a minimum-flight-route optimizer.
    """
    if context is not None:
        context.report("Building group graph")
    edges = {}
    keys = {}
    def node_key(node):
        if node not in keys:
            pipe, index = node
            segment = pipelines[pipe]["segments"][index]
            keys[node] = (*segment["midpoint"], segment["bearing"] % 180,
                          str(pipelines[pipe].get("name", "")),
                          segment.get("path_index", 0), index)
        return keys[node]

    for section_position, section in enumerate(sections):
        if context is not None and section_position % 256 == 0:
            context.check()
        p1, p2 = section["pair"]
        for match_position, match in enumerate(section['matches']):
            if context is not None and match_position % 256 == 0:
                context.check()
            a = (p1, match["pipeline_1_segment"])
            b = (p2, match["pipeline_2_segment"])
            edge = frozenset((a, b))
            distance = match.get("midpoint_distance", match["distance"])
            edges[edge] = min(edges.get(edge, float("inf")), distance)
    if context is not None:
        context.report("Sorting group candidates", 0, len(edges))
    ordered = sorted(edges, key=lambda edge: (edges[edge], sorted(node_key(n) for n in edge)))
    parent, members = {}, {}
    def root(node):
        if node not in parent:
            parent[node] = node
            members[node] = {node}
        while parent[node] != node:
            if context is not None:
                context.checkpoint()
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    if context is not None:
        context.report("Calculating savings", 0, len(ordered))
    for edge_index, edge in enumerate(ordered):
        if context is not None and edge_index % 256 == 0:
            context.report("Calculating savings", edge_index, len(ordered))
        a, b = sorted(edge, key=node_key)
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        left, right = members[ra], members[rb]
        if {p for p, _ in left} & {p for p, _ in right}:
            continue
        if not all(frozenset((x, y)) in edges for x in left for y in right):
            continue
        parent[rb] = ra
        members[ra] = left | right
        del members[rb]

    savings = 0.0
    for group_position, group in enumerate(members.values()):
        if context is not None and group_position % 256 == 0:
            context.check()
        lengths = [float(pipelines[p]["segments"][i].get("length", segment_length))
                   for p, i in group]
        savings += sum(lengths) - max(lengths)
    return savings
