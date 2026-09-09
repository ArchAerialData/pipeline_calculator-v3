"""Qualify continuous segment coverage once for both reporting and savings."""
from __future__ import annotations

from collections import defaultdict


def qualifying_sections(pipelines, parallel_groups, segment_length, min_parallel_length):
    """Return connected matches, measuring each covered segment only once.

    Matches are cells in the two paths' segment-index grid. Adjacent cells belong
    to one section, regardless of neighbor enumeration or digitization direction.
    Missing segment rows/columns and distinct coordinate paths split sections.
    """
    qualified = []
    for pair, matches in sorted(parallel_groups.items()):
        p1, p2 = pair
        by_paths = defaultdict(dict)
        for match in matches:
            i, j = match["pipeline_1_segment"], match["pipeline_2_segment"]
            s1, s2 = pipelines[p1]["segments"][i], pipelines[p2]["segments"][j]
            paths = (s1.get("path_index", 0), s2.get("path_index", 0))
            cell = (s1.get("path_segment_index", i), s2.get("path_segment_index", j))
            old = by_paths[paths].get(cell)
            if old is None or match["distance"] < old["distance"]:
                by_paths[paths][cell] = match
        for paths, cells in sorted(by_paths.items()):
            remaining = set(cells)
            for seed in sorted(cells):
                if seed not in remaining:
                    continue
                remaining.remove(seed)
                stack, component = [seed], []
                while stack:
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
                for match in component:
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


def savings_from_sections(pipelines, sections, segment_length):
    """Discount only qualified coverage, once per segment and participating pipe.

    A segment covered by k pipelines contributes 1/k of its covered length.
    Coverage on the longer side is scaled to the shorter side's common length.
    Overlapping sections for the same partner use their maximum coverage, never
    add repeated neighbor matches. Unsegmented tails remain undiscounted.
    """
    participants = defaultdict(dict)
    for section in sections:
        p1, p2 = section["pair"]
        for side, (pipe, partner) in enumerate(((p1, p2), (p2, p1))):
            fraction = section["length"] / section["lengths"][side]
            for index in section["segment_ids"][side]:
                members = participants[(pipe, index)]
                members[partner] = max(members.get(partner, 0.0), fraction)
    savings = 0.0
    for (pipe, index), members in participants.items():
        # Integrate the discount over fractional coverage levels. Equal full
        # coverage reduces to 1 - 1/k; partial coverage is not discounted twice.
        previous = 0.0
        discount = 0.0
        for level in sorted(set(members.values())):
            k = 1 + sum(fraction >= level for fraction in members.values())
            discount += (level - previous) * (1 - 1 / k)
            previous = level
        savings += float(pipelines[pipe]["segments"][index].get("length", segment_length)) * discount
    return savings
