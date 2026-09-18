"""Independent, deliberately direct implementation of the frozen sampled contract.

This module does not import application code.  The specification is the public
contract documented in docs/validation/kmz-fixture-agent-prompt.md, read together
with baseline 71da499d5756648ae395660f0a241ea00edbea4f and the documented
September 2026 terminal-sample numerical policy amendment. Distances use GRS80.

Differences in construction from the application are intentional: samples are
located by whole-path cumulative chainages; candidate generation is exhaustive
rectangular array arithmetic, without a spatial tree; sections are connected
components of an explicit Boolean cell graph; grouping maintains an explicit
partition and rewrites every affected member, without a union/find forest.

The public entry points are analyze(sources) and hand_worked_checks().  Sources
are dictionaries with key, name, paths, and optional motif/xml_order.  A path is
a list of (longitude, latitude) coordinates.  Sampling restarts on every path.
No per-source allocation of group savings is invented.
"""

from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "The sampled-contract reference requires enabled assertions; run Python without -O or -OO"
    )

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Iterable

import numpy as np
from pyproj import Geod


GEOD = Geod(ellps="GRS80")
SAMPLE_METERS = 5.0
DETECTION_METERS = 15.0
MINIMUM_METERS = 200.0
ANGLE_DEGREES = 15.0
TERMINAL_ALLOWANCE_METERS = min(1e-6, SAMPLE_METERS * 1e-6)
SURVEY_MILE_METERS = 1609.347218694
# This is smaller than the GRS80 minimum meridional radius of curvature.
# Therefore radius * unit-sphere chord is a lower bound on surface distance.
# The screen can retain extra candidates but cannot discard an eligible pair.
SPHERICAL_LOWER_BOUND_RADIUS = 6_330_000.0
PROFILE = {
    "ellipsoid": "GRS80", "segment_meters": SAMPLE_METERS,
    "detection_meters": DETECTION_METERS,
    "minimum_qualifying_meters": MINIMUM_METERS,
    "angular_tolerance_degrees": ANGLE_DEGREES,
    "survey_mile_meters": SURVEY_MILE_METERS,
    "terminal_sample_allowance_meters": TERMINAL_ALLOWANCE_METERS,
}


@dataclass(frozen=True)
class Sample:
    source: str
    path: int
    index: int
    source_index: int
    lon: float
    lat: float
    bearing: float
    name: str
    motif: str

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.source, self.path, self.index

    @property
    def ordering(self) -> tuple:
        # The name is part of the documented heuristic, so renaming is not a
        # general invariant.  Stable fixture keys identify nodes, not this tie.
        return (self.lon, self.lat, self.bearing % 180.0,
                self.name, self.path, self.source_index)


def _runs(indices: Iterable[int]) -> list[list[int]]:
    """Encode exact coverage using half-open [first, last_exclusive] ranges."""
    ordered = sorted(set(indices))
    result: list[list[int]] = []
    for index in ordered:
        if result and result[-1][1] == index:
            result[-1][1] += 1
        else:
            result.append([index, index + 1])
    return result


def _geodesic_arrays(operation, *values):
    """Keep singleton batches on pyproj's documented vector input path.

    pyproj 3.7.1 first tries its scalar fast path. NumPy singleton arrays can
    enter that path through deprecated array-to-scalar conversion. Lists make
    the vector intent explicit without copying the large multi-sample batches.
    """
    arguments = [value.tolist() for value in values] if len(values[0]) == 1 else values
    return tuple(np.asarray(value, dtype=float) for value in operation(*arguments))


def _sample_path(source: dict, path_index: int, source_offset: int):
    coords = np.asarray(source["paths"][path_index], dtype=float)
    if len(coords) < 2:
        return [], 0.0
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Reference requires finite longitude/latitude pairs")
    if np.any(np.abs(coords[:, 1]) > 90):
        raise ValueError("Latitude outside the ellipsoid")
    azimuths, _, lengths = _geodesic_arrays(GEOD.inv, coords[:-1, 0], coords[:-1, 1],
                                          coords[1:, 0], coords[1:, 1])
    lengths = np.abs(np.asarray(lengths))
    keep = lengths > 0.0
    starts = coords[:-1][keep]
    azimuths = np.asarray(azimuths)[keep]
    lengths = lengths[keep]
    if not len(lengths):
        return [], 0.0
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    original = float(math.fsum(map(float, lengths)))
    # Only the final boundary receives the allowance. Whole-path chainages
    # avoid adding an allowance at each redundant vertex; actual geometry and
    # original mileage remain unchanged, and the final boundary is clamped.
    count = math.floor((original + TERMINAL_ALLOWANCE_METERS) / SAMPLE_METERS)
    if not count:
        return [], original
    boundaries = np.minimum(np.arange(count + 1) * SAMPLE_METERS, cumulative[-1])
    which = np.searchsorted(cumulative[1:], boundaries, side="left")
    which = np.minimum(which, len(lengths) - 1)
    longitude, latitude, _ = _geodesic_arrays(GEOD.fwd, starts[which, 0], starts[which, 1],
                                             azimuths[which], boundaries - cumulative[which])
    # Chord geometry is measured independently of the five meters of path
    # coverage; a sample can straddle any number of original vertices.
    bearings, _, chords = _geodesic_arrays(GEOD.inv, longitude[:-1], latitude[:-1],
                                          longitude[1:], latitude[1:])
    mid_lon, mid_lat, _ = _geodesic_arrays(GEOD.fwd, longitude[:-1], latitude[:-1],
                                         bearings, np.abs(chords) / 2.0)
    samples = [Sample(str(source["key"]), path_index, i, source_offset + i,
                      float(mid_lon[i]), float(mid_lat[i]), float(bearings[i]),
                      str(source.get("name", "")), str(source.get("motif", "")))
               for i in range(count)]
    return samples, original


def _coordinates(samples: list[Sample]):
    values = np.array([(s.lon, s.lat, s.bearing) for s in samples], dtype=float)
    radians = np.deg2rad(values[:, :2])
    unit_sphere = np.column_stack((np.cos(radians[:, 1]) * np.cos(radians[:, 0]),
                                  np.cos(radians[:, 1]) * np.sin(radians[:, 0]),
                                  np.sin(radians[:, 1])))
    return values, unit_sphere


def _matches(left: list[Sample], right: list[Sample], counters: Counter):
    """Examine every sample pair in bounded arrays, then exact geodesic tests."""
    if not left or not right:
        return []
    aa, sphere_a = _coordinates(left)
    bb, sphere_b = _coordinates(right)
    maximum_distance = math.hypot(DETECTION_METERS, SAMPLE_METERS)
    angular_chord_limit = ((maximum_distance + 1e-7) /
                           SPHERICAL_LOWER_BOUND_RADIUS) ** 2
    matches = []
    # About 16 MiB for two temporary 512 x 2048 float arrays.  This screen is
    # vectorized brute force, not an index; visits are reported honestly.
    for a0 in range(0, len(left), 512):
        a1 = min(a0 + 512, len(left))
        for b0 in range(0, len(right), 2048):
            b1 = min(b0 + 2048, len(right))
            squared = np.zeros((a1-a0, b1-b0))
            for dimension in range(3):
                difference = (sphere_a[a0:a1, dimension, None] -
                              sphere_b[None, b0:b1, dimension])
                squared += difference * difference
            counters["exhaustive_sample_pair_visits"] += squared.size
            ii, jj = np.nonzero(squared <= angular_chord_limit)
            if not len(ii):
                continue
            ii, jj = ii+a0, jj+b0
            counters["spherical_screen_survivors"] += len(ii)
            # Reducing modulo 180 handles opposite digitization directly.
            angle = np.abs((aa[ii, 2] - bb[jj, 2] + 90.0) % 180.0 - 90.0)
            orient = angle <= ANGLE_DEGREES
            ii, jj = ii[orient], jj[orient]
            if not len(ii):
                continue
            first_azimuth, second_back_azimuth, distance = _geodesic_arrays(GEOD.inv,
                aa[ii, 0], aa[ii, 1], bb[jj, 0], bb[jj, 1])
            # pyproj's second azimuth points from the second midpoint back to
            # the first, including convergence of meridians on the ellipsoid.
            first_delta = np.deg2rad(first_azimuth - aa[ii, 2])
            second_delta = np.deg2rad(second_back_azimuth - bb[jj, 2])
            transverse = distance * np.maximum(np.abs(np.sin(first_delta)),
                                                np.abs(np.sin(second_delta)))
            longitudinal = distance * np.maximum(np.abs(np.cos(first_delta)),
                                                  np.abs(np.cos(second_delta)))
            counters["geodesic_sample_pair_tests"] += len(ii)
            accepted = ((transverse <= DETECTION_METERS) &
                        (longitudinal < SAMPLE_METERS - 1e-8))
            for p in np.flatnonzero(accepted):
                matches.append((int(ii[p]), int(jj[p]), float(distance[p]),
                                float(transverse[p]), float(longitudinal[p])))
    counters["eligible_sample_pair_count"] += len(matches)
    return matches


def _components(cells: dict):
    """Flood the explicit eight-neighbor sample-index grid."""
    unvisited = set(cells)
    for first in sorted(cells):
        if first not in unvisited:
            continue
        queue = deque([first])
        unvisited.remove(first)
        component = []
        while queue:
            cell = queue.popleft()
            component.append(cells[cell])
            for neighbor in itertools.product(range(cell[0]-1, cell[0]+2),
                                               range(cell[1]-1, cell[1]+2)):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
        yield component


def _group_savings(nodes: dict, eligible_edges: dict):
    """Greedily merge an explicit partition only when its union is a clique."""
    def priority(edge):
        return (eligible_edges[edge], tuple(sorted(nodes[n].ordering for n in edge)))

    # A node's value is its actual current set. Every changed membership is
    # rewritten, making all group membership and disjointness directly visible.
    membership = {n: frozenset([n]) for edge in eligible_edges for n in edge}
    rejected_same_source = rejected_nonclique = accepted_merges = 0
    for edge in sorted(eligible_edges, key=priority):
        a, b = edge
        left, right = membership[a], membership[b]
        if left == right:
            continue
        candidate = left | right
        if len({n[0] for n in candidate}) < len(candidate):
            rejected_same_source += 1
            continue
        if any(tuple(sorted((x, y))) not in eligible_edges
               for x in left for y in right):
            rejected_nonclique += 1
            continue
        for node in candidate:
            membership[node] = candidate
        accepted_merges += 1
    groups = set(membership.values())
    counted = set()
    histogram = Counter()
    motif_savings = Counter()
    source_set_savings = Counter()
    for group in groups:
        assert not counted.intersection(group), "Savings groups must be disjoint"
        counted.update(group)
        assert len(group) == len({n[0] for n in group})
        assert all(tuple(sorted(pair)) in eligible_edges
                   for pair in itertools.combinations(group, 2))
        histogram[len(group)] += 1
        saving = (len(group)-1) * SAMPLE_METERS
        motifs = {nodes[n].motif for n in group}
        motif_key = next(iter(motifs)) if len(motifs) == 1 else "+".join(sorted(motifs))
        motif_savings[motif_key] += saving
        source_set_savings[tuple(sorted(n[0] for n in group))] += saving
    canonical_groups = sorted([sorted(group) for group in groups if len(group) > 1])
    digest = hashlib.sha256(json.dumps(canonical_groups, separators=(",", ":")).encode()).hexdigest()
    savings = sum((size-1)*SAMPLE_METERS*count for size, count in histogram.items())
    return savings, {
        "group_size_counts": {str(k): histogram[k] for k in sorted(histogram)},
        "group_count": len(groups), "accepted_merges": accepted_merges,
        "rejected_shared_source_merges": rejected_same_source,
        "rejected_nonclique_merges": rejected_nonclique,
        "nontrivial_groups_sha256": digest,
        "mutual_compatibility_and_disjointness_checked": True,
        "savings_by_source_set": [
            {"source_keys": list(keys), "savings_meters": source_set_savings[keys]}
            for keys in sorted(source_set_savings) if source_set_savings[keys]],
    }, dict(sorted(motif_savings.items()))


def analyze(sources: list[dict]) -> dict:
    """Return a JSON-serializable independently computed sampled expectation.

    sections includes both qualified and rejected positive match components.
    Coverage ranges are zero-based path-local half-open sample-index intervals.
    source_coverage counts each covered sample once across all qualified pairs.
    motif_savings is only a grouping of measured group savings by construction
    label, never a per-source allocation of multi-source savings.
    """
    source_keys = [str(source["key"]) for source in sources]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Fixture source keys must be unique")
    sources = sorted(sources, key=lambda source: str(source["key"]))
    sampled = {}
    original_by_source = {}
    path_originals = {}
    nodes = {}
    for source in sources:
        key = str(source["key"])
        sampled[key] = []
        path_originals[key] = []
        for p in range(len(source["paths"])):
            samples, length = _sample_path(source, p, len(sampled[key]))
            sampled[key].extend(samples)
            path_originals[key].append(length)
            nodes.update((sample.identity, sample) for sample in samples)
        original_by_source[key] = math.fsum(path_originals[key])

    counters = Counter()
    sections = []
    graph = {}
    coverage = defaultdict(set)
    for first, second in itertools.combinations(sorted(sampled), 2):
        left, right = sampled[first], sampled[second]
        matches = _matches(left, right, counters)
        by_path = defaultdict(dict)
        for match in matches:
            a, b = left[match[0]], right[match[1]]
            by_path[a.path, b.path][a.index, b.index] = match
        for path_pair, cells in sorted(by_path.items()):
            for component in _components(cells):
                first_indices = {left[match[0]].index for match in component}
                second_indices = {right[match[1]].index for match in component}
                section_length = min(len(first_indices), len(second_indices)) * SAMPLE_METERS
                qualified = section_length + 1e-8 >= MINIMUM_METERS
                section = {
                    "source_keys": [first, second], "path_indices": list(path_pair),
                    "qualified": qualified, "length_meters": section_length,
                    "coverage_sample_counts": [len(first_indices), len(second_indices)],
                    "coverage_meters": [len(first_indices)*SAMPLE_METERS,
                                        len(second_indices)*SAMPLE_METERS],
                    "coverage_ranges": [_runs(first_indices), _runs(second_indices)],
                    "eligible_match_count": len(component),
                    "minimum_transverse_meters": min(m[3] for m in component),
                    "maximum_transverse_meters": max(m[3] for m in component),
                    "maximum_longitudinal_meters": max(m[4] for m in component),
                }
                sections.append(section)
                if qualified:
                    for match in component:
                        a, b = left[match[0]].identity, right[match[1]].identity
                        graph[tuple(sorted((a, b)))] = match[2]
                        coverage[first].add((a[1], a[2]))
                        coverage[second].add((b[1], b[2]))

    savings, group_evidence, motif_savings = _group_savings(nodes, graph)
    per_source = {}
    for key in sorted(sampled):
        per_source[key] = {
            "original_meters": original_by_source[key],
            "original_survey_miles": original_by_source[key] / SURVEY_MILE_METERS,
            "path_original_meters": path_originals[key],
            "path_original_survey_miles": [m / SURVEY_MILE_METERS for m in path_originals[key]],
            "sample_count": len(sampled[key]),
            "sampled_meters": len(sampled[key]) * SAMPLE_METERS,
            "unsampled_tail_meters": original_by_source[key] - len(sampled[key])*SAMPLE_METERS,
            "qualifying_covered_sample_count": len(coverage[key]),
            "qualifying_covered_meters": len(coverage[key]) * SAMPLE_METERS,
            "qualifying_coverage_ranges_by_path": {
                str(path): _runs(index for p, index in coverage[key] if p == path)
                for path in range(len(path_originals[key]))},
            "path_sample_counts": [sum(s.path == path for s in sampled[key])
                                   for path in range(len(path_originals[key]))],
        }
    original = math.fsum(original_by_source.values())
    count = sum(map(len, sampled.values()))
    qualifying = [section for section in sections if section["qualified"]]
    return {
        "reference_version": "sampled-contract-2", "profile": dict(PROFILE),
        "status": "complete", "diagnostics": [], "source_count": len(sources),
        "original_meters": original, "original_survey_miles": original / SURVEY_MILE_METERS,
        "sample_count": count, "sampled_meters": count * SAMPLE_METERS,
        "qualifying_section_count": len(qualifying),
        "rejected_section_count": len(sections)-len(qualifying),
        "qualifying_pairwise_meters": math.fsum(s["length_meters"] for s in qualifying),
        "savings_meters": savings, "savings_survey_miles": savings / SURVEY_MILE_METERS,
        "adjusted_meters": original - savings,
        "adjusted_survey_miles": (original - savings) / SURVEY_MILE_METERS,
        "sections": sections, "source_coverage": per_source,
        "motif_savings": motif_savings, "grouping": group_evidence,
        "workload": dict(counters),
    }


def hand_worked_checks() -> dict:
    """Certify exact integer controls using geographic samples and set proofs."""
    def meridians(offsets, length=302.3):
        sources = []
        origin = (-100.0, 31.0)
        for i, offset in enumerate(offsets):
            longitude, _, _ = GEOD.fwd(*origin, 90, offset)
            start = (longitude, origin[1])
            end = GEOD.fwd(*start, 0, length)[:2]
            sources.append({"key": chr(65+i), "name": chr(65+i),
                            "motif": "hand_worked", "paths": [[start, end]]})
        return sources

    evidence = {}
    for label, offsets, pair_count, saving in [
        ("aligned_pair", [0, 8], 1, 300.0),
        ("all_compatible_trio", [0, 6, 13], 3, 600.0),
        ("unequal_nonclique_chain", [0, 7, 18], 2, 300.0),
    ]:
        sources = meridians(offsets)
        result = analyze(sources)
        assert result["qualifying_section_count"] == pair_count, (label, result)
        assert result["savings_meters"] == saving, (label, result)
        assert all(row["sample_count"] == 60 for row in result["source_coverage"].values())
        assert all(s["coverage_ranges"] == [[[0, 60]], [[0, 60]]]
                   for s in result["sections"])
        # Equal-latitude meridians align samples. Prove the actual eligibility
        # graph is exactly diagonal, rather than assuming Q from design length.
        assert all(s["eligible_match_count"] == 60 for s in result["sections"])
        for permutation in itertools.permutations(sources):
            permuted = analyze(list(permutation))
            assert permuted == result, "Stable identities must remove XML-order dependence"
        evidence[label] = {
            "passed": True, "fully_covered_samples_per_source": 60,
            "qualifying_sample_meters_Q": 300.0,
            "qualifying_pair_count": pair_count,
            "qualifying_pairwise_meters": result["qualifying_pairwise_meters"],
            "expected_savings_meters": saving,
            "measured_savings_meters": result["savings_meters"],
            "eligible_graph": "60 diagonal cells per eligible pair; no off-diagonal cells",
            "group_size_counts": result["grouping"]["group_size_counts"],
            "all_source_order_permutations_equal": True,
        }
    # A disconnected source must restart samples and section qualification.
    short = meridians([0, 8], length=123.0)
    for source in short:
        start = GEOD.fwd(*source["paths"][0][0], 0, 500.0)[:2]
        end = GEOD.fwd(*start, 0, 123.0)[:2]
        source["paths"].append([start, end])
    result = analyze(short)
    assert result["savings_meters"] == 0
    assert result["qualifying_section_count"] == 0
    assert [s["length_meters"] for s in result["sections"]] == [120.0, 120.0]
    evidence["disconnected_subthreshold"] = {
        "passed": True, "section_meters": [120.0, 120.0],
        "summed_pairwise_candidate_meters": 240.0, "savings_meters": 0.0,
    }
    retrace = meridians([0])[0]
    retrace["paths"][0].append(retrace["paths"][0][0])
    result = analyze([retrace])
    assert result["savings_meters"] == 0 and not result["sections"]
    evidence["single_source_retrace"] = {"passed": True, "savings_meters": 0.0,
                                          "sample_count": result["sample_count"]}
    # A 1.7 m phase shift creates off-diagonal matches. The nearest eligible
    # pairing still covers all 60 samples, unlike the alignment proof above.
    shifted = meridians([0, 8])
    shifted[1]["paths"][0] = [GEOD.fwd(*point, 0, 1.7)[:2]
                                for point in shifted[1]["paths"][0]]
    result = analyze(shifted)
    assert result["savings_meters"] == 300.0
    assert result["sections"][0]["eligible_match_count"] == 119
    evidence["nonaligned_finite_tangents"] = {
        "passed": True, "longitudinal_phase_meters": 1.7,
        "eligible_match_count": 119, "savings_meters": 300.0,
        "group_size_counts": result["grouping"]["group_size_counts"],
    }
    reversed_pair = meridians([0, 8])
    reversed_pair[1]["paths"][0].reverse()
    result = analyze(reversed_pair)
    assert result["savings_meters"] == 300.0
    assert result["sample_count"] == 120
    evidence["opposite_direction_phase_safe_pair"] = {
        "passed": True, "savings_meters": 300.0, "sample_count": 120,
        "scope": "This 302.3 m two-meridian control only; reversal is not a universal savings invariant",
    }
    redundant = meridians([0, 8])
    path = redundant[1]["paths"][0]
    path.insert(1, GEOD.fwd(*path[0], 0, 82.1)[:2])
    result = analyze(redundant)
    assert result["savings_meters"] == 300.0 and result["sample_count"] == 120
    evidence["redundant_geodesic_vertex"] = {
        "passed": True, "savings_meters": 300.0,
        "sample_count": 120, "inserted_chainage_meters": 82.1,
    }
    # Use a second geodesic implementation to prove the five-meter sample
    # spans the bend and uses its endpoint chord, not the path's half-chainage.
    from geographiclib.geodesic import Geodesic
    second_geod = Geodesic(6378137.0, 1.0 / 298.257222101)
    origin = (-100.0, 31.0)
    turn = second_geod.Direct(origin[1], origin[0], 90.0, 2.0)
    finish = second_geod.Direct(turn["lat2"], turn["lon2"], 0.0, 6.3)
    boundary = second_geod.Direct(turn["lat2"], turn["lon2"], 0.0, 3.0)
    chord = second_geod.Inverse(origin[1], origin[0], boundary["lat2"], boundary["lon2"])
    mid = second_geod.Direct(origin[1], origin[0], chord["azi1"], chord["s12"]/2)
    bent = {"key": "bend", "name": "bend", "paths": [[origin,
            (turn["lon2"], turn["lat2"]), (finish["lon2"], finish["lat2"])]]}
    samples, original = _sample_path(bent, 0, 0)
    assert len(samples) == 1 and abs(original - 8.3) < 1e-6
    sample = samples[0]
    midpoint_error = second_geod.Inverse(sample.lat, sample.lon, mid["lat2"], mid["lon2"])["s12"]
    bearing_error = abs(sample.bearing - chord["azi1"])
    assert midpoint_error < 1e-6 and bearing_error < 1e-6
    assert abs(sample.bearing) > 30.0
    assert abs(chord["s12"] - math.sqrt(13)) < 1e-6
    evidence["bend_spanning_chord"] = {
        "passed": True, "independent_implementation": "GeographicLib GRS80",
        "original_meters": original, "sampled_path_meters": 5.0,
        "sample_chord_meters": chord["s12"],
        "sample_bearing_degrees": sample.bearing,
        "midpoint_implementation_difference_meters": midpoint_error,
        "bearing_implementation_difference_degrees": bearing_error,
    }
    return evidence


if __name__ == "__main__":
    print(json.dumps(hand_worked_checks(), indent=2, sort_keys=True))
