"""Numerical intent checks, separate from regenerating a golden JSON document.

Assertions below encode the requested behavior and generator design. A changed
reference result must satisfy these constraints before it can become a golden.
The full-precision saved expectation additionally fixes every measured sample
count, qualifying section and savings value; these checks do not replace it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pyproj import Geod

SUITE = Path(__file__).resolve().parents[1]
GEOD = Geod(ellps="GRS80")


def check_fixture(expected: dict, *, root=SUITE) -> list[dict]:
    """Return reviewable check records; a failed intent is never silently reset."""
    checks = []

    def check(name, passed, assertion, measured, sources=()):
        key_to_motif = {s["key"]: s.get("motif", "") for s in expected["geometry"]["sources"]}
        checks.append({"id": name, "passed": bool(passed), "assertion": assertion,
                       "measured": measured, "source_keys": list(sources),
                       "motifs": sorted({key_to_motif[k] for k in sources if k in key_to_motif})})

    geometry = expected["geometry"]
    analyses = expected["analyses"]
    combined = analyses["Combined"]
    sources = {s["key"]: s for s in geometry["sources"]}
    root = Path(root)
    manifest = json.loads((root / "validation/design_manifest.json").read_text(encoding="utf-8"))
    stem = Path(expected["fixture"]).stem.split("__")[0]
    design = next((f for f in manifest["fixtures"] if f["name"] == stem), None)
    if design is None:
        raise ValueError(f'No reviewed design entry for {stem}')
    from pipeline_kmz_regression_suite.reference.geometry import read_kmz
    serialized = {s["key"]: s for s in read_kmz(root / "fixtures" / expected["fixture"])}
    states = sorted(k for k in analyses if k != "Combined")
    original = geometry["original_meters"]
    tolerance = max(.001, original * 1e-10)

    def rows(keys, scope="Combined", qualified=True):
        keys = set(keys)
        return [s for s in analyses[scope]["sections"]
                if set(s["source_keys"]) <= keys and
                (qualified is None or s["qualified"] == qualified)]

    def group_savings(keys, scope="Combined"):
        keys = set(keys)
        return math.fsum(g["savings_meters"] for g in analyses[scope]["grouping"]["savings_by_source_set"]
                         if set(g["source_keys"]) <= keys)

    def covered(key, scope="Combined"):
        return analyses[scope]["source_coverage"].get(key, {}).get("qualifying_covered_sample_count", 0)

    def interior(key, code):
        return sources[key]["states"].get(code, {}).get("interior_meters", 0.0)

    def intervals(key, kind=None):
        return [r for r in geometry["intervals"] if r["source_key"] == key and
                (kind is None or r["kind"] == kind)]

    def events(key):
        return [c for c in geometry["crossings"] if c["source_key"] == key]

    def zero_controls(keys):
        measured = {k: covered(k) for k in keys}
        check("negative_source_coverage_" + keys[0], not any(measured.values()),
              "Each named negative source has exactly zero qualifying covered samples", measured, keys)

    check("resolved_complete_coverage", geometry["outside_meters"] == geometry["unresolved_meters"] == 0,
          "Outside and unresolved length are both exactly zero",
          {k: geometry[k] for k in ("outside_meters", "unresolved_meters")})
    total_attributed = math.fsum(s["attributed_original_meters"] for s in geometry["states"].values())
    difference = total_attributed + geometry["outside_meters"] + geometry["unresolved_meters"] - original
    check("fixture_conservation", abs(difference) <= tolerance,
          "sum(state attributed) + outside + unresolved equals original within max(0.001 m, original*1e-10)",
          {"difference_meters": difference, "tolerance_meters": tolerance})
    for key, source in sources.items():
        attributed = math.fsum(s["attributed_original_meters"] for s in source["states"].values())
        other = math.fsum(r["length_meters"] for r in intervals(key) if r["kind"] in ("outside", "unresolved"))
        delta = attributed + other - source["original_meters"]
        bound = max(.001, source["original_meters"] * 1e-10)
        check("source_conservation_" + key, abs(delta) <= bound,
              "Source attributed + outside + unresolved equals its original within the conservation bound",
              {"difference_meters": delta, "bound_meters": bound}, [key])
    cut_bound = geometry["certification"]["maximum_cut_error_bound_meters"]
    check("certified_cut_locations", cut_bound <= .01,
          "Every cut is certified to <=0.01 m along its source geodesic", {"maximum_bound_meters": cut_bound})
    if design:
        desired_states = sorted(design["description"]["states"])
        check("represented_states", states == desired_states,
              "Represented states equal the verified design states", {"actual": states, "required": desired_states})
        check("source_identity_count", len(sources) == design["source_count"] == expected["archive"]["source_count"],
              "Every actual source has a distinct fixture identity; repeated display fields do not merge it",
              {"expected": design["source_count"], "actual": len(sources)})

    for scope, analysis in analyses.items():
        all_sections = analysis["sections"]
        valid_counts = all(s["coverage_meters"] == [5.0*n for n in s["coverage_sample_counts"]]
                           and s["length_meters"] == 5.0*min(s["coverage_sample_counts"])
                           and s["qualified"] == (min(s["coverage_sample_counts"]) >= 40)
                           for s in all_sections)
        check("sample_units_" + scope, valid_counts,
              "Coverage uses exact 5 m units; a component qualifies iff both unique sample counts reach 40",
              {"section_count": len(all_sections)})
        union = {key: set() for key in analysis["source_coverage"]}
        for section in all_sections:
            if not section["qualified"]:
                continue
            for key, path, ranges in zip(section["source_keys"], section["path_indices"], section["coverage_ranges"]):
                for start, stop in ranges:
                    union[key].update((path, index) for index in range(start, stop))
        exact_unique_coverage = True
        for key, entry in analysis["source_coverage"].items():
            reported = {(int(path), index)
                        for path, ranges in entry["qualifying_coverage_ranges_by_path"].items()
                        for start, stop in ranges for index in range(start, stop)}
            exact_unique_coverage &= (reported == union[key] and
                                      len(reported) == entry["qualifying_covered_sample_count"] and
                                      5 * len(reported) == entry["qualifying_covered_meters"])
        check("unique_coverage_union_" + scope, exact_unique_coverage,
              "Each source's reported coverage is the exact union of qualified sample-index sets, counted once across pairs",
              {"covered_sample_counts": {key: len(value) for key, value in union.items()}})
        base = analysis.get("attributed_original_meters", analysis["original_meters"])
        delta = analysis["adjusted_meters"] - (base - analysis["savings_meters"])
        histogram = analysis["grouping"]["group_size_counts"]
        group_total = sum((int(size)-1)*5*count for size, count in histogram.items())
        check("group_and_adjusted_accounting_" + scope,
              abs(delta) < 1e-7 and group_total == analysis["savings_meters"],
              "Adjusted equals attributed original minus savings; disjoint group savings are (members-1)*5 m",
              {"adjusted_difference_meters": delta, "group_savings_meters": group_total})
        if scope != "Combined":
            account = geometry["states"][scope]
            check("exclusive_state_inputs_" + scope,
                  abs(analysis["original_meters"] - account["interior_meters"]) <= tolerance and
                  abs(account["attributed_original_meters"] - account["interior_meters"] - account["shared_allocation_meters"]) <= tolerance,
                  "Only exclusive interiors enter state sampling; attributed original adds shared allocation",
                  {"sample_input_original_meters": analysis["original_meters"], **account})

    if stem.startswith(("01_", "02_", "03_")):
        check("zero_shared_border", geometry["shared_meters"] == 0.0,
              "No positive-length interval is shared border geometry", {"shared_meters": geometry["shared_meters"]})

    if stem.startswith("01_"):
        check("states_without_crossing_events", geometry["crossing_count"] == 0 and len(states) == 3,
              "Three represented states and exactly zero border crossing events",
              {"states": states, "crossing_events": geometry["crossing_count"]})
        check("unchanged_scoped_sampling", combined["savings_meters"] == sum(analyses[c]["savings_meters"] for c in states),
              "Independent unchanged interior paths yield Combined savings equal to the sum of states",
              {"combined": combined["savings_meters"], "state_sum": sum(analyses[c]["savings_meters"] for c in states)})
        for code in ("tx", "la", "wy"):
            prefix = "01_" + code
            for kind, letters, q, saving, pair_count in [
                ("pair", "ab", 1300.0, 1300.0, 1),
                ("trio", "abc", 800.0, 1600.0, 3),
                ("chain", "abc", 700.0, 700.0, 2),
            ]:
                keys = [f"{prefix}_{kind}_{letter}" for letter in letters]
                selected = rows(keys)
                ranges_ok = all(s["coverage_ranges"] == [[[0, int(q/5)]], [[0, int(q/5)]]]
                                for s in selected)
                measured_saving = group_savings(keys)
                check(prefix + "_" + kind, len(selected) == pair_count and ranges_ok and measured_saving == saving,
                      f"Full-coverage control has {pair_count} qualifying source pairs, Q={q:g} m fully covered per source and savings={saving:g} m",
                      {"pair_count": len(selected), "coverage_ranges": [s["coverage_ranges"] for s in selected],
                       "savings_meters": measured_saving}, keys)
                if kind == "chain":
                    selected_groups = [g for g in combined["grouping"]["savings_by_source_set"] if set(g["source_keys"]) <= set(keys)]
                    check(prefix + "_chain_no_triple", all(len(g["source_keys"]) <= 2 for g in selected_groups),
                          "The incompatible outside pair prevents every three-member group", selected_groups, keys)
            keys = [f"{prefix}_multipart_{letter}" for letter in "ab"]
            parts = rows(keys, qualified=None)
            lengths = [p["original_meters"] for k in keys for p in sources[k]["paths"]]
            check(prefix + "_multipart_no_gap_bridge",
                  len(parts) == 2 and all(s["length_meters"] == 120 and not s["qualified"] for s in parts)
                  and all(0 < n < 200 for n in lengths) and sum(lengths[:2]) > 200,
                  "Two disconnected 120 m sampled runs remain separate rejected components despite >200 m combined path length",
                  {"component_lengths_meters": [s["length_meters"] for s in parts], "path_originals_meters": lengths}, keys)
            zero_controls([f"{prefix}_{suffix}" for suffix in ("cross_east_west", "cross_north_south", "loop", "isolated", "diverging_branch", "feeder")])

    elif stem.startswith("02_"):
        route = "02_three_state_route"
        transitions = events(route)
        check("continuous_three_state_route", set(sources[route]["states"]) == {"NM", "TX", "OK"} and len(transitions) == 2,
              "One continuous route visits NM, TX and OK through two separate crossings",
              {"states": sorted(sources[route]["states"]), "crossing_count": len(transitions)}, [route])
        route_coordinates = serialized[route]["paths"][0]
        first_edge = abs(GEOD.inv(*route_coordinates[0], *route_coordinates[1])[2])
        first_cut = transitions[0]["chainage_meters"] if transitions else -1
        check("sparse_edge_interior_crossing", 0.01 < first_cut < first_edge-.01,
              "The first crossing is strictly inside the first serialized geodesic edge, more than 0.01 m from either endpoint",
              {"crossing_chainage_meters": first_cut, "first_edge_length_meters": first_edge}, [route])
        angles = [c["angle_degrees"] for c in geometry["crossings"]]
        check("transverse_crossing_angles", bool(angles) and all(a is not None and a > 80 for a in angles),
              "Every measured crossing angle exceeds 80 degrees relative to the local boundary tangent", {"angles_degrees": angles})
        reentry = "02_reentry_route"
        nm_parts = [r for r in intervals(reentry, "interior") if r["state_codes"] == ["NM"]]
        check("state_reentry_preserved", len(events(reentry)) >= 3 and len(nm_parts) == 2,
              "Reentry has at least 3 transitions and retains 2 separated NM interior fragments",
              {"crossing_count": len(events(reentry)), "nm_fragment_chainages": [[r["start_m"], r["end_m"]] for r in nm_parts]}, [reentry])
        touch = "02_endpoint_touch"
        check("endpoint_touch_zero_mileage", interior(touch, "TX") == 0 and len(events(touch)) == 0
              and any(t["source_key"] == touch and t["touched_states"] == ["TX"] for t in geometry["endpoint_touches"]),
              "Endpoint touch records TX with exactly 0 positive TX mileage and no crossing event",
              {"tx_meters": interior(touch, "TX"), "crossing_count": len(events(touch))}, [touch])
        short = "02_short_state_visit"
        check("short_visit_preserved", len(events(short)) == 2 and 0 < interior(short, "OK") < 250,
              "A short OK visit retains two crossings and positive mileage below 250 m",
              {"ok_meters": interior(short, "OK"), "crossing_count": len(events(short))}, [short])
        for code in ("NM", "TX", "OK"):
            keys = [f"02_{code.lower()}_pair_{letter}" for letter in "ab"]
            check(code + "_exclusive_positive_pair", group_savings(keys) == group_savings(keys, code) == 1100,
                  "The isolated interior pair has 220 fully covered samples and 1100 m savings in Combined and its state",
                  {"combined_savings": group_savings(keys), "state_savings": group_savings(keys, code)}, keys)
        zero_controls([k for k in sources if "_pair_" not in k])

    elif stem.startswith("03_"):
        for label, required_scopes in (("long", ("NM", "TX")), ("short", ()), ("asymmetric", ("NM",))):
            keys = [f"03_{label}_{letter}" for letter in "ab"]
            scoped = {scope: group_savings(keys, scope) for scope in states}
            check(label + "_state_qualification", group_savings(keys) > 0 and
                  all((scoped[c] > 0) == (c in required_scopes) for c in states),
                  "Combined qualifies; exact qualifying states are " + (",".join(required_scopes) or "none"),
                  {"combined_savings": group_savings(keys), "state_savings": scoped}, keys)
            if label == "short":
                state_counts = {c: [analyses[c]["source_coverage"].get(k, {}).get("sample_count", 0) for k in keys] for c in ("NM", "TX")}
                check("short_split_60_vs_under40_samples", all(analyses["Combined"]["source_coverage"][k]["sample_count"] == 60 for k in keys)
                      and all(0 < n < 40 for counts in state_counts.values() for n in counts),
                      "Both complete sources have exactly 60 full samples; each state fragment has fewer than 40",
                      {"state_sample_counts": state_counts}, keys)
        trio = ["03_trio_" + letter for letter in "abc"]
        counts = {scope: len(rows(trio, scope)) for scope in ("Combined", "NM", "TX")}
        nm_pair = rows(trio, "NM")
        check("joining_trio_qualification", counts == {"Combined": 3, "NM": 1, "TX": 3}
              and set(nm_pair[0]["source_keys"]) == {"03_trio_a", "03_trio_b"},
              "Joining trio has three qualifying pairs Combined/TX; only A-B qualifies in NM",
              {"qualified_pair_counts": counts}, trio)
        check("joining_trio_no_pairwise_overcount", group_savings(trio) < sum(s["length_meters"] for s in rows(trio)),
              "Trio disjoint group savings are strictly below its summed qualifying pairwise coverage",
              {"savings_meters": group_savings(trio), "pairwise_meters": sum(s["length_meters"] for s in rows(trio))}, trio)
        keys = ["03_rejoin_a", "03_rejoin_b"]
        parts = rows(keys)
        separated = len(parts) >= 2 and all(parts[0]["coverage_ranges"][i][-1][1] < parts[1]["coverage_ranges"][i][0][0] for i in (0, 1))
        check("diverge_rejoin_separate_sections", separated,
              "At least 2 qualifying sections have a positive uncovered sample gap on both sources",
              {"coverage_ranges": [s["coverage_ranges"] for s in parts]}, keys)
        opposite = ["03_opposite_nm", "03_opposite_tx"]
        check("opposite_sides_no_state_import", group_savings(opposite) > 0 and all(group_savings(opposite, c) == 0 for c in states)
              and set(sources[opposite[0]]["states"]) == {"NM"} and set(sources[opposite[1]]["states"]) == {"TX"},
              "Opposite-side lines qualify Combined, remain exclusive NM/TX, and give zero state savings",
              {"combined_savings": group_savings(opposite), "state_savings": {c: group_savings(opposite, c) for c in states}}, opposite)
        phase = ["03_phase_a", "03_phase_b"]
        tails = [combined["source_coverage"][k]["unsampled_tail_meters"] for k in phase]
        check("phase_and_tail_positive_control", group_savings(phase) > 0 and all(.1 < t < 4.9 for t in tails)
              and len(rows(phase)) == 1 and rows(phase)[0]["eligible_match_count"] > rows(phase)[0]["coverage_sample_counts"][0],
              "Nonzero tails remain in originals; nonaligned starts produce off-diagonal finite-tangent matches and positive savings",
              {"tails_meters": tails, "savings_meters": group_savings(phase), "eligible_match_count": rows(phase)[0]["eligible_match_count"]}, phase)
        phase_paths = [serialized[k]["paths"][0] for k in phase]
        bearings = [GEOD.inv(*path[0], *path[-1])[0] for path in phase_paths]
        direction_difference = abs((bearings[0] - bearings[1] + 180) % 360 - 180)
        vertex_counts = [len(path) for path in phase_paths]
        check("opposite_digitization_dense_sparse", direction_difference > 165 and
              min(vertex_counts) == 2 and max(vertex_counts) > 20,
              "Serialized phase controls have opposite digitization within 15 degrees and sparse versus dense original vertices",
              {"endpoint_bearings_degrees": bearings, "direction_difference_degrees": direction_difference,
               "serialized_vertex_counts": vertex_counts}, phase)
        zero_controls([k for k in sources if any(tag in k for tag in ("_feeder", "_loop", "_spur"))])

    elif stem.startswith("04_"):
        shared = [r for r in geometry["intervals"] if r["kind"] == "shared"]
        allocations = geometry["shared_allocations"]
        allocation_ok = bool(shared)
        for row in shared:
            records = [a for a in allocations if a["interval_id"] == row["id"]]
            allocation_ok &= row["state_codes"] == ["NM", "TX"] and len(records) == 2
            allocation_ok &= {a["state_code"] for a in records} == {"NM", "TX"}
            allocation_ok &= all(abs(a["allocated_meters"] - row["length_meters"]/2) < 1e-9 for a in records)
        check("shared_stored_once_allocated_equally", allocation_ok and len({r["id"] for r in shared}) == len(shared),
              "Each shared interval occurs once and has exactly two half-length allocation references for NM/TX",
              {"shared_interval_count": len(shared), "allocation_count": len(allocations), "shared_meters": geometry["shared_meters"]})
        short = "04_shared_short"
        short_rows = intervals(short, "shared")
        check("subthreshold_shared_allocation", len(short_rows) == 1 and 0 < short_rows[0]["length_meters"] < 200
              and set(sources[short]["states"]) == {"NM", "TX"}
              and all(v["shared_allocation_meters"] > 0 for v in sources[short]["states"].values()),
              "A positive shared run below 200 m allocates original length to both states",
              {"original_meters": sources[short]["original_meters"], "state_accounts": sources[short]["states"]}, [short])
        for side, code in (("east", "TX"), ("west", "NM")):
            key = "04_centimeter_" + side
            native = design["description"]["boundary_edges"][0]
            meridian = native["segment_start"][0]
            assert native["segment_end"][0] == meridian
            offsets = [abs(GEOD.inv(lon, lat, meridian, lat)[2])
                       for path in serialized[key]["paths"] for lon, lat in path]
            check("centimeter_offset_" + side, set(sources[key]["states"]) == {code}
                  and not intervals(key, "shared") and interior(key, code) > 400
                  and all(.029 < offset < .031 for offset in offsets),
                  "The serialized 3 cm offset measures 0.029–0.031 m from the canonical meridian and is wholly exclusive to " + code,
                  {"state_accounts": sources[key]["states"], "shared_interval_count": len(intervals(key, "shared")),
                   "measured_offset_range_meters": [min(offsets), max(offsets)]}, [key])
        keys = ["04_shared_qualification_route", "04_shared_qualification_partner"]
        check("shared_cannot_qualify_state", group_savings(keys) > 0 and all(not rows(keys, c) for c in states),
              "Shared-adjacent control qualifies Combined but no state; shared coverage cannot qualify an interior run",
              {"combined_savings": group_savings(keys), "state_qualified_section_counts": {c: len(rows(keys, c)) for c in states}}, keys)
        tiny = "04_tiny_crossing"
        tiny_lengths = {c: interior(tiny, c) for c in ("NM", "TX")}
        check("centimeter_crossing_retained", len(events(tiny)) == 1 and all(0 < v < .1 for v in tiny_lengths.values()),
              "A real 10 cm crossing retains one event and positive sub 10 cm exclusive intervals in both states",
              tiny_lengths, [tiny])
        touch = "04_endpoint_touch"
        check("shared_fixture_endpoint_touch", interior(touch, "TX") == 0 and len(events(touch)) == 0
              and any(t["source_key"] == touch and t["touched_states"] == ["TX"] for t in geometry["endpoint_touches"]),
              "The endpoint touches TX while contributing exactly zero TX length and zero crossing events",
              {"tx_meters": interior(touch, "TX"), "crossing_count": len(events(touch))}, [touch])
    return checks
