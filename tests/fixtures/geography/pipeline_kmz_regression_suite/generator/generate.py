"""Deterministic, offline synthetic pipeline inputs; no production imports.

Run from any working directory. Numerical expectations are intentionally supplied
by the separate reference programs, which read these saved archives directly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile
import xml.etree.ElementTree as ET

from pyproj import Geod
from shapely import from_wkb
from shapely.geometry import LineString, Point, box

GENERATOR_VERSION = "1.0.0"
SEED = 71499
BASELINE = "71da499d5756648ae395660f0a241ea00edbea4f"
BOUNDARY_SHA256 = "55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539"
SUITE = Path(__file__).resolve().parents[1]
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "src/pipeline_calculator/data/states_2025.zip").is_file())
BOUNDARY = REPOSITORY / "src/pipeline_calculator/data/states_2025.zip"
GEOD = Geod(ellps="GRS80")
KML = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML)


def destination(point, bearing, distance):
    lon, lat, _ = GEOD.fwd(*point, bearing, distance)
    return (float(lon), float(lat))


def local(origin, x, y):
    return destination(origin, math.degrees(math.atan2(x, y)), math.hypot(x, y))


def geodesic_vertices(path, spacing):
    """Insert points on each existing ellipsoidal geodesic, preserving vertices."""
    result = [path[0]]
    for a, b in zip(path, path[1:]):
        bearing, _, length = GEOD.inv(*a, *b)
        n = max(1, math.ceil(length / spacing))
        result.extend(destination(a, bearing, length * i / n) for i in range(1, n))
        result.append(b)
    return result


def line(origin, points, dense=False):
    result = [local(origin, *p) for p in points]
    return geodesic_vertices(result, 43.0) if dense else result


def source(key, motif, paths, *, name=None, state=None, note=""):
    return {"key": key, "motif": motif, "name": name or key.replace("_", " "),
            "paths": paths, "state_hint": state, "note": note}


def load_boundaries():
    digest = hashlib.sha256(BOUNDARY.read_bytes()).hexdigest()
    if digest != BOUNDARY_SHA256:
        raise ValueError(f"Boundary checksum mismatch: {digest}")
    with zipfile.ZipFile(BOUNDARY) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        states = {}
        for row in manifest["states"]:
            data = archive.read(row["file"])
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Boundary member checksum mismatch: {row['code']}")
            states[row["code"]] = from_wkb(data)
    return states, manifest


def edge_provenance(states, codes, a, b):
    """Locate the actual unsimplified native ring edges containing a segment."""
    records = []
    for code in codes:
        geometry = states[code]
        polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for pi, polygon in enumerate(polygons):
            for ri, ring in enumerate([polygon.exterior, *polygon.interiors]):
                coords = list(ring.coords)
                for ei, (v, w) in enumerate(zip(coords, coords[1:])):
                    segment = LineString([v, w])
                    if segment.distance(Point(a)) < 1e-12 and segment.distance(Point(b)) < 1e-12:
                        records.append({"state": code, "polygon_index": pi, "ring_index": ri,
                                        "edge_index": ei, "canonical_start": list(v), "canonical_end": list(w)})
    if {r["state"] for r in records} != set(codes):
        raise ValueError(f"Missing native shared-edge provenance for {a}, {b}")
    return records


def crossing_frame(states, codes, target):
    """Read a nearby native common edge; choose its coordinate-linear midpoint."""
    common = states[codes[0]].boundary.intersection(states[codes[1]].boundary)
    lines = list(common.geoms) if hasattr(common, "geoms") else [common]
    edges = [(tuple(a), tuple(b)) for ln in lines if ln.geom_type == "LineString"
             for a, b in zip(ln.coords, list(ln.coords)[1:])]
    a, b = min(edges, key=lambda ab: LineString(ab).distance(Point(target)))
    seg = LineString([a, b])
    q = seg.interpolate(seg.project(Point(target)))
    center = (q.x, q.y)
    # Tiny longitude/latitude interpolation is intentionally in the existing
    # boundary coordinate model, not a new CRS transformation or geodesic edge.
    return center, {"states": list(codes), "segment_start": a, "segment_end": b,
                    "construction_crossing_center": center,
                    "native_edges": edge_provenance(states, codes, a, b)}


def fixture_one():
    out = []
    origins = {"TX": (-100.0, 31.0), "LA": (-92.5, 31.0), "WY": (-107.0, 43.0)}
    for state, origin in origins.items():
        prefix = "01_" + state.lower()
        def add(suffix, motif, points, dense=False, name=None):
            out.append(source(prefix + "_" + suffix, state.lower() + "_" + motif,
                              [line(origin, points, dense)], name=name, state=state))
        for tag, x in [("a", 0), ("b", 8)]:
            add("pair_" + tag, "long_pair", [(x, 0), (x, 1303.25)], tag == "b", "Gathering lateral")
        for tag, x in [("a", 1600), ("b", 1606), ("c", 1613)]:
            add("trio_" + tag, "clique_trio", [(x, 0), (x, 803.25)], tag == "c", "Gathering lateral")
        for tag, x in [("a", 3100), ("b", 3107), ("c", 3119)]:
            add("chain_" + tag, "chain_nonclique", [(x, 0), (x, 703.25)], tag == "b")
        add("cross_east_west", "perpendicular", [(4100, 400), (4903.25, 400)])
        add("cross_north_south", "perpendicular", [(4500, 0), (4500, 803.25)])
        add("diverging_branch", "diverging_branch", [(0, 350), (-300, 650), (-650, 1000), (-400, 1350)], True)
        add("loop", "loop", [(1700, 1500), (2100, 1500), (2100, 1900), (1700, 1900), (1700, 1500)], True)
        for tag, x in [("a", 3100), ("b", 3108)]:
            paths = [line(origin, [(x, 1450), (x, 1573.25)], tag == "b"),
                     line(origin, [(x, 1950), (x, 2073.25)], tag == "b")]
            out.append(source(prefix + "_multipart_" + tag, state.lower() + "_multipart_short", paths,
                              state=state, note="Two disconnected 123.25 m design runs; qualification restarts on each."))
        add("isolated", "isolated", [(4600, 1500), (4950, 2150), (4500, 2500)], True)
        add("feeder", "connecting_feeder", [(0, 1303.25), (500, 1750), (1100, 1750), (1700, 1500)], True)
    return out, {"states": list(origins), "centers": origins, "boundary_edges": [],
                 "intent": "Three widely separated interior networks, each with independently isolated positive and negative motifs."}


def fixture_two(states):
    out = []
    # A single sparse edge crosses NM/TX strictly between endpoints; the next
    # edge crosses TX/OK several kilometres away from the three-state junction.
    out.append(source("02_three_state_route", "three_state_route", [[(-103.064, 36.469), (-102.975, 36.469), (-102.975, 36.528)]]))
    out.append(source("02_reentry_route", "reentry", [[(-103.061, 36.448), (-103.025, 36.448),
                                                        (-103.025, 36.455), (-103.061, 36.455),
                                                        (-103.061, 36.460), (-103.025, 36.460)]]))
    out.append(source("02_ok_reentry", "reentry", [[(-102.953, 36.489), (-102.953, 36.513),
                                                    (-102.934, 36.513), (-102.934, 36.490)]]))
    center, native = crossing_frame(states, ("NM", "TX"), (-103.0417, 36.480))
    # An interpolated point on an oblique native edge can serialize fractions of
    # a nanometre to either side. Use the native vertex itself for an exact touch.
    touch = tuple(native["segment_end"])
    native["endpoint_touch_coordinate"] = touch
    out.append(source("02_endpoint_touch", "endpoint_touch", [[destination(touch, 270, 363.25), touch]],
                      note="Ends exactly at a canonical common-edge vertex; no positive TX visit."))
    # Short real visit near a local common edge. Its full U-shaped path preserves
    # the entry/exit connectors and cannot be reconstructed as one straight cut.
    shortcenter, native2 = crossing_frame(states, ("TX", "OK"), (-102.920, 36.5003))
    out.append(source("02_short_state_visit", "short_visit", [line(shortcenter, [(-100, -153.25), (-100, 2.3), (100, 2.3), (100, -163.25)])]))
    for state, origin in {"NM": (-103.09, 36.475), "TX": (-103.012, 36.432), "OK": (-102.94, 36.542)}.items():
        prefix = "02_" + state.lower()
        for tag, x in [("a", 0), ("b", 8)]:
            out.append(source(prefix + "_pair_" + tag, state.lower() + "_interior_pair",
                              [line(origin, [(x, 0), (x, 1103.25)], tag == "b")], name="Distribution branch", state=state))
        out.append(source(prefix + "_branch", state.lower() + "_branch",
                          [line(origin, [(0, 430), (-370, 710), (-640, 1010)], True)], state=state))
        out.append(source(prefix + "_loop", state.lower() + "_loop",
                          [line(origin, [(900, 0), (1250, 0), (1250, 350), (900, 350), (900, 0)], True)], state=state))
        out.append(source(prefix + "_spur", state.lower() + "_spur",
                          [line(origin, [(0, 1103.25), (380, 1530), (1000, 1630)])], state=state))
    return out, {"states": ["NM", "TX", "OK"], "boundary_edges": [native, native2],
                 "intent": "Sparse transverse crossings, repeated visits, zero-length border touch, short positive visit, and interior branch networks."}


def fixture_three(states):
    out, edges = [], []
    def center(target, codes=("NM", "TX")):
        origin, edge = crossing_frame(states, codes, target)
        edges.append(edge)
        return origin
    def add(key, motif, origin, points, dense=False, reverse=False, name=None):
        path = line(origin, points, dense)
        out.append(source("03_" + key, motif, [list(reversed(path)) if reverse else path], name=name))
    o = center((-103.04162, 36.435))
    for tag, y in [("a", 0), ("b", 8)]:
        add("long_" + tag, "long_crossborder", o, [(-803.25, y), (803.25, y)], tag == "b")
    o = center((-103.04163, 36.445))
    for tag, y in [("a", 0), ("b", 8)]:
        add("short_" + tag, "short_split", o, [(-151.625, y), (151.625, y)], tag == "b")
    o = center((-103.04166, 36.455))
    for tag, y in [("a", 0), ("b", 8)]:
        add("asymmetric_" + tag, "asymmetric_split", o, [(-337.25, y), (137.25, y)], tag == "b")
    o = center((-103.04168, 36.466))
    for tag, y, start in [("a", 0, -703.25), ("b", 6, -703.25), ("c", 13, -143.25)]:
        add("trio_" + tag, "joining_trio", o, [(start, y), (703.25, y)], tag == "c")
    o = center((-103.04171, 36.482))
    add("rejoin_a", "diverge_rejoin", o, [(-1603.25, 0), (1603.25, 0)])
    add("rejoin_b", "diverge_rejoin", o, [(-1603.25, 8), (-903.25, 8), (-503.25, 158),
                                              (203.25, 158), (603.25, 8), (1603.25, 8)], True)
    o = center((-103.04161, 36.425))
    for tag, x in [("nm", -5.5), ("tx", 5.5)]:
        add("opposite_" + tag, "opposite_state_pair", o, [(x, -353.25), (x, 353.25)], tag == "tx")
    o = center((-102.95, 36.50035), ("TX", "OK"))
    add("phase_a", "phase_tails_reverse", o, [(0, -1003.25), (0, 903.25)])
    add("phase_b", "phase_tails_reverse", o, [(8, -1001.1), (8, 907.6)], True, True)
    add("phase_spur", "phase_branch", o, [(0, 700), (500, 1100), (1100, 1230)], True)
    # Three counted feeders and a loop connect gathering motifs without hidden
    # corridor/reference overlays. All incidental interactions remain in reference.
    for state, origin in {"NM": (-103.08, 36.445), "TX": (-103.01, 36.457), "OK": (-102.926, 36.519)}.items():
        add(state.lower() + "_feeder", state.lower() + "_feeder", origin,
            [(0, 0), (450, 750), (750, 1100), (500, 1750)], True)
    o = (-103.01, 36.476)
    add("tx_loop", "tx_loop", o, [(0, 0), (650, 0), (650, 550), (0, 550), (0, 0)], True)
    return out, {"states": ["NM", "TX", "OK"], "boundary_edges": edges,
                 "intent": "Seven labeled cross-border matching motifs; shared mileage must be zero. Nonaligned phase and opposite directions are intentional."}


def fixture_four(states):
    a, b = (-103.064732, 32.744215), (-103.064732, 32.75427)
    provenance = edge_provenance(states, ("NM", "TX"), a, b)
    if not states["NM"].boundary.covers(LineString([a, b])) or not states["TX"].boundary.covers(LineString([a, b])):
        raise ValueError("Selected meridian is not a complete common native edge")
    def p(x, y):
        on_border = destination(a, 0, y)
        # Canonical longitude stays bit-for-bit exact, including generated
        # redundant vertices; a meridian is both linear and geodesic here.
        on_border = (a[0], on_border[1])
        return on_border if x == 0 else destination(on_border, 90 if x > 0 else 270, abs(x))
    def path(points):
        return [p(x, y) for x, y in points]
    out = [
        source("04_interior_shared_interior", "interior_shared_interior", [path([(-160, 50), (0, 50), (0, 370), (160, 370)])]),
        source("04_shared_long", "shared_long", [path([(0, 20), (0, 450)])]),
        source("04_near_east", "near_parallel", [path([(8, 20), (8, 450)])]),
        source("04_near_west", "near_parallel", [path([(-8, 20), (-8, 450)])]),
        source("04_centimeter_east", "centimeter_exclusive", [path([(0.03, 20), (0.03, 450)])]),
        source("04_centimeter_west", "centimeter_exclusive", [path([(-0.03, 20), (-0.03, 450)])]),
        source("04_shared_qualification_route", "shared_cannot_qualify_state", [path([(4, 540), (4, 600), (0, 600), (0, 860), (4, 860), (4, 920)])]),
        source("04_shared_qualification_partner", "shared_cannot_qualify_state", [path([(12, 540), (12, 920)])]),
        source("04_shared_short", "shared_short", [path([(0, 975), (0, 1098.25)])]),
        source("04_tiny_crossing", "tiny_crossing", [path([(-0.04, 487), (0.06, 487)])]),
        source("04_endpoint_touch", "endpoint_touch", [path([(-90, 510), (0, 510)])]),
    ]
    return out, {"states": ["NM", "TX"], "boundary_edges": [{"states": ["NM", "TX"], "segment_start": a, "segment_end": b,
                       "native_edges": provenance}], "intent": "Exact meridian sharing, 3 cm exclusive offsets, shared allocation at all lengths, and 10 cm crossing.",
                 "size_exception": "A deliberately compact 11-source meridian case keeps centimeter and sub-threshold controls legible; the first three files carry larger networks."}


def write_kmz(path, sources, title):
    root = ET.Element(f"{{{KML}}}kml")
    document = ET.SubElement(root, f"{{{KML}}}Document")
    ET.SubElement(document, f"{{{KML}}}name").text = title
    for index, item in enumerate(sources):
        placemark = ET.SubElement(document, f"{{{KML}}}Placemark", {"id": "p_" + item["key"]})
        ET.SubElement(placemark, f"{{{KML}}}name").text = item["name"]
        ext = ET.SubElement(placemark, f"{{{KML}}}ExtendedData")
        objectid = item.get("objectid", "7" if index % 4 == 0 else str(index % 9))
        for key, value in [("fixture_key", item["key"]), ("motif", item["motif"]), ("OBJECTID", objectid)]:
            data = ET.SubElement(ext, f"{{{KML}}}Data", {"name": key})
            ET.SubElement(data, f"{{{KML}}}value").text = value
        parent = ET.SubElement(placemark, f"{{{KML}}}MultiGeometry") if len(item["paths"]) > 1 else placemark
        for pi, coordinates in enumerate(item["paths"]):
            ls = ET.SubElement(parent, f"{{{KML}}}LineString")
            ET.SubElement(ls, f"{{{KML}}}tessellate").text = "1"
            # Vary altitude to expose accidental 3D mileage while deliberately
            # preserving the frozen two-dimensional geodesic accounting model.
            ET.SubElement(ls, f"{{{KML}}}coordinates").text = " ".join(
                f"{lon:.14f},{lat:.14f},{(vi * 17 + pi * 3) % 101}" for vi, (lon, lat) in enumerate(coordinates))
    ET.indent(root, space="  ")
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    entry = zipfile.ZipInfo("doc.kml", date_time=(2025, 1, 1, 0, 0, 0))
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_saved(path):
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("doc.kml"))
    result = []
    for pm in root.findall(f".//{{{KML}}}Placemark"):
        fields = {d.attrib["name"]: d.findtext(f"{{{KML}}}value") for d in pm.findall(f".//{{{KML}}}Data")}
        paths = [[tuple(map(float, token.split(",")[:2])) for token in ls.findtext(f"{{{KML}}}coordinates").split()]
                 for ls in pm.findall(f".//{{{KML}}}LineString")]
        result.append({"key": fields["fixture_key"], "motif": fields["motif"], "paths": paths,
                       "objectid": fields["OBJECTID"], "name": pm.findtext(f"{{{KML}}}name")})
    return result


def preview(path, sources, states, description, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab20")
    if name.startswith("01"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
        panels = [(ax, [s for s in sources if s["key"].startswith("01_" + state.lower())], state) for ax, state in zip(axes, ["TX", "LA", "WY"])]
    elif name.startswith("04"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
        panels = [(axes[0], sources, "Whole verified meridian"), (axes[1], sources, "Near-border detail (transverse axis expanded)")]
    else:
        fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
        panels = [(ax, sources, " / ".join(description["states"]))]
    for ax, items, label in panels:
        coords = [v for s in items for p in s["paths"] for v in p]
        xmin, xmax = min(p[0] for p in coords), max(p[0] for p in coords)
        ymin, ymax = min(p[1] for p in coords), max(p[1] for p in coords)
        dx, dy = max(xmax - xmin, 0.002), max(ymax - ymin, 0.002)
        bounds = box(xmin - dx * .06, ymin - dy * .06, xmax + dx * .06, ymax + dy * .06)
        for code in description["states"]:
            edge = states[code].boundary.intersection(bounds)
            components = list(edge.geoms) if hasattr(edge, "geoms") else [edge]
            for component in components:
                if component.geom_type == "LineString":
                    x, y = component.xy
                    ax.plot(x, y, color="#202830", linewidth=1.6, linestyle="--", zorder=1)
            region = states[code].intersection(bounds)
            if not region.is_empty and not name.startswith("01"):
                label_point = region.representative_point()
                ax.text(label_point.x, label_point.y, code, color="#657181", fontsize=16,
                        alpha=.75, ha="center", va="center", zorder=0, clip_on=True)
        motif_centers = {}
        for i, item in enumerate(items):
            for coordinates in item["paths"]:
                ax.plot([c[0] for c in coordinates], [c[1] for c in coordinates], linewidth=1.6, color=colors(i % 20), zorder=2)
            motif_centers.setdefault(item["motif"], item["paths"][0][0])
        for i, (motif, xy) in enumerate(motif_centers.items()):
            text = motif.replace("_", " ")
            ax.annotate(text, xy, xytext=(5, 5 + 9 * (i % 2)), textcoords="offset points", fontsize=7,
                        bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": 1})
        ax.set_title(label)
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.ticklabel_format(axis="both", useOffset=False, style="plain")
        ax.grid(alpha=.2)
        if name.startswith("04"):
            if ax is axes[1]:
                ax.set_xlim(-103.06491, -103.06455)
                ax.set_title("Near-border controls; 3 cm lines visually coincide at this scale")
        else:
            ax.set_aspect(1 / math.cos(math.radians((ymin + ymax) / 2)))
    fig.suptitle(name.replace("_", " ") + "\nSynthetic pipelines; dashed lines are canonical state borders, preview only", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, metadata={"Software": "pipeline-kmz-fixture-generator " + GENERATOR_VERSION})
    plt.close(fig)


def control_preview(output, sources, states, description, name):
    """Separate inspection maps resolve meter offsets without adding KML overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    if name.startswith("01"):
        panels = [("TX long pair", "tx_long_pair"), ("TX compatible trio", "tx_clique_trio"),
                  ("TX non-clique chain", "tx_chain_nonclique"), ("TX disconnected sub-threshold paths", "tx_multipart_short")]
    elif name.startswith("03"):
        panels = [("Short split: Combined qualification only", "short_split"),
                  ("Asymmetric exclusive lengths", "asymmetric_split"),
                  ("Member joins near border", "joining_trio"),
                  ("Separate lines on opposite sides", "opposite_state_pair"),
                  ("Diverging and rejoining runs", "diverge_rejoin"),
                  ("Nonaligned starts; reversed dense source", "phase_tails_reverse")]
    elif name.startswith("04"):
        panels = [("3 cm exclusive offsets and true shared line", "centimeter_exclusive"),
                  ("Shared geometry cannot qualify state interiors", "shared_cannot_qualify_state"),
                  ("A real 10 cm crossing", "tiny_crossing"), ("Short shared interval still allocates", "shared_short")]
    if not panels:
        return
    fig, axes = plt.subplots(math.ceil(len(panels) / 2), 2, figsize=(14, 4.7 * math.ceil(len(panels) / 2)), constrained_layout=True)
    for ax, (title, motif) in zip(axes.flat, panels):
        items = [s for s in sources if s["motif"] == motif]
        if motif == "centimeter_exclusive":
            items += [s for s in sources if s["key"] == "04_shared_long"]
        coords = [v for s in items for p in s["paths"] for v in p]
        origin = coords[0]
        def xy(point):
            bearing, _, distance = GEOD.inv(*origin, *point)
            return distance * math.sin(math.radians(bearing)), distance * math.cos(math.radians(bearing))
        lon_margin, lat_margin = .00004, .00004
        area = box(min(p[0] for p in coords) - lon_margin, min(p[1] for p in coords) - lat_margin,
                   max(p[0] for p in coords) + lon_margin, max(p[1] for p in coords) + lat_margin)
        for code in description["states"]:
            border = states[code].boundary.intersection(area)
            parts = list(border.geoms) if hasattr(border, "geoms") else [border]
            for part in parts:
                if part.geom_type == "LineString":
                    values = [xy(p) for p in part.coords]
                    ax.plot([p[0] for p in values], [p[1] for p in values], "--", color="#414a55", linewidth=1.2)
        for item in items:
            for pi, points in enumerate(item["paths"]):
                # A visually useful geodesic trace, not a coordinate-linear
                # substitute for the geometry reference's independent crossings.
                values = [xy(p) for p in geodesic_vertices(points, 20)]
                ax.plot([p[0] for p in values], [p[1] for p in values], linewidth=2,
                        label=item["key"] if pi == 0 else None)
        if motif == "centimeter_exclusive":
            ax.set_xlim(-.085, .025)
        elif motif == "tiny_crossing":
            ax.set_xlim(-.025, .125)
            ax.set_ylim(-.02, .02)
        elif motif == "shared_short":
            ax.set_xlim(-.1, .1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("East from first source start (m)")
        ax.set_ylabel("North from first source start (m)")
        ax.ticklabel_format(axis="both", useOffset=False, style="plain")
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=7, framealpha=.8)
    fig.suptitle(name[:2] + " — named control details\nAxes use independent scales to expose narrow separations; dashed lines are canonical borders", fontsize=13)
    path = output / "previews/details" / (name[:2] + "_control_details.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, metadata={"Software": "pipeline-kmz-fixture-generator " + GENERATOR_VERSION})
    plt.close(fig)


def design_assertions(name):
    common = ["No unresolved or outside mileage; per-source and total original conservation within max(0.001 m, original*1e-10).",
              "Stable fixture keys remain distinct despite repeated names and OBJECTID fields."]
    if name.startswith("01"):
        return common + ["TX/LA/WY represented with zero crossings and zero shared length; Combined savings equals sum of state savings.",
            "In each state: long_pair savings>0; clique_trio has three qualified source pairs and disjoint groups; chain_nonclique has exactly two qualified source pairs and no 3-member group.",
            "In each state: multipart_short paths each below 200 m, combined above 200 m, and no qualifying sections; perpendicular/isolated/loop sources have zero cross-pipeline qualifying coverage."]
    if name.startswith("02"):
        return common + ["Three states represented; every border crossing transverse; shared length zero.",
            "three_state_route visits NM,TX,OK; sparse first edge contains an interior crossing; reentry has at least three transitions.",
            "endpoint_touch has zero TX positive length; short_visit retains two crossings and a positive OK interval.",
            "Interior pair in each state qualifies without cross-border matching near clipped crossings."]
    if name.startswith("03"):
        return common + ["Shared length zero; all seven motifs analyzed independently and every source included globally.",
            "long_crossborder qualifies Combined and both exclusive states; short_split qualifies Combined and neither state.",
            "asymmetric_split qualifies Combined and NM only; joining_trio qualifies three pairs Combined and three pairs in TX, but only A/B in NM.",
            "diverge_rejoin has at least two distinct qualified sections; opposite_state_pair qualifies Combined but never state savings.",
            "phase_tails_reverse covers nonaligned starts, nonmultiple tails, opposite digitization, and dense/sparse original vertices."]
    return common + ["Shared meridian intervals have exactly NM/TX ownership, allocation half to each; shared_short is allocated despite length below 200 m.",
        "3 cm east/west offsets remain exclusive TX/NM respectively, with zero shared length.",
        "shared_cannot_qualify_state pair qualifies Combined but its route/partner has zero qualified state sections.",
        "tiny_crossing retains positive intervals in both states; endpoint_touch has no TX positive interval.",
        "Reverse and redundant-geodesic-vertex variants preserve original and allocated geography within certified bounds; savings must be recomputed because sampling phase can change."]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUITE, help="Suite output directory; preserves the organized fixtures/previews/validation layout.")
    parser.add_argument("--profile", choices=["core", "stress"], default="core")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--stress-groups", type=int, default=24, help="Number of isolated branching four-source stress groups (profile stress).")
    args = parser.parse_args(argv)
    states, manifest = load_boundaries()
    output = args.output.resolve()
    if args.profile == "stress":
        sources = []
        for group in range(args.stress_groups):
            origin = local((-100, 31), (group % 6) * 2400, (group // 6) * 2200)
            for member, offset in enumerate([0, 5, 12]):
                sources.append(source(f"stress_{group:03d}_{member}", f"stress_group_{group:03d}",
                    [line(origin, [(offset, 0), (offset, 1203.25), (offset + 200, 1503.25)], member == 2)]))
            sources.append(source(f"stress_{group:03d}_branch", f"stress_group_{group:03d}",
                [line(origin, [(0, 650), (-300, 850), (-650, 1350)])]))
        path = output / "fixtures/stress/stress_branching_network.kmz"
        digest = write_kmz(path, sources, "Deterministic branching overlap stress input")
        print(json.dumps({"profile": "stress", "file": str(path), "sha256": digest, "groups": args.stress_groups,
                          "source_count": len(sources), "note": "Generated input only; successful analysis, runtime and memory require separate measurement."}))
        return
    makers = [("01_three_state_disconnected_networks", lambda: fixture_one()),
              ("02_three_state_transverse_crossings", lambda: fixture_two(states)),
              ("03_parallel_corridors_crossing_borders", lambda: fixture_three(states)),
              ("04_shared_border_and_near_border", lambda: fixture_four(states))]
    report = {"schema_version": "pipeline-kmz-design/1", "generator_version": GENERATOR_VERSION,
              "random_seed": SEED, "baseline_commit": BASELINE, "boundary_sha256": BOUNDARY_SHA256,
              "boundary_manifest": manifest, "boundary_resource": "src/pipeline_calculator/data/states_2025.zip",
              "coordinate_serialization_decimal_places": 14, "zip_timestamp": "2025-01-01T00:00:00",
              "longitude_wrapping": "Construction uses Geod longitude normalization. Core fixture edges never cross the antimeridian; the reference handles wrapping explicitly.",
              "expectation_policy": "This design manifest states intended controls, not numerical goldens. Independent references must measure the saved KMZ and confirm every assertion.",
              "fixtures": []}
    for name, make in makers:
        sources, description = make()
        path = output / "fixtures" / (name + ".kmz")
        digest = write_kmz(path, sources, name)
        saved = read_saved(path)
        if not args.no_previews:
            preview(output / "previews" / (name + ".png"), saved, states, description, name)
            control_preview(output, saved, states, description, name)
        entry = {"name": name, "file": path.relative_to(output).as_posix(), "sha256": digest,
                 "description": description, "source_count": len(sources), "path_count": sum(len(s["paths"]) for s in saved),
                 "vertex_count": sum(len(p) for s in saved for p in s["paths"]),
                 "sources": [{"key": s["key"], "name": s["name"], "motif": s["motif"], "xml_order": i,
                              "path_count": len(s["paths"]), "design_state_hint": sources[i].get("state_hint"),
                              "design_note": sources[i].get("note", "")} for i, s in enumerate(saved)],
                 "planned_assertions": design_assertions(name), "variants": []}
        if name.startswith("04"):
            for mode in ["reversed", "redundant_vertices"]:
                variant = copy.deepcopy(saved)
                for s in variant:
                    s["paths"] = [list(reversed(p)) if mode == "reversed" else geodesic_vertices(p, 19.0) for p in s["paths"]]
                vp = output / "fixtures/variants" / (name + "__" + mode + ".kmz")
                vd = write_kmz(vp, variant, name + " " + mode)
                entry["variants"].append({"kind": mode, "file": vp.relative_to(output).as_posix(), "sha256": vd})
        if name.startswith("01"):
            variant = copy.deepcopy(saved)
            random.Random(SEED).shuffle(variant)
            vp = output / "fixtures/variants" / (name + "__source_order.kmz")
            vd = write_kmz(vp, variant, name + " shuffled source order")
            entry["variants"].append({"kind": "source_order", "file": vp.relative_to(output).as_posix(), "sha256": vd})
        report["fixtures"].append(entry)
        print(json.dumps({"file": entry["file"], "sha256": digest, "source_count": len(saved), "path_count": entry["path_count"]}))
    report_path = output / "validation/design_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
