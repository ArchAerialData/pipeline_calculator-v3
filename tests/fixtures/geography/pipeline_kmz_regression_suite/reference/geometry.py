"""Independent, serialized-input geographic accounting for the KMZ suite.

No pipeline_calculator imports. GeographicLib is the primary distance engine;
PROJ/pyproj is a second implementation audit. Intersections solve a native
longitude/latitude boundary line equation along the ellipsoidal source curve.
This deliberately does not use the application's radial projection method.

The search bounds coordinate curvature from the geodesic differential equations.
For a twice differentiable coordinate with |x''| <= C, its straight interpolant
has error <= C*h*h/8 on a station interval of length h. That bound both expands
candidate envelopes and rejects boundary lines that cannot meet a source curve.
Non-transverse near tangencies fail closed rather than becoming golden values.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from geographiclib.geodesic import Geodesic
import numpy as np
from pyproj import Geod
from shapely import from_wkb, linestrings
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree

SURVEY_MILE = 1609.347218694
BOUNDARY_SHA256 = "55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539"
GRS80_A = 6378137.0
GRS80_F = 1 / 298.257222101
GEODESIC = Geodesic(GRS80_A, GRS80_F)
SECOND_GEODESIC = Geod(ellps="GRS80")
ROOT_WIDTH_M = 0.00000025
NUMERICAL_FLOOR_M = 0.000002
COORDINATE_EVALUATION_ALLOWANCE_DEGREES = 2e-12
NS = {"k": "http://www.opengis.net/kml/2.2"}


def read_kmz(path):
    """Parse the actual single-document KMZ and preserve source/path identities."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["doc.kml"]:
            raise ValueError(f"{path.name}: suite inputs must contain only doc.kml")
        root = ET.fromstring(archive.read("doc.kml"))
    for forbidden in ("NetworkLink", "Polygon", "Point", "Model", "GroundOverlay"):
        if root.findall(f".//k:{forbidden}", NS):
            raise ValueError(f"Input contains forbidden geometry or overlay: {forbidden}")
    sources, keys, ids = [], set(), set()
    for order, feature in enumerate(root.findall(".//k:Placemark", NS)):
        metadata = {node.attrib["name"]: node.findtext("k:value", default="", namespaces=NS)
                    for node in feature.findall("k:ExtendedData/k:Data", NS)}
        metadata.update({node.attrib["name"]: node.text or ""
                         for node in feature.findall("k:ExtendedData/k:SchemaData/k:SimpleData", NS)})
        key, xml_id = metadata.get("fixture_key"), feature.attrib.get("id")
        if not key or key in keys:
            raise ValueError("Missing or duplicate stable fixture_key")
        if not xml_id or xml_id in ids or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", xml_id):
            raise ValueError("Main fixture XML IDs must be valid and unique")
        keys.add(key)
        ids.add(xml_id)
        paths = []
        for line in feature.findall(".//k:LineString", NS):
            values = (line.findtext("k:coordinates", default="", namespaces=NS)).split()
            coordinates = [tuple(map(float, token.split(",")[:2])) for token in values]
            if len(coordinates) < 2 or any(len(p) != 2 or not all(map(math.isfinite, p))
                                          or not -180 <= p[0] <= 180 or not -90 <= p[1] <= 90
                                          for p in coordinates):
                raise ValueError(f"Invalid serialized path in {key}")
            paths.append(coordinates)
        if not paths:
            raise ValueError(f"Non-pipeline Placemark: {key}")
        sources.append({"key": key, "name": feature.findtext("k:name", default="", namespaces=NS),
                        "motif": metadata.get("motif", ""), "xml_order": order,
                        "xml_id": xml_id, "objectid": metadata.get("OBJECTID", ""),
                        "paths": paths, "metadata": metadata})
    return sources


def _parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _parts(part)


def _ring_coordinates(ring, longitude_anchor=None):
    """Keep consecutive longitude differences in [-180,180], including holes."""
    points = []
    for lon, lat, *_ in ring.coords:
        if points:
            lon += 360 * round((points[-1][0] - lon) / 360)
        points.append((lon, lat))
    if longitude_anchor is not None:
        shift = 360 * round((longitude_anchor - sum(p[0] for p in points) / len(points)) / 360)
        points = [(lon + shift, lat) for lon, lat in points]
    return points


def distance(a, b):
    return GEODESIC.Inverse(a[1], a[0], b[1], b[0])["s12"]


class _SourceEdge:
    def __init__(self, start, end):
        self.start, self.end = tuple(start), tuple(end)
        inverse = GEODESIC.Inverse(start[1], start[0], end[1], end[0])
        self.length, self.azimuth = inverse["s12"], inverse["azi1"]
        self.line = GEODESIC.Line(start[1], start[0], self.azimuth)
        self._points = {0.0: self.start, self.length: self._near(self.end)}
        # |phi'| <= 1/M_min bounds all intervening latitudes, not just endpoints.
        eccentricity2 = GRS80_F * (2 - GRS80_F)
        m_min = GRS80_A * (1 - eccentricity2)
        latitude_limit = abs(math.radians(start[1])) + self.length / m_min
        if latitude_limit >= math.radians(85) or self.length > 200_000:
            raise ValueError("Reference certification domain: nonpolar edges <= 200 km")
        tangent, cosine = math.tan(latitude_limit), math.cos(latitude_limit)
        n_derivative = eccentricity2 / (2 * (1 - eccentricity2))
        m_derivative = 3 * n_derivative
        # phi'=cos(alpha)/M; lambda'=sin(alpha)/(N*cos(phi));
        # alpha'=sin(alpha)*tan(phi)/N. All bounds use N>=a, M>=M_min.
        self.lat_curvature = math.degrees(tangent / (GRS80_A * m_min)
                                         + m_derivative / m_min**2)
        self.lon_curvature = math.degrees(tangent / (GRS80_A**2 * cosine)
                                         + (n_derivative + tangent) / (m_min * GRS80_A * cosine))

    def _near(self, coordinate):
        return (coordinate[0] + 360 * round((self.start[0] - coordinate[0]) / 360), coordinate[1])

    def point(self, station):
        if station not in self._points:
            point = self.line.Position(station, Geodesic.LATITUDE | Geodesic.LONGITUDE | Geodesic.LONG_UNROLL)
            self._points[station] = (point["lon2"], point["lat2"])
        return self._points[station]

    def bounds(self):
        n = max(1, math.ceil(self.length / 500))
        points = [self.point(self.length * i / n) for i in range(n + 1)]
        span = self.length / n
        # A floating-coordinate evaluation allowance expands search only; it
        # does not snap a source point or change ownership.
        dx = self.lon_curvature * span**2 / 8 + COORDINATE_EVALUATION_ALLOWANCE_DEGREES
        dy = self.lat_curvature * span**2 / 8 + COORDINATE_EVALUATION_ALLOWANCE_DEGREES
        return (min(p[0] for p in points) - dx, min(p[1] for p in points) - dy,
                max(p[0] for p in points) + dx, max(p[1] for p in points) + dy)


class GeometryReference:
    def __init__(self, boundary_path):
        path = Path(boundary_path)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != BOUNDARY_SHA256:
            raise ValueError(f"Boundary resource checksum mismatch: {actual_hash}")
        self.polygons, self.polygon_states, self.state_names = [], [], {}
        with zipfile.ZipFile(path) as archive:
            self.manifest = json.loads(archive.read("manifest.json"))
            if self.manifest.get("schema_version") != 1:
                raise ValueError("Unsupported boundary manifest")
            for entry in self.manifest["states"]:
                raw = archive.read(entry["file"])
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError(f"WKB checksum mismatch: {entry['code']}")
                self.state_names[entry["code"]] = entry["name"]
                for part in _parts(from_wkb(raw)):
                    exterior = _ring_coordinates(part.exterior)
                    anchor = sum(p[0] for p in exterior) / len(exterior)
                    holes = [_ring_coordinates(ring, anchor) for ring in part.interiors]
                    polygon = Polygon(exterior, holes)
                    if not polygon.is_valid:
                        raise ValueError("Invalid unwrapped boundary polygon; no automatic repair allowed")
                    self.polygons.append(polygon)
                    self.polygon_states.append(entry["code"])
        if len(self.state_names) != 51 or "DC" not in self.state_names:
            raise ValueError("Incomplete boundary resource")
        self.tree = STRtree(self.polygons)
        self.edge_trees = {}
        self.provenance = dict(self.manifest, resource_sha256=actual_hash)

    def states_at(self, coordinate):
        lon, lat = coordinate
        # Keep canonical native coordinates bit-for-bit. Applying modulo even
        # inside the canonical range can move an exact boundary point by an ulp
        # (for example, 10.1 becomes 10.099999999999994).
        if not -180 <= lon <= 180:
            lon -= 360 * math.floor((lon + 180) / 360)
        result = set()
        for shift in (-360, 0, 360):
            for i in self.tree.query(Point(lon + shift, lat), predicate="intersects"):
                result.add(self.polygon_states[int(i)])
        return sorted(result)

    def _candidate_boundaries(self, bounds):
        collected = {}
        for shift in (-360, 0, 360):
            query = box(bounds[0] + shift, bounds[1], bounds[2] + shift, bounds[3])
            for index in self.tree.query(query):
                index = int(index)
                if index not in self.edge_trees:
                    part = self.polygons[index]
                    arrays = [np.asarray(ring.coords)[:, :2] for ring in [part.exterior, *part.interiors]]
                    segments = np.concatenate([np.stack((a[:-1], a[1:]), axis=1) for a in arrays])
                    self.edge_trees[index] = (segments, STRtree(linestrings(segments)))
                segments, tree = self.edge_trees[index]
                for position in tree.query(query):
                    a, b = segments[int(position)]
                    first, last = sorted(((float(a[0]) - shift, float(a[1])),
                                          (float(b[0]) - shift, float(b[1]))))
                    if first != last:
                        collected.setdefault((first, last), set()).add(self.polygon_states[index])
        return [(a, b, sorted(codes)) for (a, b), codes in sorted(collected.items())]

    @staticmethod
    def _shared(edge, a, b):
        p, q = edge.point(0), edge.point(edge.length)
        axis = 1 if p[0] == q[0] == a[0] == b[0] else (
            0 if p[1] == q[1] == a[1] == b[1] == 0 else None)
        if axis is None:
            return None
        left = max(min(p[axis], q[axis]), min(a[axis], b[axis]))
        right = min(max(p[axis], q[axis]), max(a[axis], b[axis]))
        if right <= left:
            return None
        def at(value):
            if value == p[axis]:
                return 0.0
            if value == q[axis]:
                return edge.length
            coordinate = (p[0], value) if axis == 1 else (value, 0.0)
            return distance(p, coordinate)
        return sorted((at(left), at(right)))

    @staticmethod
    def _roots(edge, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        nx, ny = -dy / norm, dx / norm
        curvature = abs(nx) * edge.lon_curvature + abs(ny) * edge.lat_curvature
        evaluation_allowance = (abs(nx) + abs(ny)) * COORDINATE_EVALUATION_ALLOWANCE_DEGREES
        def endpoint_value(p):
            # Native endpoint identities are exact binary coordinates. An
            # interpolated point rounded slightly across a boundary is not an
            # endpoint touch, even when the excursion is less than an ulp of
            # its path's accumulated station.
            with localcontext() as context:
                context.prec = 80
                d = Decimal.from_float
                determinant = ((d(b[0]) - d(a[0])) * (d(p[1]) - d(a[1]))
                               - (d(b[1]) - d(a[1])) * (d(p[0]) - d(a[0])))
                return float(determinant) / norm
        def value(s):
            p = edge.point(s)
            if s == 0.0 or s == edge.length:
                return endpoint_value(p)
            return nx * (p[0] - a[0]) + ny * (p[1] - a[1])
        results = []
        def accept(lo, hi, coarse_mid=None):
            station = (lo + hi) / 2
            point = edge.point(station)
            fraction = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / norm**2
            # Boundary-segment parameter verification uses only a numerical
            # allowance compatible with the cut's station uncertainty.
            if -1e-10 <= fraction <= 1 + 1e-10:
                if abs(fraction - min(1, max(0, fraction))) * norm > 1e-12:
                    raise ValueError("Root too close to a boundary vertex to certify membership")
                station_allowance = NUMERICAL_FLOOR_M
                if hi > lo:
                    # Coordinate rounding becomes a larger station error at
                    # shallow crossings. A fixed micrometer station allowance
                    # would falsely certify these ill-conditioned roots.
                    probe_lo, probe_hi = max(0.0, station - .5), min(edge.length, station + .5)
                    probe_width = probe_hi - probe_lo
                    secant = abs((value(probe_hi) - value(probe_lo)) / probe_width)
                    derivative_lower_bound = (secant - curvature * probe_width
                                              - 2 * evaluation_allowance / probe_width)
                    if derivative_lower_bound <= 0:
                        raise ValueError("Shallow root has no certified derivative lower bound")
                    station_allowance = max(station_allowance, evaluation_allowance / derivative_lower_bound)
                    if (hi - lo) / 2 + station_allowance > .01:
                        raise ValueError("Shallow root uncertainty exceeds the 0.01 m cut target")
                results.append({"station": station, "low": lo, "high": hi,
                                "error_meters": (hi - lo) / 2 + station_allowance,
                                "convergence_change_meters": abs(station - coarse_mid) if coarse_mid is not None else 0.0,
                                "boundary_coordinates": [list(a), list(b)]})
        p, q = edge.point(0), edge.point(edge.length)
        axis = 1 if p[0] == q[0] == a[0] == b[0] else (
            0 if p[1] == q[1] == a[1] == b[1] == 0 else None)
        if axis is not None:
            # _partition_edge handles positive exact coincidence separately.
            # A collinear segment beyond the source endpoint may merely touch
            # the source, or miss it entirely; neither implies an unproved run.
            left = max(min(p[axis], q[axis]), min(a[axis], b[axis]))
            right = min(max(p[axis], q[axis]), max(a[axis], b[axis]))
            if right < left:
                return []
            if right == left:
                station = 0.0 if left == p[axis] else edge.length
                accept(station, station)
                return results
            raise ValueError("Positive exact coincidence must be partitioned as an interval")
        def inspect(lo, vlo, hi, vhi, depth=0):
            width = hi - lo
            if vlo == vhi == 0:
                raise ValueError("Unproved positive coincidence with a nonmeridian boundary")
            slope = (vhi - vlo) / width
            monotonic = abs(slope) > curvature * width + 2 * evaluation_allowance / width
            if monotonic:
                if vlo == 0:
                    accept(lo, lo)
                if vhi == 0:
                    accept(hi, hi)
                if vlo * vhi >= 0:
                    return
                left, right, vl = lo, hi, vlo
                coarse = None
                while right - left > ROOT_WIDTH_M:
                    middle = (left + right) / 2
                    vm = value(middle)
                    if vm == 0:
                        # Coordinate evaluation rounding can make the exact
                        # root zero; retain a conservative station bracket.
                        left = max(left, middle - ROOT_WIDTH_M / 2)
                        right = min(right, middle + ROOT_WIDTH_M / 2)
                        break
                    if vl * vm < 0:
                        right = middle
                    else:
                        left, vl = middle, vm
                    if coarse is None and right - left <= 0.000004:
                        coarse = (left + right) / 2
                accept(left, right, coarse)
                return
            # Interpolation error bounds the entire curve between endpoints.
            if vlo * vhi > 0 and min(abs(vlo), abs(vhi)) > curvature * width**2 / 8 + evaluation_allowance:
                return
            if width <= 0.00001 or depth > 60:
                raise ValueError("Non-transverse root cannot be certified; redesign the fixture")
            middle = (lo + hi) / 2
            vm = value(middle)
            inspect(lo, vlo, middle, vm, depth + 1)
            inspect(middle, vm, hi, vhi, depth + 1)
        inspect(0.0, value(0.0), edge.length, value(edge.length))
        return results

    @staticmethod
    def _same_root_locus(first, second):
        """Only conflate cuts whose geometry proves one boundary event.

        Overlapping numerical brackets are not evidence of identity: distinct
        close boundaries can enclose a genuine short visit. Exact source
        endpoint roots and identical supporting boundary lines are provable.
        Other indistinguishable cuts fail closed below.
        """
        if ((first.get("source_endpoint") or second.get("source_endpoint"))
                and first["low"] == first["high"] == second["low"] == second["high"]):
            return True
        a = first.get("boundary_coordinates")
        b = second.get("boundary_coordinates")
        if not a or not b:
            return False
        with localcontext() as context:
            context.prec = 80
            d = Decimal.from_float
            dx, dy = d(a[1][0]) - d(a[0][0]), d(a[1][1]) - d(a[0][1])
            return all(dx * (d(y) - d(a[0][1])) == dy * (d(x) - d(a[0][0])) for x, y in b)

    def _partition_edge(self, edge):
        cuts = [{"station": 0.0, "low": 0.0, "high": 0.0, "error_meters": 0.0, "source_endpoint": True},
                {"station": edge.length, "low": edge.length, "high": edge.length,
                 "error_meters": 0.0, "source_endpoint": True}]
        shared = []
        for a, b, codes in self._candidate_boundaries(edge.bounds()):
            coincident = self._shared(edge, a, b)
            if coincident:
                lo, hi = coincident
                shared.append((lo, hi, codes))
                for station in (lo, hi):
                    cuts.append({"station": station, "low": station, "high": station,
                                 "error_meters": NUMERICAL_FLOOR_M,
                                 "boundary_coordinates": [list(a), list(b)], "boundary_states": codes})
            else:
                for cut in self._roots(edge, a, b):
                    cut["boundary_states"] = codes
                    cuts.append(cut)
        ordered = []
        for cut in sorted(cuts, key=lambda c: c["station"]):
            if cut["high"] > cut["low"] and (cut["low"] == 0.0 or cut["high"] == edge.length):
                raise ValueError("Root uncertainty overlaps a source endpoint without exact boundary identity; redesign the endpoint")
            previous = ordered[-1] if ordered else None
            overlapping_brackets = previous and (previous["station"] == cut["station"] or (
                previous["high"] > previous["low"] and cut["high"] > cut["low"]
                and max(previous["low"], cut["low"]) <= min(previous["high"], cut["high"])))
            same_event = overlapping_brackets and self._same_root_locus(previous, cut)
            if same_event:
                if "boundary_coordinates" not in previous and "boundary_coordinates" in cut:
                    previous.update({k: v for k, v in cut.items() if k.startswith("boundary_")})
                previous["error_meters"] = max(previous["error_meters"],
                                              abs(previous["station"] - cut["station"]) + cut["error_meters"])
                continue
            if previous and (cut["station"] - previous["station"] <=
                             cut["error_meters"] + previous["error_meters"]):
                raise ValueError("Distinct boundary roots cannot be ordered within certified uncertainty; "
                                 "the intervening interval must not be discarded")
            ordered.append(cut)
        intervals = []
        for left, right in zip(ordered, ordered[1:]):
            lo, hi = left["station"], right["station"]
            if hi <= lo:
                continue
            midpoint = (lo + hi) / 2
            adjoining = sorted({code for a, b, codes in shared if a < midpoint < b for code in codes})
            codes = self.states_at(edge.point(midpoint))
            if len(adjoining) > 1:
                kind, codes = "shared", adjoining
            elif len(codes) == 1:
                kind = "interior"
            elif not codes:
                kind = "outside"
            else:
                kind = "unresolved"
            intervals.append({"local_start_m": lo, "local_end_m": hi, "kind": kind,
                              "state_codes": codes, "length_meters": hi - lo,
                              "coordinates": [list(edge.point(lo)), list(edge.point(hi))],
                              "start_cut": left, "end_cut": right})
        return intervals

    def analyze(self, sources):
        ledger, original_sources, crossings, touches, distance_checks = [], [], [], [], []
        state_inputs = defaultdict(list)
        maximum_root_error, maximum_convergence = 0.0, 0.0
        for source in sources:
            source_rows, paths = [], []
            for path_index, coordinates in enumerate(source["paths"]):
                path_rows, edge_lengths = [], []
                for edge_index, (start, end) in enumerate(zip(coordinates, coordinates[1:])):
                    edge = _SourceEdge(start, end)
                    if edge.length == 0:
                        continue
                    offset = math.fsum(edge_lengths)
                    edge_lengths.append(edge.length)
                    second_length = abs(SECOND_GEODESIC.inv(*start, *end)[2])
                    distance_checks.append({"source_key": source["key"], "path_index": path_index,
                                            "edge_index": edge_index, "geographiclib_meters": edge.length,
                                            "pyproj_meters": second_length,
                                            "difference_meters": second_length - edge.length})
                    for part in self._partition_edge(edge):
                        left, right = part.pop("start_cut"), part.pop("end_cut")
                        maximum_root_error = max(maximum_root_error, left["error_meters"], right["error_meters"])
                        maximum_convergence = max(maximum_convergence, left.get("convergence_change_meters", 0),
                                                  right.get("convergence_change_meters", 0))
                        row = {"source_key": source["key"], "path_index": path_index,
                               "start_m": offset + part.pop("local_start_m"),
                               "end_m": offset + part.pop("local_end_m"), **part,
                               "start_error_bound_meters": left["error_meters"],
                               "end_error_bound_meters": right["error_meters"],
                               "_start_cut": left, "_end_cut": right}
                        if path_rows and (path_rows[-1]["kind"], path_rows[-1]["state_codes"]) == (row["kind"], row["state_codes"]):
                            previous = path_rows[-1]
                            previous["end_m"] = row["end_m"]
                            previous["length_meters"] += row["length_meters"]
                            previous["coordinates"].extend(row["coordinates"][1:])
                            previous["end_error_bound_meters"] = row["end_error_bound_meters"]
                            previous["_end_cut"] = row["_end_cut"]
                        else:
                            path_rows.append(row)
                path_length = math.fsum(edge_lengths)
                paths.append({"path_index": path_index, "vertex_count": len(coordinates),
                              "original_meters": path_length, "original_survey_miles": path_length / SURVEY_MILE})
                previous_interior = None
                for row_index, row in enumerate(path_rows):
                    row["id"] = f"{source['key']}:p{path_index}:i{row_index}"
                    if row["kind"] == "interior":
                        if previous_interior and previous_interior["state_codes"] != row["state_codes"]:
                            cut = row["_start_cut"]
                            coordinate = row["coordinates"][0]
                            boundary = cut.get("boundary_coordinates")
                            angle = None
                            if boundary and row["length_meters"] > 0:
                                _, latitude = coordinate
                                eccentricity2 = GRS80_F * (2 - GRS80_F)
                                phi = math.radians(latitude)
                                n = GRS80_A / math.sqrt(1 - eccentricity2 * math.sin(phi)**2)
                                m = GRS80_A * (1 - eccentricity2) / (1 - eccentricity2 * math.sin(phi)**2)**1.5
                                dx = (boundary[1][0] - boundary[0][0]) * n * math.cos(phi)
                                dy = (boundary[1][1] - boundary[0][1]) * m
                                boundary_bearing = math.degrees(math.atan2(dx, dy))
                                next_coordinate = row["coordinates"][1]
                                pipeline_bearing = GEODESIC.Inverse(latitude, coordinate[0], next_coordinate[1], next_coordinate[0])["azi1"]
                                difference = abs((pipeline_bearing - boundary_bearing + 180) % 360 - 180)
                                angle = min(difference, 180 - difference)
                            crossings.append({"source_key": source["key"], "path_index": path_index,
                                              "chainage_meters": row["start_m"], "coordinate": coordinate,
                                              "from_states": previous_interior["state_codes"],
                                              "to_states": row["state_codes"], "angle_degrees": angle,
                                              "via_shared_interval": row_index > 0 and path_rows[row_index - 1]["kind"] == "shared",
                                              "cut_error_bound_meters": row["start_error_bound_meters"]})
                        previous_interior = row
                    elif row["kind"] != "shared":
                        previous_interior = None
                if path_rows:
                    for position, coordinate, row in (("start", coordinates[0], path_rows[0]),
                                                       ("end", coordinates[-1], path_rows[-1])):
                        endpoint_states = self.states_at(coordinate)
                        touched = sorted(set(endpoint_states) - set(row["state_codes"]))
                        if touched and len(endpoint_states) >= 2 and row["kind"] == "interior":
                            touches.append({"source_key": source["key"], "path_index": path_index,
                                            "endpoint": position, "coordinate": list(coordinate),
                                            "chainage_meters": 0.0 if position == "start" else path_length,
                                            "interior_states": row["state_codes"], "touched_states": touched,
                                            "positive_length_in_touched_states_meters": 0.0})
                source_rows.extend(path_rows)
            for row in source_rows:
                row.pop("_start_cut")
                row.pop("_end_cut")
                row["key"] = row["source_key"]
                row["cut_error_bound_meters"] = max(row["start_error_bound_meters"], row["end_error_bound_meters"])
            total = math.fsum(p["original_meters"] for p in paths)
            source_states = self._account(source_rows)
            attributed = math.fsum(r["attributed_original_meters"] for r in source_states.values())
            outside = math.fsum(r["length_meters"] for r in source_rows if r["kind"] == "outside")
            unresolved = math.fsum(r["length_meters"] for r in source_rows if r["kind"] == "unresolved")
            measured_fragments = math.fsum(distance(a, b) for r in source_rows
                                           for a, b in zip(r["coordinates"], r["coordinates"][1:]))
            tolerance = max(.001, total * 1e-10)
            delta = math.fsum((attributed, outside, unresolved, -total))
            if abs(delta) > tolerance or abs(measured_fragments - total) > tolerance:
                raise ValueError(f"Source conservation failed for {source['key']}")
            for path in paths:
                path_rows = [r for r in source_rows if r["path_index"] == path["path_index"]]
                path["states"] = self._account(path_rows)
                path["interval_ids"] = [r["id"] for r in path_rows]
            original_sources.append({"key": source["key"], "name": source["name"], "motif": source.get("motif", ""),
                                     "xml_order": source["xml_order"], "paths": paths, "states": source_states,
                                     "original_meters": total, "original_survey_miles": total / SURVEY_MILE,
                                     "conservation_difference_meters": delta,
                                     "fragment_geometry_difference_meters": measured_fragments - total,
                                     "conservation_tolerance_meters": tolerance, "conservation_passed": True})
            for code in source_states:
                interiors = [r for r in source_rows if r["kind"] == "interior" and r["state_codes"] == [code]]
                if interiors:
                    state_inputs[code].append({**source, "paths": [r["coordinates"] for r in interiors],
                                               "fragment_references": [{"path_index": r["path_index"],
                                                                         "interval_id": r["id"], "start_m": r["start_m"],
                                                                         "end_m": r["end_m"]} for r in interiors]})
            ledger.extend(source_rows)
        total = math.fsum(r["original_meters"] for r in original_sources)
        states = self._account(ledger)
        outside = math.fsum(r["length_meters"] for r in ledger if r["kind"] == "outside")
        unresolved = math.fsum(r["length_meters"] for r in ledger if r["kind"] == "unresolved")
        delta = math.fsum([r["attributed_original_meters"] for r in states.values()] + [outside, unresolved, -total])
        if abs(delta) > max(.001, total * 1e-10):
            raise ValueError("Fixture conservation failed")
        max_distance_difference = max((abs(r["difference_meters"]) for r in distance_checks), default=0.0)
        if max_distance_difference > .000002 or maximum_root_error > .01:
            raise ValueError("Independent distance/crossing certification failed")
        return {"original_meters": total, "original_survey_miles": total / SURVEY_MILE,
                "combined_original_meters": total, "combined_original_survey_miles": total / SURVEY_MILE,
                "sources": original_sources, "intervals": ledger, "states": states,
                "represented_states": sorted(states), "crossings": crossings, "crossing_count": len(crossings),
                "endpoint_touches": touches, "outside_meters": outside, "unresolved_meters": unresolved,
                "shared_meters": math.fsum(r["length_meters"] for r in ledger if r["kind"] == "shared"),
                "shared_allocations": [{"interval_id": r["id"], "source_key": r["source_key"],
                                        "state_code": code, "allocated_meters": r["length_meters"] / len(r["state_codes"])}
                                       for r in ledger if r["kind"] == "shared" for code in r["state_codes"]],
                "state_inputs": dict(state_inputs),
                "interior_fragments": {code: {source["key"]: [
                    {"coordinates": coordinates, **reference, "interval_ids": [reference["interval_id"]]}
                    for coordinates, reference in zip(source["paths"], source["fragment_references"])]
                    for source in inputs} for code, inputs in state_inputs.items()},
                "certification": {
                    "method": "GeographicLib GRS80 geodesic; native coordinate-line roots; analytic coordinate-curvature envelope and sign exclusion; station bisection",
                    "longitude_wrapping": "Unwrap each native exterior and hole; query periodic copies without CRS transformation or simplification",
                    "cut_target_meters": .01, "maximum_cut_error_bound_meters": maximum_root_error,
                    "root_bracket_width_target_meters": ROOT_WIDTH_M,
                    "coordinate_roundoff_allowance_meters": NUMERICAL_FLOOR_M,
                    "maximum_root_refinement_change_meters": maximum_convergence,
                    "conservation_difference_meters": delta, "conservation_tolerance_meters": max(.001, total * 1e-10),
                    "conservation_passed": True, "distance_crosscheck": {
                        "primary": "geographiclib 2.0 Python Geodesic(a=6378137,f=1/298.257222101)",
                        "secondary": "pyproj 3.7.1 PROJ Geod(ellps=GRS80)",
                        "independence_limit": "Distinct Python/C implementations of related Karney geodesic algorithms; not independent earth models",
                        "checked_edge_count": len(distance_checks), "maximum_absolute_difference_meters": max_distance_difference,
                        "largest_differences": sorted(distance_checks, key=lambda r: abs(r["difference_meters"]), reverse=True)[:12]},
                    "domain_limits": "Nonpolar source edges <=200 km, unambiguous transverse roots or exact meridian/equator coincidence; unresolved near tangencies raise an error",
                    "boundary_resource_sha256": BOUNDARY_SHA256}}

    def _account(self, intervals):
        values = defaultdict(lambda: {"interior_meters": [], "shared_allocation_meters": [],
                                       "length_error_bound_meters": [], "shared_interval_references": []})
        for row in intervals:
            if row["kind"] not in ("interior", "shared"):
                continue
            divisor = len(row["state_codes"]) if row["kind"] == "shared" else 1
            for code in row["state_codes"]:
                field = "shared_allocation_meters" if row["kind"] == "shared" else "interior_meters"
                values[code][field].append(row["length_meters"] / divisor)
                values[code]["length_error_bound_meters"].append((row["start_error_bound_meters"] + row["end_error_bound_meters"]) / divisor)
                if row["kind"] == "shared":
                    values[code]["shared_interval_references"].append({"interval_id": row["id"],
                                                                     "allocated_meters": row["length_meters"] / divisor})
        result = {}
        for code, value in sorted(values.items()):
            interior, allocation = math.fsum(value["interior_meters"]), math.fsum(value["shared_allocation_meters"])
            attributed = interior + allocation
            result[code] = {"name": self.state_names[code], "interior_meters": interior,
                            "shared_allocation_meters": allocation, "attributed_original_meters": attributed,
                            "attributed_original_survey_miles": attributed / SURVEY_MILE,
                            "length_error_bound_meters": math.fsum(value["length_error_bound_meters"]),
                            "shared_interval_references": value["shared_interval_references"]}
        return result
