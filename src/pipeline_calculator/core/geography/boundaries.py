"""Versioned offline boundaries and bounded spatial queries.

Boundary edges retain their source coordinate-linear interpretation. Pipeline
edges are GRS80 geodesics. Numerical clipping accuracy is not survey accuracy.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import zipfile

import numpy as np
from shapely import from_wkb, linestrings
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.strtree import STRtree


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def _unwrap_ring(ring, center=None):
    arr = np.asarray(ring.coords, dtype=float)[:, :2].copy()
    arr[:, 0] = np.degrees(np.unwrap(np.radians(arr[:, 0])))
    if center is not None:
        arr[:, 0] += 360 * round((center - float(np.mean(arr[:, 0]))) / 360)
    return arr


def canonical_geometry(geometry):
    """Split antimeridian rings without inventing globe-spanning edges."""
    parts = []
    for part in polygon_parts(geometry):
        outer = _unwrap_ring(part.exterior)
        center = float(np.mean(outer[:, 0]))
        unwrapped = Polygon(outer, [_unwrap_ring(h, center) for h in part.interiors])
        if not unwrapped.is_valid:
            raise ValueError("Invalid state polygon; automatic topology repair is not permitted")
        lo, _, hi, _ = unwrapped.bounds
        first = int(np.floor((lo + 180) / 360))
        last = int(np.floor((hi + 180) / 360))
        for zone in range(first, last + 1):
            piece = unwrapped.intersection(box(-180 + 360*zone, -90, 180 + 360*zone, 90))
            parts.extend(polygon_parts(translate(piece, xoff=-360*zone)))
    if not parts:
        raise ValueError("State geometry is empty")
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


class BoundaryDataset:
    """Injectable boundary snapshot. Geometry/index objects never enter results."""

    def __init__(self, geometries, state_names=None, boundary_source=None):
        self.geometries = {str(code): canonical_geometry(g) for code, g in geometries.items()}
        self.state_names = dict(state_names or {code: code for code in geometries})
        self.boundary_source = dict(boundary_source or {"name": "Injected test boundaries", "vintage": "test"})
        self._parts = []
        self._part_codes = []
        for code, geometry in self.geometries.items():
            for part in polygon_parts(geometry):
                self._parts.append(part)
                self._part_codes.append(code)
        self._tree = STRtree(self._parts)
        self._edge_cache = {}

    def states_at(self, lon, lat):
        from shapely.geometry import Point
        point = Point(lon, lat)
        indices = self._tree.query(point, predicate="intersects")
        return sorted({self._part_codes[int(i)] for i in indices})

    def candidate_edges(self, bounds, *, context=None):
        """Yield native boundary segments from component-local spatial indexes."""
        query = box(*bounds)
        for position in self._tree.query(query):
            if context is not None:
                context.check()
            position = int(position)
            if position not in self._edge_cache:
                # Bound the retained native geometry indexes across nationwide jobs.
                if len(self._edge_cache) >= 8:
                    self._edge_cache.pop(next(iter(self._edge_cache)))
                part = self._parts[position]
                rings = [np.asarray(r.coords)[:, :2] for r in [part.exterior, *part.interiors]]
                coords = np.concatenate([np.stack([r[:-1], r[1:]], axis=1) for r in rings])
                if len(coords) > 2_000_000:
                    raise ValueError("State boundary vertex budget exceeded")
                segments = linestrings(coords)
                self._edge_cache[position] = (coords, STRtree(segments))
            coords, tree = self._edge_cache[position]
            for index in tree.query(query):
                yield self._part_codes[position], coords[int(index)]


@lru_cache(maxsize=1)
def load_boundaries(resource_path=None):
    path = Path(resource_path) if resource_path is not None else (
        Path(__file__).resolve().parents[2] / "data" / "states_2025.zip")
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("manifest.json"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported state boundary resource schema")
        geometries = {}
        names = {}
        for entry in metadata["states"]:
            raw = archive.read(entry["file"])
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"State boundary checksum mismatch: {entry['code']}")
            geometries[entry["code"]] = from_wkb(raw)
            names[entry["code"]] = entry["name"]
    if len(geometries) != 51 or "DC" not in geometries:
        raise ValueError("State resource must contain all 50 states and Washington, DC")
    provenance = {k: v for k, v in metadata.items() if k != "states"}
    provenance["resource_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return BoundaryDataset(geometries, names, provenance)
