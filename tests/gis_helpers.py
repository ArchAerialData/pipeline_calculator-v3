"""Small GIS fixtures with explicit topology; no production normalization code."""
from pathlib import Path
import json
import zipfile

import numpy as np
from pyproj import CRS
import pyogrio.raw
from shapely import to_wkb
from shapely.geometry import mapping


LINE = [(-100.0, 30.0), (-100.001, 30.0)]
OTHER_LINE = [(-100.1, 30.1), (-100.101, 30.1)]


def write_vector(path, geometries, *, layer="pipes", crs="EPSG:4326", fields=None,
                 geometry_type=None):
    """Use the public raw writer, avoiding a GeoPandas/Arrow dependency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields if fields is not None else {"Name": [f"feature {i}" for i in range(len(geometries))]}
    types = {geom.geom_type for geom in geometries if geom is not None}
    geometry_type = geometry_type or (next(iter(types)) if len(types) == 1 else "Unknown")
    driver = {".gpkg": "GPKG", ".shp": "ESRI Shapefile"}[path.suffix.lower()]
    pyogrio.raw.write(
        path, np.array([to_wkb(g) if g is not None else None for g in geometries], dtype=object),
        [np.asarray(values) for values in fields.values()], list(fields), layer=layer,
        driver=driver, geometry_type=geometry_type,
        crs=CRS(crs).to_wkt() if crs is not None else None, encoding="UTF-8",
    )
    return path


def write_geojson(path, geometries, *, properties=None):
    path = Path(path)
    data = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": i, "properties": (properties or [{}] * len(geometries))[i],
         "geometry": mapping(geometry) if geometry is not None else None}
        for i, geometry in enumerate(geometries)
    ]}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def zip_files(path, files, *, prefix=""):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, prefix + Path(file).name)
    return path


def paths(parsed):
    return [path for pipeline in parsed.pipelines for path in pipeline["coordinate_paths"]]


def diagnostic_codes(parsed):
    return {item["code"] for item in parsed.diagnostics}
