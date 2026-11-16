"""The tested import allowlist, shared by GUI and headless entrypoints."""
from pathlib import Path

SUPPORTED_SUFFIXES = ('.kml', '.kmz', '.shp', '.gpkg', '.geojson', '.json', '.zip')
FILE_TYPES = [
    ('All supported GIS files', ' '.join('*' + suffix for suffix in SUPPORTED_SUFFIXES)),
    ('ZIP GIS packages', '*.zip'), ('Shapefiles', '*.shp'),
    ('GeoPackage', '*.gpkg'), ('GeoJSON', '*.geojson *.json'),
    ('KMZ files', '*.kmz'), ('KML files', '*.kml'), ('All files', '*.*'),
]
DRIVERS = {'.shp': 'ESRI Shapefile', '.gpkg': 'GPKG', '.geojson': 'GeoJSON', '.json': 'GeoJSON'}


def is_supported_path(path):
    return Path(str(path)).suffix.lower() in SUPPORTED_SUFFIXES
