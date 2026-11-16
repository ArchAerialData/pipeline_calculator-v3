"""Explicit horizontal CRS policy. No network grid download or guessed CRS."""
import math
import hashlib
import warnings

import pyproj
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

from .models import ImportFailure


def parse_crs(value):
    if value is None or str(value).strip().lower() in ('', 'undefined', 'unknown'):
        return None
    try:
        crs = CRS.from_user_input(value)
    except (ValueError, pyproj.exceptions.CRSError):
        return None
    if not (crs.is_geographic or crs.is_projected or crs.is_compound):
        return None
    return crs


def _area_contains(area, longitude, latitude):
    if area is None:
        return False
    west, south, east, north = area.west, area.south, area.east, area.north
    longitude_ok = west <= longitude <= east if west <= east else longitude >= west or longitude <= east
    return longitude_ok and south <= latitude <= north


class CoordinateOperation:
    def __init__(self, source_crs):
        self.source = parse_crs(source_crs)
        if self.source is None:
            raise ImportFailure('The assigned source CRS is invalid or has no horizontal coordinates.', code='invalid_crs')
        # A compound CRS may contain vertical information we do not analyze.
        horizontal = self.source.to_2d()
        self._geographic = horizontal.is_geographic
        self._identity = horizontal.equals(CRS.from_epsg(4326), ignore_axis_order=True)
        self._angular_limits = None
        if self._geographic:
            longitude_axis = next((axis for axis in horizontal.axis_info if axis.direction in ('east', 'west')), None)
            latitude_axis = next((axis for axis in horizontal.axis_info if axis.direction in ('north', 'south')), None)
            if longitude_axis is not None and latitude_axis is not None:
                longitude_factor = longitude_axis.unit_conversion_factor
                latitude_factor = latitude_axis.unit_conversion_factor
                if longitude_factor and latitude_factor:
                    # CRS angular unit factors convert native units to radians.
                    # Longitude here is relative to that CRS's prime meridian;
                    # PROJ applies the meridian shift during transformation.
                    self._angular_limits = (math.nextafter(math.pi / longitude_factor, math.inf),
                                            math.nextafter(math.pi / (2 * latitude_factor), math.inf))
        pyproj.network.set_network_enabled(False)
        self.target = CRS.from_epsg(4326)
        self._used = {}
        self.last_operation_id = None
        if self._identity:
            self._candidates = [Transformer.from_crs(horizontal, self.target, always_xy=True)]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                group = TransformerGroup(horizontal, self.target, always_xy=True, allow_ballpark=False)
            if not group.transformers:
                raise ImportFailure('No available offline coordinate operation can transform this CRS. Install the required grids or re-export in EPSG:4326.', code='unavailable_coordinate_operation')
            self._candidates = list(group.transformers)
            if not group.best_available:
                # Explicit published mainland NAD83 policy already used by the
                # app's boundary dataset, never a general ballpark fallback.
                self._candidates = [candidate for candidate in self._candidates
                    if any(operation.to_json_dict().get('id', {}) == {'authority': 'EPSG', 'code': 1188}
                           for operation in candidate.operations)]
                if not self._candidates:
                    raise ImportFailure('The preferred coordinate operation needs unavailable grids. Install those grids or re-export in EPSG:4326.', code='unavailable_coordinate_operation')
        # PROJ network is off in this process by default; explicitly avoid any
        # operation requiring unavailable grids rather than enabling downloads.

    def path(self, coordinates):
        if not coordinates:
            return []
        xy = []
        for point in coordinates:
            if len(point) < 2 or not math.isfinite(point[0]) or not math.isfinite(point[1]):
                raise ImportFailure('Geometry contains a non-finite XY coordinate.', code='invalid_geometry')
            if self._angular_limits is not None:
                longitude_limit, latitude_limit = self._angular_limits
                if not (-longitude_limit <= point[0] <= longitude_limit and -latitude_limit <= point[1] <= latitude_limit):
                    raise ImportFailure('Geometry contains an out-of-range geographic XY coordinate in its declared angular units.', code='invalid_geometry')
            xy.append((float(point[0]), float(point[1])))
        xs, ys = zip(*xy)
        valid_transformation = False
        for candidate in self._candidates:
            try:
                tx, ty = candidate.transform(xs, ys, errcheck=True)
            except pyproj.exceptions.ProjError:
                continue
            result = list(zip(tx, ty))
            if not all(math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90 for x, y in result):
                continue
            valid_transformation = True
            area = candidate.area_of_use
            if area is not None and not all(_area_contains(area, x, y) for x, y in result):
                continue
            if area is None and not self._identity:
                continue
            key = candidate.definition
            self.last_operation_id = hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]
            grids = []
            for operation in candidate.operations:
                for grid in operation.grids:
                    grids.append({'name': grid.short_name, 'available': grid.available})
            self._used[key] = {
                'id': self.last_operation_id,
                'description': candidate.description, 'pipeline': candidate.definition,
                'accuracy_m': candidate.accuracy if candidate.accuracy >= 0 else None,
                'area': {'name': area.name, 'west': area.west, 'south': area.south,
                         'east': area.east, 'north': area.north} if area else None,
                'grids': grids,
            }
            return [(float(x), float(y)) for x, y in result]
        if not valid_transformation:
            raise ImportFailure('Geometry cannot be transformed to finite, valid longitude/latitude coordinates.', code='invalid_geometry')
        raise ImportFailure('No available offline coordinate operation covers this complete geometry part.', code='coordinate_operation_area')

    def metadata(self):
        return {'source_crs': self.source.to_wkt(), 'target_crs': 'EPSG:4326', 'axis_order': 'longitude, latitude',
                'operations': list(self._used.values()), 'pyproj': pyproj.__version__,
                'proj': pyproj.proj_version_str,
                'epsg_database': pyproj.database.get_database_metadata('EPSG.VERSION'),
                'measurement': 'GRS80 geodesics between transformed source XY vertices; US survey miles',
                'densification': False}
