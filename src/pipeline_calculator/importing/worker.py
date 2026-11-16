"""Public GDAL calls isolated in a cancellable process with bounded JSON IPC.

Input signatures are checked before GDAL opens a dataset. GeoPackage original
geometry is audited before a reader can approximate curves or surfaces.
"""
import base64
import datetime
from decimal import Decimal, InvalidOperation
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
import struct
import warnings

from .limits import (BATCH_FEATURES, MAX_FEATURES, MAX_VERTICES, MAX_COLLECTION_DEPTH,
                     MAX_LAYERS, MAX_IPC_BYTES, MAX_ATTRIBUTE_BYTES, check)
from .models import ImportFailure


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _database(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA trusted_schema=OFF')
    # CPython omits this method when SQLite was compiled without loadable
    # extension support (notably some macOS builds). In that case loading is
    # unavailable already; do not require an optional capability to disable it.
    disable_extensions = getattr(connection, 'enable_load_extension', None)
    if disable_extensions is not None:
        disable_extensions(False)
    return connection


def _wkb_type_audit(blob, expected_srs=None):
    """Bounded traversal of original WKB structure, never a geometry decoder."""
    if len(blob) < 8 or blob[:2] != b'GP' or blob[2] != 0:
        raise ImportFailure('Invalid GeoPackage geometry header.', code='invalid_geometry')
    flags = blob[3]
    header_srs = struct.unpack_from(('<' if flags & 1 else '>') + 'i', blob, 4)[0]
    if expected_srs is not None and header_srs != expected_srs:
        raise ImportFailure('Geometry header CRS differs from its GeoPackage layer CRS.', code='inconsistent_crs')
    envelope = (flags >> 1) & 7
    if flags & 0xE0 or envelope not in (0, 1, 2, 3, 4):
        raise ImportFailure('Extended GeoPackage geometry is unsupported.', code='unsupported_geometry')
    start = 8 + (0, 4, 6, 6, 8)[envelope] * 8
    total_vertices = 0
    dimensions = set()

    def walk(offset, depth):
        nonlocal total_vertices
        if depth > MAX_COLLECTION_DEPTH or offset + 5 > len(blob):
            raise ImportFailure('Invalid or excessive GeoPackage geometry nesting.', code='invalid_geometry')
        order = blob[offset]
        if order not in (0, 1):
            raise ImportFailure('Invalid WKB byte order.', code='invalid_geometry')
        endian = '<' if order else '>'
        encoded = struct.unpack_from(endian + 'I', blob, offset + 1)[0]
        offset += 5
        has_z, has_m, has_srid = bool(encoded & 0x80000000), bool(encoded & 0x40000000), bool(encoded & 0x20000000)
        kind = encoded & 0x1FFFFFFF
        if kind >= 1000:
            dimensional, kind = divmod(kind, 1000)
            if dimensional not in (1, 2, 3):
                raise ImportFailure('Unsupported original WKB geometry type.', code='unsupported_geometry')
            has_z, has_m = dimensional in (1, 3), dimensional in (2, 3)
        if kind not in range(1, 8):
            raise ImportFailure('Curves, surfaces and non-simple-feature geometry are unsupported.', code='unsupported_geometry')
        if has_srid:
            offset += 4
        if has_z:
            dimensions.add('Z')
        if has_m:
            dimensions.add('M')
        stride = 8 * (2 + has_z + has_m)

        def count(position):
            if position + 4 > len(blob):
                raise ImportFailure('Truncated WKB count.', code='invalid_geometry')
            value = struct.unpack_from(endian + 'I', blob, position)[0]
            if value > MAX_VERTICES:
                raise ImportFailure('Geometry exceeds the vertex budget.', code='vertex_limit')
            return value, position + 4

        def vertices(position, number):
            nonlocal total_vertices
            total_vertices += number
            if total_vertices > MAX_VERTICES or position + number * stride > len(blob):
                raise ImportFailure('Geometry is truncated or exceeds the vertex budget.', code='vertex_limit')
            return position + number * stride

        if kind == 1:
            return vertices(offset, 1)
        number, offset = count(offset)
        if kind == 2:
            return vertices(offset, number)
        if kind == 3:
            for _ in range(number):
                size, offset = count(offset)
                offset = vertices(offset, size)
            return offset
        for _ in range(number):
            offset = walk(offset, depth + 1)
        return offset

    if walk(start, 0) != len(blob):
        raise ImportFailure('GeoPackage geometry has trailing or missing bytes.', code='invalid_geometry')
    return total_vertices, sorted(dimensions)


def _gpkg_metadata(path):
    with _database(path) as db:
        tables = {row[0]: row[1] for row in db.execute("SELECT name, sql FROM sqlite_schema WHERE type='table'")}
        if not {'gpkg_contents', 'gpkg_geometry_columns', 'gpkg_spatial_ref_sys'} <= tables.keys():
            raise ImportFailure('SQLite file lacks required GeoPackage metadata.', code='invalid_geopackage')
        contents = db.execute('SELECT table_name, data_type, identifier, srs_id FROM gpkg_contents').fetchall()
        if len(contents) > MAX_LAYERS:
            raise ImportFailure('GeoPackage exceeds the layer limit.', code='layer_limit')
        layers = []
        for table, data_type, identifier, srs_id in contents:
            if not isinstance(table, str) or table not in tables or 'VIRTUAL TABLE' in (tables[table] or '').upper():
                raise ImportFailure('GeoPackage feature views or virtual tables are unsupported.', code='unsupported_geopackage_table')
            geometry = db.execute('SELECT column_name, geometry_type_name, srs_id, z, m FROM gpkg_geometry_columns WHERE table_name=?', (table,)).fetchall()
            if data_type != 'features' or len(geometry) != 1:
                layers.append({'name': table, 'geometry_type': data_type, 'supported': False,
                               'reason': 'Only vector feature layers can be analyzed.', 'feature_count': None, 'crs': None})
                continue
            column, kind, geometry_srs, z, m = geometry[0]
            if geometry_srs != srs_id:
                raise ImportFailure('GeoPackage contents and geometry-column CRS disagree.', code='inconsistent_crs')
            crs = None
            if geometry_srs not in (-1, 0):
                row = db.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?', (geometry_srs,)).fetchone()
                if row and row[0] not in ('undefined', ''):
                    crs = row[0]
            count = db.execute('SELECT count(*) FROM ' + _quote(table)).fetchone()[0]
            if count > MAX_FEATURES:
                raise ImportFailure('GeoPackage exceeds the feature budget.', code='feature_limit')
            supported = str(kind).upper() in {'GEOMETRY', 'GEOMETRYCOLLECTION', 'POINT', 'MULTIPOINT', 'LINESTRING', 'MULTILINESTRING'}
            layers.append({'name': table, 'geometry_type': kind, 'supported': supported,
                           'reason': '' if supported else 'This geometry family does not contribute pipelines or pins.',
                           'feature_count': count, 'crs': crs, 'geometry_column': column,
                           'srs_id': geometry_srs,
                           'dimensions': [name for name, value in (('Z', z), ('M', m)) if value]})
        return layers


def _audit_selected_gpkg(path, layer):
    descriptors = _gpkg_metadata(path)
    descriptor = next((item for item in descriptors if item['name'] == layer), None)
    if descriptor is None or 'geometry_column' not in descriptor:
        raise ImportFailure('Selected GeoPackage layer is unavailable.', code='invalid_layer_selection')
    vertices, dimensions = 0, set(descriptor.get('dimensions', []))
    with _database(path) as db:
        query = 'SELECT ' + _quote(descriptor['geometry_column']) + ' FROM ' + _quote(layer)
        for index, (blob,) in enumerate(db.execute(query)):
            if index >= MAX_FEATURES:
                raise ImportFailure('GeoPackage exceeds the feature budget.', code='feature_limit')
            if blob is None:
                continue
            count, dims = _wkb_type_audit(blob, descriptor['srs_id'])
            vertices += count
            dimensions.update(dims)
            if vertices > MAX_VERTICES:
                raise ImportFailure('GeoPackage exceeds the vertex budget.', code='vertex_limit')
    return sorted(dimensions)


def _preflight(path, driver):
    with open(path, 'rb') as stream:
        header = stream.read(100)
    if driver == 'GPKG':
        if not header.startswith(b'SQLite format 3\x00'):
            raise ImportFailure('Input is not a GeoPackage SQLite database.', code='format_mismatch')
        return _gpkg_metadata(path)
    if driver == 'ESRI Shapefile':
        if len(header) != 100 or struct.unpack_from('>I', header)[0] != 9994 or struct.unpack_from('<I', header, 28)[0] != 1000:
            raise ImportFailure('Invalid shapefile header.', code='format_mismatch')
        kind = struct.unpack_from('<I', header, 32)[0]
        if kind not in (0, 1, 3, 5, 8, 11, 13, 15, 18, 21, 23, 25, 28):
            raise ImportFailure('Multipatch or unknown shapefile geometry is unsupported.', code='unsupported_geometry')
        return _audit_shp(path, kind)
    if driver == 'GeoJSON':
        try:
            with open(path, encoding='utf-8-sig') as stream:
                document = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ImportFailure('Invalid GeoJSON JSON content.', code='invalid_geojson') from exc
        if not isinstance(document, dict) or document.get('type') not in {'FeatureCollection', 'Feature', 'Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon', 'GeometryCollection'}:
            raise ImportFailure('JSON is not a GeoJSON object.', code='invalid_geojson')
        if document['type'] == 'FeatureCollection':
            original_features = document.get('features')
            if not isinstance(original_features, list) or len(original_features) > MAX_FEATURES:
                raise ImportFailure('GeoJSON feature collection is invalid or exceeds its feature budget.', code='feature_limit')
        elif document['type'] == 'Feature':
            original_features = [document]
        else:
            original_features = [{'type': 'Feature', 'geometry': document, 'properties': {}}]
        for feature in original_features:
            if (not isinstance(feature, dict) or feature.get('type') != 'Feature' or 'geometry' not in feature
                    or feature.get('geometry') is not None and not isinstance(feature['geometry'], dict)
                    or feature.get('properties') is not None and not isinstance(feature['properties'], dict)):
                raise ImportFailure('GeoJSON contains a malformed Feature.', code='invalid_geojson')
        pending = [(document, 0)]
        count = 0
        vertex_count = 0
        def coordinate(value):
            nonlocal vertex_count
            vertex_count += 1
            if vertex_count > MAX_VERTICES:
                raise ImportFailure('GeoJSON exceeds the vertex budget.', code='vertex_limit')
            if (not isinstance(value, list) or len(value) < 2 or
                    any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
                        for number in value)):
                raise ImportFailure('GeoJSON contains an invalid coordinate tuple.', code='invalid_geometry')
        def coordinates(value, nesting):
            if nesting == 0:
                coordinate(value)
            elif not isinstance(value, list):
                raise ImportFailure('GeoJSON coordinates have invalid nesting.', code='invalid_geometry')
            else:
                for child in value:
                    coordinates(child, nesting - 1)
        while pending:
            item, depth = pending.pop()
            if depth > MAX_COLLECTION_DEPTH:
                raise ImportFailure('GeoJSON exceeds the nesting budget.', code='geometry_depth_limit')
            if isinstance(item, dict):
                count += 1
                if count > MAX_FEATURES * 10:
                    raise ImportFailure('GeoJSON exceeds the object budget.', code='feature_limit')
                if 'crs' in item and item['crs'] is not None:
                    raise ImportFailure('Legacy GeoJSON CRS declarations are unsupported. Re-export as RFC 7946 longitude/latitude.', code='geojson_legacy_crs')
                kind = item.get('type')
                nesting = {'Point': 0, 'MultiPoint': 1, 'LineString': 1, 'MultiLineString': 2, 'Polygon': 2, 'MultiPolygon': 3}
                if kind in nesting:
                    value = item.get('coordinates')
                    if value != []:
                        coordinates(value, nesting[kind])
                elif kind not in {'FeatureCollection', 'Feature', 'GeometryCollection'}:
                    raise ImportFailure('Unsupported GeoJSON geometry type.', code='unsupported_geometry')
                for key in ('features', 'geometry', 'geometries'):
                    value = item.get(key)
                    if isinstance(value, dict):
                        pending.append((value, depth + 1))
                    elif isinstance(value, list):
                        pending.extend((child, depth + 1) for child in value if isinstance(child, dict))
        return {'original_features': [{'properties': feature.get('properties') or {}, 'id': feature.get('id')}
                                     for feature in original_features]}
    raise ImportFailure('Driver is not in the import allowlist.', code='unsupported_driver')


def _audit_shp(path, header_kind):
    data = Path(path).read_bytes()
    index = Path(path).with_suffix('.shx').read_bytes()
    dbf = Path(path).with_suffix('.dbf').read_bytes()
    if len(index) < 100 or (len(index) - 100) % 8 or len(dbf) < 32:
        raise ImportFailure('Truncated SHX or DBF component.', code='invalid_shapefile')
    if struct.unpack_from('>I', data, 24)[0] * 2 != len(data):
        raise ImportFailure('Shapefile header length does not match its content.', code='invalid_shapefile')
    if (struct.unpack_from('>I', index)[0] != 9994 or struct.unpack_from('<I', index, 28)[0] != 1000
            or struct.unpack_from('<I', index, 32)[0] != header_kind
            or struct.unpack_from('>I', index, 24)[0] * 2 != len(index)):
        raise ImportFailure('Shapefile index header is inconsistent.', code='invalid_shapefile')
    dbf_count = struct.unpack_from('<I', dbf, 4)[0]
    header_bytes, row_bytes = struct.unpack_from('<HH', dbf, 8)
    if not row_bytes or header_bytes + dbf_count * row_bytes > len(dbf) or dbf_count > MAX_FEATURES:
        raise ImportFailure('DBF is truncated or exceeds the record budget.', code='invalid_shapefile')
    flags = [dbf[header_bytes + number * row_bytes] for number in range(dbf_count)]
    if any(flag not in (0x20, 0x2A) for flag in flags):
        raise ImportFailure('DBF has invalid record deletion flags.', code='invalid_shapefile')
    deleted = flags.count(0x2A)
    offset, count, vertices = 100, 0, 0
    while offset < len(data):
        if offset + 12 > len(data) or 100 + count * 8 + 8 > len(index):
            raise ImportFailure('Shapefile record/index alignment is invalid.', code='invalid_shapefile')
        length = struct.unpack_from('>I', data, offset + 4)[0] * 2
        indexed_offset, indexed_length = struct.unpack_from('>II', index, 100 + count * 8)
        if indexed_offset * 2 != offset or indexed_length * 2 != length or offset + 8 + length > len(data):
            raise ImportFailure('Shapefile record differs from its index.', code='invalid_shapefile')
        body = memoryview(data)[offset + 8:offset + 8 + length]
        kind = struct.unpack_from('<I', body)[0]
        if kind not in (0, header_kind):
            raise ImportFailure('Shapefile contains an incompatible original record type.', code='unsupported_geometry')
        if kind in (1, 11, 21):
            number, start = 1, 4
        elif kind in (8, 18, 28):
            if len(body) < 40:
                raise ImportFailure('Truncated multipoint geometry.', code='invalid_geometry')
            number, start = struct.unpack_from('<I', body, 36)[0], 40
        elif kind in (3, 5, 13, 15, 23, 25):
            if len(body) < 44:
                raise ImportFailure('Truncated multipart geometry.', code='invalid_geometry')
            parts, number = struct.unpack_from('<II', body, 36)
            start = 44 + parts * 4
            if parts > MAX_VERTICES or start > len(body):
                raise ImportFailure('Invalid geometry part offsets.', code='invalid_geometry')
            starts = list(struct.unpack_from('<' + str(parts) + 'I', body, 44)) if parts else []
            if starts and (starts[0] != 0 or starts != sorted(set(starts)) or starts[-1] >= number):
                raise ImportFailure('Invalid geometry part boundaries.', code='invalid_geometry')
            if number and not parts:
                raise ImportFailure('Geometry vertices have no part boundaries.', code='invalid_geometry')
        else:
            number, start = 0, 4
        vertices += number
        if vertices > MAX_VERTICES or start + number * 16 > len(body):
            raise ImportFailure('Shapefile exceeds vertex bounds or is truncated.', code='vertex_limit')
        for position in range(number):
            if not all(math.isfinite(value) for value in struct.unpack_from('<dd', body, start + position * 16)):
                raise ImportFailure('Shapefile contains a non-finite XY vertex.', code='invalid_geometry')
        count += 1
        offset += 8 + length
    if count != dbf_count or count != (len(index) - 100) // 8:
        raise ImportFailure('SHP, SHX and DBF record counts differ.', code='invalid_shapefile')
    return {'dimensions': ['Z', 'M'] if 11 <= header_kind <= 18 else ['M'] if header_kind >= 21 else [],
            'deleted_records': deleted, 'physical_records': count}


def _gpkg_attributes(path, layer):
    db = _database(path)
    columns = db.execute('PRAGMA table_info(' + _quote(layer) + ')').fetchall()
    primary = [row for row in columns if row[5]]
    if len(primary) != 1 or primary[0][2].strip().upper() != 'INTEGER':
        db.close()
        raise ImportFailure('GeoPackage feature IDs require one INTEGER PRIMARY KEY.', code='invalid_geopackage')
    return db, primary[0][1]


def _shp_integer_overrides(path):
    """Retain exact integral DBF values before GDAL nullable-float conversion."""
    data = Path(path).with_suffix('.dbf').read_bytes()
    count = struct.unpack_from('<I', data, 4)[0]
    header, width = struct.unpack_from('<HH', data, 8)
    integer_fields, offset, field_index = [], 1, 0
    position = 32
    while position + 32 <= header and data[position] != 0x0D:
        descriptor = data[position:position + 32]
        length, decimals = descriptor[16], descriptor[17]
        if descriptor[11] == ord('N') and decimals == 0:
            integer_fields.append((field_index, offset, length))
        offset += length
        position += 32
        field_index += 1
    def values(fid):
        result = {}
        if not 0 <= fid < count:
            raise ImportFailure('Shapefile FID does not match a physical DBF row.', code='invalid_feature_identity')
        for name, start, length in integer_fields:
            raw = data[header + fid * width + start:header + fid * width + start + length].strip()
            if not raw or all(char == ord('*') for char in raw):
                result[name] = None
            else:
                try:
                    value = Decimal(raw.decode('ascii'))
                    if not value.is_finite() or value != value.to_integral_value():
                        raise ValueError('not integral')
                    result[name] = int(value)
                except (ValueError, UnicodeError, InvalidOperation) as exc:
                    raise ImportFailure('An integral DBF field cannot be preserved exactly.', code='invalid_attribute') from exc
        return result
    return values


def _plain(value):
    if value is None:
        return None
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {'encoding': 'base64', 'value': base64.b64encode(value).decode('ascii')}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def _send(pipe, value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_IPC_BYTES:
        raise ImportFailure('A geometry batch exceeds the transfer budget.', code='batch_size_limit')
    pipe.send_bytes(encoded)


def _entry(pipe, path, driver, command, layer):
    attributes_db = None
    try:
        # Import inside the child; parent never performs an indivisible GDAL read.
        import pyogrio
        from pyogrio.raw import read
        warnings.filterwarnings('ignore', module='pyogrio')
        preflight = _preflight(path, driver)
        if command == 'discover':
            if driver == 'GPKG':
                layers = preflight
            else:
                layers = []
                for name, kind in pyogrio.list_layers(path):
                    info = pyogrio.read_info(path, layer=str(name))
                    if info['driver'] != driver:
                        raise ImportFailure('Detected driver differs from the selected format.', code='format_mismatch')
                    geometry = str(kind or '')
                    supported = any(token in geometry.lower() for token in ('line', 'point', 'unknown', 'geometrycollection'))
                    crs = 'EPSG:4326' if driver == 'GeoJSON' else info.get('crs')
                    layers.append({'name': str(name), 'geometry_type': geometry, 'feature_count': int(info['features']),
                                   'crs': crs, 'supported': supported,
                                   'reason': '' if supported else 'This geometry family does not contribute pipelines or pins.',
                                   'encoding': info.get('encoding'), 'dimensions': preflight.get('dimensions', [])})
            if len(layers) > MAX_LAYERS:
                raise ImportFailure('Input exceeds the layer budget.', code='layer_limit')
            _send(pipe, {'kind': 'discovery', 'layers': layers,
                         'versions': {'pyogrio': pyogrio.__version__, 'gdal': pyogrio.__gdal_version_string__}})
        else:
            dimensions = _audit_selected_gpkg(path, layer) if driver == 'GPKG' else preflight.get('dimensions', [])
            if driver == 'GPKG':
                attributes_db, primary_key = _gpkg_attributes(path, layer)
            integer_overrides = _shp_integer_overrides(path) if driver == 'ESRI Shapefile' else None
            offset, active_count, attribute_bytes = 0, 0, 0
            seen_fids = set()
            while True:
                meta, fids, geometries, values = read(path, layer=layer, skip_features=offset,
                    max_features=BATCH_FEATURES, return_fids=True, datetime_as_string=True)
                if len(fids) == 0:
                    break
                if active_count + len(fids) > MAX_FEATURES:
                    raise ImportFailure('Input exceeds the feature budget.', code='feature_limit')
                records = []
                for position, fid in enumerate(fids):
                    if driver != 'GeoJSON':
                        if int(fid) in seen_fids:
                            raise ImportFailure('Reader repeated a source feature identity.', code='duplicate_feature_identity')
                        seen_fids.add(int(fid))
                    attrs = {str(field): _plain(array[position]) for field, array in zip(meta['fields'], values)}
                    original_feature = None
                    if driver == 'GeoJSON':
                        original_index = offset + position
                        if original_index >= len(preflight['original_features']):
                            raise ImportFailure('GeoJSON reader record order differs from its source features.', code='feature_count_mismatch')
                        original_feature = preflight['original_features'][original_index]
                        attrs = original_feature['properties']
                    if attributes_db is not None and len(meta['fields']):
                        row = attributes_db.execute('SELECT ' + ','.join(_quote(str(field)) for field in meta['fields'])
                            + ' FROM ' + _quote(layer) + ' WHERE ' + _quote(primary_key) + '=?', (int(fid),)).fetchone()
                        if row is None:
                            raise ImportFailure('Original feature attributes cannot be matched to their FID.', code='invalid_feature_identity')
                        attrs = {str(field): _plain(value) for field, value in zip(meta['fields'], row)}
                    if integer_overrides is not None:
                        for field_index, value in integer_overrides(int(fid)).items():
                            if field_index >= len(meta['fields']):
                                raise ImportFailure('DBF field schema differs from its decoded attributes.', code='invalid_attribute')
                            attrs[str(meta['fields'][field_index])] = value
                    attribute_bytes += len(json.dumps(attrs, ensure_ascii=False).encode('utf-8'))
                    if attribute_bytes > MAX_ATTRIBUTE_BYTES:
                        raise ImportFailure('Input attributes exceed the retained byte budget.', code='attribute_limit')
                    geometry = geometries[position] if geometries is not None else None
                    records.append({'fid': str(int(fid)),
                                    'record_index': int(fid) if driver == 'ESRI Shapefile' else offset + position,
                                    'original_feature_id': original_feature['id'] if original_feature else None, 'attributes': attrs,
                                    'wkb': base64.b64encode(geometry).decode('ascii') if geometry is not None else None})
                _send(pipe, {'kind': 'batch', 'records': records, 'dimensions': dimensions,
                             'deleted_records': preflight.get('deleted_records', 0) if isinstance(preflight, dict) else 0,
                             'fields': [{'name': str(name), 'dtype': str(dtype)} for name, dtype in zip(meta['fields'], meta['dtypes'])]})
                active_count += len(fids)
                # The Shapefile driver's SetNextByIndex uses physical DBF/SHP
                # slots, while its iterator skips deleted rows. Counting only
                # yielded rows here repeats features after the first deletion.
                offset = int(fids[-1]) + 1 if driver == 'ESRI Shapefile' else offset + len(fids)
                if len(fids) < BATCH_FEATURES:
                    break
            if driver == 'GeoJSON' and offset != len(preflight['original_features']):
                raise ImportFailure('GeoJSON reader omitted source features.', code='feature_count_mismatch')
            if driver == 'ESRI Shapefile' and active_count != preflight['physical_records'] - preflight['deleted_records']:
                raise ImportFailure('Shapefile reader omitted active source records.', code='feature_count_mismatch')
            if driver == 'GPKG' and active_count != next(item['feature_count'] for item in preflight if item['name'] == layer):
                raise ImportFailure('GeoPackage reader omitted source records.', code='feature_count_mismatch')
        _send(pipe, {'kind': 'done'})
    except BaseException as exc:
        try:
            _send(pipe, {'kind': 'error', 'code': getattr(exc, 'code', 'reader_failed'),
                         'message': str(exc).replace(str(path), Path(path).name)})
        except (OSError, BrokenPipeError):
            pass
    finally:
        if attributes_db is not None:
            attributes_db.close()
        pipe.close()


def run_native(path, driver, command, layer=None, context=None):
    """Yield reader messages. Cancellation terminates native work before cleanup."""
    check(context)
    multiprocessing_context = multiprocessing.get_context('spawn')
    parent, child = multiprocessing_context.Pipe(duplex=False)
    process = multiprocessing_context.Process(target=_entry,
        args=(child, str(path), driver, command, layer), daemon=True)
    process.start()
    child.close()
    completed = False
    try:
        while True:
            check(context)
            if parent.poll(.05):
                try:
                    message = json.loads(parent.recv_bytes(MAX_IPC_BYTES))
                except (EOFError, OSError, ValueError) as exc:
                    raise ImportFailure('GIS reader ended without a valid result.', code='reader_failed') from exc
                if message['kind'] == 'done':
                    completed = True
                    break
                if message['kind'] == 'error':
                    raise ImportFailure(message['message'], code=message['code'])
                yield message
            elif not process.is_alive():
                raise ImportFailure('GIS reader exited unexpectedly.', code='reader_failed')
    finally:
        parent.close()
        if not completed and process.is_alive():
            process.terminate()
        process.join(.2)
        if process.is_alive():
            process.kill()
            process.join(2)
        process.close()
