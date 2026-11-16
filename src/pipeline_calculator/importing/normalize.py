"""Normalize supported features without joining paths or interpreting outlines."""
import base64
from collections import Counter
import json
import math

import shapely
from shapely import from_wkb

from .coordinates import CoordinateOperation
from .limits import (MAX_FEATURES, MAX_VERTICES, MAX_COLLECTION_DEPTH,
                     MAX_ATTRIBUTE_BYTES, MAX_DIAGNOSTICS, check, report)
from .models import ImportFailure, VectorParseResult


class Normalizer:
    def __init__(self, source_digest, context=None):
        self.source_digest = source_digest
        self.context = context
        self.result = VectorParseResult()
        self.coverage = Counter()
        self.layers = []
        self.vertex_count = 0
        self.attribute_bytes = 0

    def diagnostic(self, code, message, *, level='error', **details):
        if len(self.result.diagnostics) >= MAX_DIAGNOSTICS:
            raise ImportFailure('Input exceeds the retained diagnostics budget.', code='diagnostic_limit')
        self.result.diagnostics.append({'level': level, 'code': code, 'message': message, 'context': details})

    def _geometry(self, geometry, operation, component=(), depth=0):
        check(self.context)
        if depth > MAX_COLLECTION_DEPTH:
            raise ImportFailure('Geometry exceeds the collection nesting budget.', code='geometry_depth_limit')
        kind = geometry.geom_type
        if geometry.is_empty:
            raise ImportFailure('Feature contains empty geometry.', code='empty_geometry')
        if kind == 'Point':
            coords = operation.path(list(geometry.coords))
            return [], [(coords[0], list(component), operation.last_operation_id)], 0
        if kind in ('LineString', 'LinearRing'):
            if kind == 'LinearRing':
                return [], [], 1
            coords = list(geometry.coords)
            if len(coords) < 2:
                raise ImportFailure('Line has fewer than two vertices.', code='invalid_geometry')
            transformed = operation.path(coords)
            return [(transformed, list(component), operation.last_operation_id)], [], 0
        if kind in ('Polygon', 'MultiPolygon'):
            return [], [], 1 if kind == 'Polygon' else len(geometry.geoms)
        if kind in ('MultiLineString', 'MultiPoint', 'GeometryCollection'):
            lines, points, polygons = [], [], 0
            for index, child in enumerate(geometry.geoms):
                child_lines, child_points, child_polygons = self._geometry(child, operation, (*component, index), depth + 1)
                lines.extend(child_lines)
                points.extend(child_points)
                polygons += child_polygons
            return lines, points, polygons
        raise ImportFailure('Unsupported source geometry: ' + kind, code='unsupported_geometry')

    def add_layer(self, descriptor, packets, crs, *, assignment=None):
        operation = CoordinateOperation(crs)
        coverage = Counter()
        fields = []
        dimensions = set(descriptor.get('dimensions', []))
        deleted_records = 0
        for packet in packets:
            if packet['kind'] != 'batch':
                continue
            fields = packet.get('fields', fields)
            dimensions.update(packet.get('dimensions', []))
            deleted_records = packet.get('deleted_records', deleted_records)
            for record in packet['records']:
                check(self.context)
                coverage['source_features'] += 1
                self.coverage['source_features'] += 1
                if self.coverage['source_features'] > MAX_FEATURES:
                    raise ImportFailure('Selected layers exceed the feature budget.', code='feature_limit')
                report(self.context, 'Normalizing GIS features', self.coverage['source_features'])
                attrs = record['attributes']
                self.attribute_bytes += len(json.dumps(attrs, ensure_ascii=False).encode('utf-8'))
                if self.attribute_bytes > MAX_ATTRIBUTE_BYTES:
                    raise ImportFailure('Selected attributes exceed the retained byte budget.', code='attribute_limit')
                source_id = str(record['fid'])
                provenance = {'source_dataset': descriptor['dataset'], 'source_layer': descriptor['name'],
                    'source_feature_id': source_id, 'source_component_path': [], 'source_format': descriptor['format'],
                    'source_record_index': record.get('record_index'),
                    'source_crs': operation.source.to_wkt(), 'attributes': attrs,
                    'source_dimensions': sorted(dimensions),
                    'internal_id': f'{self.source_digest}/{descriptor["id"]}/{source_id}:{record.get("record_index", 0)}'}
                if record.get('original_feature_id') is not None:
                    provenance['source_original_feature_id'] = record['original_feature_id']
                if record['wkb'] is None:
                    coverage['null_geometry'] += 1
                    self.diagnostic('null_geometry', 'Source feature has no geometry and is excluded from analysis.',
                                    source_dataset=descriptor['dataset'], source_layer=descriptor['name'], source_feature_id=source_id)
                    continue
                try:
                    geometry = from_wkb(base64.b64decode(record['wkb']))
                    self.vertex_count += int(shapely.get_num_coordinates(geometry))
                    if self.vertex_count > MAX_VERTICES:
                        raise ImportFailure('Selected geometry exceeds the vertex budget.', code='vertex_limit')
                    if getattr(geometry, 'has_z', False):
                        dimensions.add('Z')
                    if getattr(geometry, 'has_m', False):
                        dimensions.add('M')
                    provenance['source_dimensions'] = sorted(dimensions)
                    lines, points, polygons = self._geometry(geometry, operation)
                except ImportFailure as exc:
                    if exc.code.endswith('_limit') or exc.code in {'unavailable_coordinate_operation', 'coordinate_operation_area'}:
                        raise
                    coverage[exc.code] += 1
                    self.diagnostic(exc.code, str(exc), source_dataset=descriptor['dataset'],
                                    source_layer=descriptor['name'], source_feature_id=source_id)
                    continue
                except (ValueError, shapely.errors.GEOSException) as exc:
                    coverage['invalid_geometry'] += 1
                    self.diagnostic('invalid_geometry', 'Source feature cannot be decoded without changing its geometry.',
                                    source_dataset=descriptor['dataset'], source_layer=descriptor['name'], source_feature_id=source_id)
                    continue
                name = next((str(attrs[key]).strip() for key in ('Name', 'NAME', 'name', 'PLINE_ID')
                             if key in attrs and attrs[key] is not None and str(attrs[key]).strip()),
                            f'{descriptor["name"]} feature {source_id}')
                provenance['source_coordinate_operations'] = [
                    {'component_path': component, 'operation_id': operation_id}
                    for _, component, operation_id in (*lines, *points)]
                if lines:
                    paths = [path for path, _, _ in lines]
                    self.result.pipelines.append({'id': len(self.result.pipelines), 'placemark_id': 'N/A',
                        'objectid': attrs.get('OBJECTID', 'N/A'), 'name': name,
                        'coordinates': paths[0], 'coordinate_paths': paths,
                        'path_component_paths': [component for _, component, _ in lines], **provenance})
                    coverage['pipeline_features'] += 1
                    coverage['line_paths'] += len(paths)
                    coverage['line_vertices'] += sum(len(path) for path in paths)
                for coordinate, component, operation_id in points:
                    self.result.placemarks.append({'Placemark_ID': source_id, 'Name': name, 'Count': 1,
                        'coordinates': coordinate, **provenance, 'source_component_path': component,
                        'internal_id': provenance['internal_id'] + ':' + '.'.join(map(str, component))})
                    coverage['point_count'] += 1
                if polygons:
                    coverage['excluded_polygon_features'] += 1
                    coverage['excluded_polygon_parts'] += polygons
        if coverage['excluded_polygon_features']:
            self.diagnostic('polygon_excluded', 'Polygon geometry is excluded from centerline mileage and pin counts.', level='info',
                            source_dataset=descriptor['dataset'], source_layer=descriptor['name'],
                            feature_count=coverage['excluded_polygon_features'])
        if dimensions:
            self.diagnostic('horizontal_xy_only', 'Horizontal length and locations use XY only; elevation/measure values are not used.',
                            level='info', source_dataset=descriptor['dataset'], source_layer=descriptor['name'], dimensions=sorted(dimensions))
        if deleted_records:
            coverage['deleted_attribute_records'] = deleted_records
            self.diagnostic('deleted_attribute_records', 'Deleted shapefile attribute records are not active features.', level='info',
                            source_dataset=descriptor['dataset'], source_layer=descriptor['name'], count=deleted_records)
        coordinate_metadata = operation.metadata()
        coordinate_metadata['user_assignment'] = assignment
        layer_info = {'id': descriptor['id'], 'dataset': descriptor['dataset'], 'name': descriptor['name'],
                      'format': descriptor['format'], 'coverage': dict(coverage), 'fields': fields,
                      'dimensions': sorted(dimensions), 'coordinate_operation': coordinate_metadata}
        self.layers.append(layer_info)
        for key, value in coverage.items():
            if key != 'source_features':
                self.coverage[key] += value

    def finish(self):
        if not self.result.pipelines and not self.result.placemarks:
            self.diagnostic('no_supported_features', 'No supported line or point features were found in the selected layers.')
        self.result.parsed_sources = [{'dataset': layer['dataset'], 'layer': layer['name'], 'format': layer['format']} for layer in self.layers]
        self.result.import_metadata = {'schema_version': 1, 'source_sha256': self.source_digest,
            'selected_layers': [layer['id'] for layer in self.layers], 'layers': self.layers,
            'coverage': dict(self.coverage),
            'selected_input_complete': not any(item['level'] == 'error' for item in self.result.diagnostics),
            'coordinate_operations': [layer['coordinate_operation'] for layer in self.layers]}
        return self.result
