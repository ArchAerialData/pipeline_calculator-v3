"""Folder provenance and disjoint point inventories shared by UI and exports."""
from __future__ import annotations

from copy import deepcopy


SOURCE_FIELDS = (
    'source_dataset', 'source_layer', 'source_feature_id', 'source_component_path',
    'source_format', 'source_crs', 'source_dimensions', 'attributes', 'internal_id',
    'component_path', 'source_record_index', 'source_original_feature_id',
    'path_component_paths', 'source_coordinate_operations',
)


def copy_source_provenance(record):
    """Copy source identity without sharing mutable attributes across analyses."""
    copied = {key: deepcopy(record[key]) for key in SOURCE_FIELDS if key in record}
    copied.update({key: record[key] for key in ('folder_name', 'folder_id', 'source_kml') if key in record})
    if 'folder_path' in record:
        copied['folder_path'] = list(record['folder_path'])
    return copied


def copy_folder_provenance(record):
    """Compatibility name for source provenance, including original KML folders."""
    return copy_source_provenance(record)


def folder_display_fields(record):
    """Return the owning folder, readable full path, and source document."""
    if 'source_dataset' in record:
        return '', '', ''
    path = record.get('folder_path')
    source = record.get('source_kml', '')
    if not isinstance(path, (list, tuple)):
        return 'Folder not recorded', '', source
    if not path:
        return 'No subfolder', '', source
    return path[-1], ' / '.join(path), source


def source_display_fields(record):
    """Display a GIS layer or KML folder without conflating their identities."""
    if 'source_dataset' in record:
        layer = record.get('source_layer', '')
        return layer or 'Unnamed layer', layer, record['source_dataset']
    return folder_display_fields(record)


def summarize_placemarks(records):
    """Count valid Point records once, grouped by immediate owning folder.

    Ancestors are provenance, not additional groups. Folder IDs distinguish
    identically named sibling folders; source identity distinguishes linked KMLs.
    Legacy records without provenance are never represented as known root pins.
    """
    if not isinstance(records, list):
        return {'total': None, 'groups': [], 'folder_count': 0}
    groups = {}
    for record in records:
        if 'source_dataset' in record:
            dataset, layer = record['source_dataset'], record.get('source_layer', '')
            key = ('gis', dataset, layer)
            if key not in groups:
                groups[key] = {
                    'name': layer or 'Unnamed layer', 'grouping_kind': 'layer',
                    'source_dataset': dataset, 'source_layer': layer,
                    'folder_path': [], 'source_kml': '', 'folder_id': '', 'count': 0,
                }
            groups[key]['count'] += 1
            continue
        name, _, source = folder_display_fields(record)
        path = record.get('folder_path')
        known = isinstance(path, (list, tuple))
        path = list(path) if known else []
        folder_id = record.get('folder_id', '')
        key = (source, known, folder_id, tuple(path))
        if key not in groups:
            groups[key] = {'name': name, 'folder_path': path, 'source_kml': source,
                           'folder_id': folder_id, 'count': 0}
        groups[key]['count'] += 1
    result = {'total': len(records), 'groups': list(groups.values()),
              'folder_count': sum(bool(group['folder_path']) for group in groups.values())}
    layer_count = sum(group.get('grouping_kind') == 'layer' for group in groups.values())
    if layer_count:
        result['layer_count'] = layer_count
    return result
