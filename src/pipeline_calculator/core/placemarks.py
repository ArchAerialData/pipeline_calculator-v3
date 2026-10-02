"""Folder provenance and disjoint point inventories shared by UI and exports."""
from __future__ import annotations


def copy_folder_provenance(record):
    """Detach folder paths while keeping missing legacy provenance distinguishable."""
    copied = {key: record[key] for key in ('folder_name', 'folder_id', 'source_kml') if key in record}
    if 'folder_path' in record:
        copied['folder_path'] = list(record['folder_path'])
    return copied


def folder_display_fields(record):
    """Return the owning folder, readable full path, and source document."""
    path = record.get('folder_path')
    source = record.get('source_kml', '')
    if not isinstance(path, (list, tuple)):
        return 'Folder not recorded', '', source
    if not path:
        return 'No subfolder', '', source
    return path[-1], ' / '.join(path), source


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
    return {'total': len(records), 'groups': list(groups.values()),
            'folder_count': sum(bool(group['folder_path']) for group in groups.values())}
