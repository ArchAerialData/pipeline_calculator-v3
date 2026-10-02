"""Folder counts must partition valid pins without conflating source identities."""
from pipeline_calculator.core.placemarks import (
    copy_folder_provenance, folder_display_fields, summarize_placemarks,
)


def pin(path, folder_id, source='doc.kml'):
    return {'folder_path': path, 'folder_name': path[-1] if path else '',
            'folder_id': folder_id, 'source_kml': source, 'Count': 1}


def test_totals_partition_nested_duplicate_and_linked_folders():
    records = [pin(['Facilities'], 'folder-1'),
               pin(['Facilities', 'Valves'], 'folder-2'),
               pin(['Facilities', 'Valves'], 'folder-2'),
               pin(['Facilities', 'Valves'], 'folder-3'),
               pin(['Facilities', 'Valves'], 'folder-2', 'child.kml'),
               pin([], ''), {}]
    summary = summarize_placemarks(records)
    assert summary['total'] == 7
    assert summary['folder_count'] == 4
    assert [g['count'] for g in summary['groups']] == [1, 2, 1, 1, 1, 1]
    assert sum(g['count'] for g in summary['groups']) == summary['total']
    assert [g['name'] for g in summary['groups']] == [
        'Facilities', 'Valves', 'Valves', 'Valves', 'No subfolder', 'Folder not recorded']
    summary['groups'][1]['folder_path'].clear()
    assert records[1]['folder_path'] == ['Facilities', 'Valves']


def test_missing_inventory_is_distinct_from_zero():
    assert summarize_placemarks(None) == {'total': None, 'groups': [], 'folder_count': 0}
    assert summarize_placemarks([]) == {'total': 0, 'groups': [], 'folder_count': 0}


def test_display_keeps_nested_names_and_legacy_provenance_distinct():
    assert folder_display_fields(pin(['Antero', 'Valves'], 'folder-2')) == ('Valves', 'Antero / Valves', 'doc.kml')
    assert folder_display_fields(pin([], '')) == ('No subfolder', '', 'doc.kml')
    assert folder_display_fields({}) == ('Folder not recorded', '', '')


def test_copied_paths_do_not_alias_source_or_invent_legacy_provenance():
    original = pin(['Facilities'], 'folder-1')
    copied = copy_folder_provenance(original)
    copied['folder_path'].append('Valves')
    assert original['folder_path'] == ['Facilities']
    assert copy_folder_provenance({}) == {}
