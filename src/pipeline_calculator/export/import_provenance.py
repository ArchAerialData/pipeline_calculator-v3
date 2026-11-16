"""Format-neutral source columns and coordinate-operation audit for exports."""
from __future__ import annotations

import json


SOURCE_HEADERS = ['Source Dataset', 'Source Layer', 'Source Feature ID', 'Source Format', 'Source Component']


def has_gis_sources(results):
    return 'import_metadata' in results or any(
        'source_dataset' in row for key in ('pipelines', 'placemarks')
        for row in (results.get(key) or []))


def source_values(record):
    component = record.get('source_component_path')
    return [record.get('source_dataset', ''), record.get('source_layer', ''),
            str(record.get('source_feature_id', '')), record.get('source_format', ''),
            json.dumps(component) if component is not None else '']


def add_import_details(workbook, results):
    metadata = results.get('import_metadata')
    if not isinstance(metadata, dict):
        return
    from openpyxl.styles import Alignment, Font, PatternFill

    sheet = workbook.create_sheet('Import Details')
    sheet.append(['Detail', 'Value'])
    sheet.freeze_panes = 'A2'
    sheet.column_dimensions['A'].width = 55
    sheet.column_dimensions['B'].width = 110

    def append(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                append(child, f'{path}.{key}' if path else str(key))
        elif isinstance(value, (list, tuple)):
            if not value:
                sheet.append([path, '[]'])
            for index, child in enumerate(value):
                append(child, f'{path}[{index}]')
        else:
            sheet.append([path, value])

    append(metadata, '')
    for cell in sheet[1]:
        cell.font = Font(name='Aptos Display', bold=True, color='FFFFFFFF')
        cell.fill = PatternFill('solid', fgColor='FF285D73')
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical='top', wrap_text=True)
            if cell.data_type == 'f':
                cell.data_type = 's'
    sheet.auto_filter.ref = sheet.dimensions
