from __future__ import annotations

from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.table_loading import load_rows
from pipeline_calculator.gui.sorting import HeaderSorter, sort_records
from pipeline_calculator.gui.tabs.summary_tab import number
from pipeline_calculator.core.constants import SURVEY_MILE_METERS


def create(parent, current_results: dict) -> None:
    if current_results.get('parsed_sources') or any('source_dataset' in row for row in current_results.get('pipelines', [])):
        return create_gis_pipelines(parent, current_results)
    if current_results.get('state_code'):
        return create_state_pipelines(parent, current_results)
    columns = ("Placemark ID", "Name", "Length (m)", "Length (miles)")
    tree = create_table(parent, columns, (150, 300, 150, 150))
    records = []

    def populate():
        for pipeline in current_results.get("pipelines", []):
            item = tree.insert(
                "",
                "end",
                values=(
                    pipeline.get("Placemark_ID") or "N/A",
                    pipeline.get("Name"),
                    f"{pipeline.get('Shape_Length', 0.0):.3f}",
                    f"{pipeline.get('pipelinelength', 0.0):.6f}",
                ),
            )
            records.append((item, pipeline))
            yield

        total = tree.insert(
            "",
            "end",
            values=(
                "TOTAL",
                "TOTAL",
                f"{current_results.get('total_meters', 0.0):.3f}",
                f"{current_results.get('total_miles', 0.0):.6f}",
            ),
        )
        fields = dict(zip(columns, ('Placemark_ID', 'Name', 'Shape_Length', 'pipelinelength')))
        def sort(column, descending):
            ordered = sort_records(records, lambda row: row[1].get(fields[column]),
                                   numeric=column in ('Length (m)', 'Length (miles)'), descending=descending)
            tree.set_children('', *(item for item, _ in ordered), total)
            tree.yview_moveto(0)
        tree.sorter = HeaderSorter(tree, columns, sort)
    load_rows(tree, populate())


def create_gis_pipelines(parent, current_results):
    """Keep layer-scoped feature IDs distinguishable in GIS and state results."""
    state_view = bool(current_results.get('state_code'))
    identity_columns = ('Feature ID', 'Name', 'Layer', 'Source dataset')
    numeric_columns = (('Interior (mi)', 'Shared allocation (mi)', 'Attributed (mi)') if state_view else
                       ('Length (m)', 'Length (miles)'))
    numeric_keys = (('interior_meters', 'shared_allocation_meters', 'Shape_Length') if state_view else
                    ('Shape_Length', 'pipelinelength'))
    columns = identity_columns + numeric_columns
    tree = create_table(parent, columns, (160, 260, 220, 300) + (150,) * len(numeric_columns))
    fields = dict(zip(columns, ('source_feature_id', 'Name', 'source_layer', 'source_dataset') + numeric_keys))
    records = []

    def numeric_values(row):
        return tuple(number(row.get(key, 0) / SURVEY_MILE_METERS) if state_view else
                     f"{row.get(key, 0):.{3 if key == 'Shape_Length' else 6}f}" for key in numeric_keys)

    def populate():
        for row in current_results.get('pipelines') or []:
            feature_id = row.get('source_feature_id')
            item = tree.insert('', 'end', values=(
                str(feature_id) if feature_id is not None else row.get('Placemark_ID') or 'N/A',
                row.get('Name', ''), row.get('source_layer', ''), row.get('source_dataset', ''),
                *numeric_values(row)))
            records.append((item, row))
            yield
        totals = dict(current_results, Shape_Length=current_results.get('total_meters', 0),
                      pipelinelength=current_results.get('total_miles', 0))
        total = tree.insert('', 'end', values=('TOTAL', 'TOTAL', '', '', *numeric_values(totals)))

        def sort(column, descending):
            ordered = sort_records(records, lambda pair: pair[1].get(fields[column]),
                numeric=column in numeric_columns, descending=descending)
            tree.set_children('', *(item for item, _ in ordered), total)
            tree.yview_moveto(0)

        tree.sorter = HeaderSorter(tree, columns, sort)
    load_rows(tree, populate())


def create_state_pipelines(parent, current_results):
    columns = ('Placemark ID', 'Name', 'Interior (mi)', 'Shared allocation (mi)', 'Attributed (mi)')
    tree = create_table(parent, columns, (150, 260, 150, 180, 150))
    fields = dict(zip(columns, ('Placemark_ID', 'Name', 'interior_meters', 'shared_allocation_meters', 'Shape_Length')))
    records = []
    def populate():
        for pipeline in current_results.get('pipelines') or []:
            item = tree.insert('', 'end', values=(pipeline.get('Placemark_ID') or 'N/A', pipeline.get('Name'),
                *(number(pipeline.get(key, 0) / SURVEY_MILE_METERS) for key in
                  ('interior_meters', 'shared_allocation_meters', 'Shape_Length'))))
            records.append((item, pipeline))
            yield
        total = tree.insert('', 'end', values=('TOTAL', 'TOTAL',
            *(number(current_results.get(key, 0) / SURVEY_MILE_METERS) for key in
              ('interior_meters', 'shared_allocation_meters', 'total_meters'))))
        def sort(column, descending):
            ordered = sort_records(records, lambda row: row[1].get(fields[column]),
                                   numeric=column not in ('Placemark ID', 'Name'), descending=descending)
            tree.set_children('', *(item for item, _ in ordered), total)
            tree.yview_moveto(0)
        tree.sorter = HeaderSorter(tree, columns, sort)
    load_rows(tree, populate())
