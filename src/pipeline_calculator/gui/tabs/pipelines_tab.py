from __future__ import annotations

from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.sorting import HeaderSorter, sort_records


def create(parent, current_results: dict) -> None:
    columns = ("Placemark ID", "Name", "Length (m)", "Length (miles)")
    tree = create_table(parent, columns, (150, 300, 150, 150))
    records = []

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
        for position, (item, _) in enumerate(ordered):
            tree.move(item, '', position)
        tree.move(total, '', 'end')
        tree.yview_moveto(0)
    tree.sorter = HeaderSorter(tree, columns, sort)
