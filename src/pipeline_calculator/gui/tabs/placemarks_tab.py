from __future__ import annotations

from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.table_loading import load_rows
from pipeline_calculator.gui.layout import WrappedLabel


def create(parent, current_results: dict) -> None:
    placemarks = current_results.get('placemarks')
    total = f'{len(placemarks):,}' if isinstance(placemarks, list) else 'Not recorded'
    WrappedLabel(parent, text=f'Point pins: {total}', font=('Arial', 18, 'bold'),
                 anchor='w').pack(fill='x', padx=12, pady=(10, 2))
    columns = ("ID", "Name", "Count")
    tree = create_table(parent, columns, (150, 400, 100))

    def populate():
        for placemark in placemarks or []:
            tree.insert(
                "",
                "end",
                values=(
                    placemark.get("Placemark_ID"),
                    placemark.get("Name"),
                    placemark.get("Count", 1),
                ),
            )
            yield
    load_rows(tree, populate())
