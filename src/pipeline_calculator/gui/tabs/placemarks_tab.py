from __future__ import annotations

from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.table_loading import load_rows


def create(parent, current_results: dict) -> None:
    columns = ("ID", "Name", "Count")
    tree = create_table(parent, columns, (150, 400, 100))

    def populate():
        for placemark in current_results.get("placemarks", []):
            tree.insert(
                "",
                "end",
                values=(
                    placemark.get("Placemark_ID"),
                    placemark.get("Name"),
                    placemark.get("Count"),
                ),
            )
            yield
    load_rows(tree, populate())
