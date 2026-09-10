from __future__ import annotations

from pipeline_calculator.gui.tables import create_table


def create(parent, current_results: dict) -> None:
    columns = ("OBJECTID", "Name", "Length (m)", "Length (miles)")
    tree = create_table(parent, columns, (100, 300, 150, 150))

    for pipeline in current_results.get("pipelines", []):
        tree.insert(
            "",
            "end",
            values=(
                pipeline.get("OBJECTID"),
                pipeline.get("Name"),
                f"{pipeline.get('Shape_Length', 0.0):.3f}",
                f"{pipeline.get('pipelinelength', 0.0):.6f}",
            ),
        )

    tree.insert(
        "",
        "end",
        values=(
            "TOTAL",
            "TOTAL",
            f"{current_results.get('total_meters', 0.0):.3f}",
            f"{current_results.get('total_miles', 0.0):.6f}",
        ),
    )
