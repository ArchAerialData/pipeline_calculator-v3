from __future__ import annotations

from pipeline_calculator.gui.tables import create_table


def create(parent, current_results: dict) -> None:
    columns = ("Level", "Code", "Message", "Context")
    tree = create_table(parent, columns, (90, 220, 520, 420))

    for diag in current_results.get("diagnostics", []) or []:
        tree.insert(
            "",
            "end",
            values=(
                diag.get("level", ""),
                diag.get("code", ""),
                diag.get("message", ""),
                str(diag.get("context", "")),
            ),
        )
