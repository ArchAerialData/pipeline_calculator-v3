from __future__ import annotations

from tkinter import ttk


def create(parent, current_results: dict) -> None:
    columns = ("Level", "Code", "Message", "Context")
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)

    for column in columns:
        tree.heading(column, text=column)

    tree.column("Level", width=90)
    tree.column("Code", width=220)
    tree.column("Message", width=520)
    tree.column("Context", width=420)

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

    tree.pack(fill="both", expand=True, padx=10, pady=10)
