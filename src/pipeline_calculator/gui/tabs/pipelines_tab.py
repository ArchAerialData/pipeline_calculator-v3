from __future__ import annotations

from tkinter import ttk


def create(parent, current_results: dict) -> None:
    columns = ("OBJECTID", "Name", "Length (m)", "Length (miles)")
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)

    tree.heading("OBJECTID", text="Object ID")
    tree.heading("Name", text="Name")
    tree.heading("Length (m)", text="Length (meters)")
    tree.heading("Length (miles)", text="Length (miles)")

    tree.column("OBJECTID", width=100)
    tree.column("Name", width=300)
    tree.column("Length (m)", width=150)
    tree.column("Length (miles)", width=150)

    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
    )

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

    tree.pack(fill="both", expand=True, padx=10, pady=10)

