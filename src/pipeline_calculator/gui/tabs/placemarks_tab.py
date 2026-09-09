from __future__ import annotations

from tkinter import ttk


def create(parent, current_results: dict) -> None:
    columns = ("ID", "Name", "Count")
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)

    tree.heading("ID", text="Placemark ID")
    tree.heading("Name", text="Name")
    tree.heading("Count", text="Count")

    tree.column("ID", width=150)
    tree.column("Name", width=400)
    tree.column("Count", width=100)

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

    tree.pack(fill="both", expand=True, padx=10, pady=10)

