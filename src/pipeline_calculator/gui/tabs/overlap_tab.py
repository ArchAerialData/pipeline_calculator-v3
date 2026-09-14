from __future__ import annotations

from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.layout import WrappedLabel

import customtkinter as ctk


def create(parent, current_results: dict, *, on_open_corridor) -> None:
    overlap = current_results.get("overlap_analysis") or {}

    main_frame = ctk.CTkFrame(parent)
    main_frame.pack(fill="both", expand=True, padx=4)

    bundled_sections = overlap.get("bundled_sections") or []
    if bundled_sections:
        navigation = ctk.CTkFrame(main_frame)
        navigation.pack(side="bottom", fill="x", padx=6)
        tree = create_table(main_frame, ("Pipeline Pair", "Length (miles)", "Avg Sep (m)", "Action"),
                            (420, 150, 120, 140), vertical_padding=0)

        page = 0
        page_size = 20
        item_map = {}

        def _open_for_item(item_id):
            if item_id not in item_map:
                return
            section, idx = item_map[item_id]
            on_open_corridor(section, idx)

        def on_click(event):
            region = tree.identify("region", event.x, event.y)
            if region != "cell":
                return
            row_id = tree.identify_row(event.y)
            col = tree.identify_column(event.x)
            if row_id and col == "#4":
                _open_for_item(row_id)

        def on_double_click(event):
            row_id = tree.identify_row(event.y)
            if row_id:
                _open_for_item(row_id)

        tree.bind("<ButtonRelease-1>", on_click)
        tree.bind("<Double-1>", on_double_click)

        def open_selected():
            selection = tree.selection()
            if selection:
                _open_for_item(selection[0])

        tree.bind("<Return>", lambda event: open_selected())

        def load_page(delta=0):
            nonlocal page
            last_page = (len(bundled_sections) - 1) // page_size
            page = max(0, min(last_page, page + delta))
            item_map.clear()
            for item_id in tree.get_children(""):
                tree.delete(item_id)
            first = page * page_size
            for index in range(first, min(first + page_size, len(bundled_sections))):
                section = bundled_sections[index]
                item_id = tree.insert("", "end", values=(
                    f"{section.get('pipeline_1')} + {section.get('pipeline_2')}",
                    f"{section.get('bundled_length_miles', 0.0):.3f}",
                    f"{section.get('average_separation', 0.0):.1f}", "View corridor",
                ))
                item_map[item_id] = (section, index + 1)
            tree.yview_moveto(0)
            tree.heading("Pipeline Pair", text=f"Pipeline Pair ({first + 1}-{min(first + page_size, len(bundled_sections))} of {len(bundled_sections)})")
            previous.configure(state="normal" if page else "disabled")
            next_button.configure(state="normal" if page < last_page else "disabled")
            children = tree.get_children("")
            if children:
                tree.selection_set(children[0])
                tree.focus(children[0])

        actions = ctk.CTkFrame(navigation, fg_color="transparent")
        actions.pack(fill="x", padx=4)
        actions.grid_columnconfigure(2, weight=1)
        previous = ctk.CTkButton(actions, text="Previous", width=72, command=lambda: load_page(-1))
        next_button = ctk.CTkButton(actions, text="Next", width=64, command=lambda: load_page(1))
        view = ctk.CTkButton(actions, text="View Corridor", width=110, command=open_selected)
        for column, button in enumerate((previous, next_button, view)):
            button.grid(row=0, column=column, sticky="ew", padx=3)
        load_page()

    else:
        WrappedLabel(
            main_frame,
            text="No bundled sections found with current parameters",
            font=("Arial", 12),
        ).pack(pady=20)

