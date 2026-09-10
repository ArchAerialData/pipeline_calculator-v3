from __future__ import annotations

from tkinter import ttk

import customtkinter as ctk


def create(parent, current_results: dict, *, on_open_corridor) -> None:
    overlap = current_results.get("overlap_analysis") or {}

    main_frame = ctk.CTkFrame(parent)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkLabel(
        main_frame,
        text="Bundled Pipeline Sections",
        font=("Arial", 16, "bold"),
    ).pack(pady=(10, 0))

    bundled_sections = overlap.get("bundled_sections") or []
    if bundled_sections:
        table_frame = ctk.CTkFrame(main_frame)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass
        style.configure(
            "Overlap.Treeview",
            background="#2b2b2b",
            foreground="white",
            fieldbackground="#2b2b2b",
            rowheight=26,
        )
        style.configure("Overlap.Treeview.Heading", font=("Arial", 12, "bold"))

        columns = ("Pipeline Pair", "Length (miles)", "Avg Sep (m)", "Action")
        tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=20,
            style="Overlap.Treeview",
        )
        try:
            tree.column("#0", width=0, stretch=False)
        except Exception:
            pass

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        hsb.pack(side="bottom", fill="x")
        tree.configure(yscrollcommand=vsb.set)

        tree.heading("Pipeline Pair", text="Pipeline Pair")
        tree.heading("Length (miles)", text="Length (miles)")
        tree.heading("Avg Sep (m)", text="Avg Sep (m)")
        tree.heading("Action", text="Action")

        tree.column("Pipeline Pair", width=700, anchor="w")
        tree.column("Length (miles)", width=150, anchor="center")
        tree.column("Avg Sep (m)", width=120, anchor="center")
        tree.column("Action", width=110, anchor="center")

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

        action_buttons = {}

        def ensure_buttons_positioned(event=None):
            try:
                for item_id in tree.get_children(""):
                    bbox = tree.bbox(item_id, column="Action")
                    btn = action_buttons.get(item_id)
                    if not bbox:
                        if btn:
                            btn.place_forget()
                        continue
                    x, y, w, h = bbox
                    if x < 0 or x + w > tree.winfo_width():
                        if btn:
                            btn.place_forget()
                        continue
                    if btn is None:
                        section, idx = item_map[item_id]

                        def make_cmd(s=section, i=idx):
                            return lambda: on_open_corridor(s, i)

                        btn = ctk.CTkButton(
                            table_frame,
                            text="View Corridor",
                            width=min(110, max(80, w - 8)),
                            height=min(26, max(22, h - 6)),
                            command=make_cmd(),
                        )
                        action_buttons[item_id] = btn
                    btn.place(
                        x=tree.winfo_x() + x + (w // 2),
                        y=tree.winfo_y() + y + (h // 2),
                        anchor="center",
                    )
            except Exception:
                pass

        def on_tree_scroll(first, last):
            try:
                vsb.set(first, last)
            finally:
                ensure_buttons_positioned()

        tree.configure(yscrollcommand=on_tree_scroll)
        def on_horizontal_scroll(first, last):
            hsb.set(first, last)
            ensure_buttons_positioned()
        tree.configure(xscrollcommand=on_horizontal_scroll)
        tree.bind("<Configure>", ensure_buttons_positioned)
        tree.bind("<ButtonRelease-1>", ensure_buttons_positioned, add="+")
        tree.bind("<Motion>", lambda e: None)
        try:
            tree.after(100, ensure_buttons_positioned)
        except Exception:
            pass

        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        navigation = ctk.CTkFrame(main_frame)
        navigation.pack(fill="x", padx=10, pady=5)
        page_label = ctk.CTkLabel(navigation, text="")
        page_label.pack(side="left", padx=10)

        def load_page(delta=0):
            nonlocal page
            last_page = (len(bundled_sections) - 1) // page_size
            page = max(0, min(last_page, page + delta))
            for button in action_buttons.values():
                button.destroy()
            action_buttons.clear()
            item_map.clear()
            for item_id in tree.get_children(""):
                tree.delete(item_id)
            first = page * page_size
            for index in range(first, min(first + page_size, len(bundled_sections))):
                section = bundled_sections[index]
                item_id = tree.insert("", "end", values=(
                    f"{section.get('pipeline_1')} + {section.get('pipeline_2')}",
                    f"{section.get('bundled_length_miles', 0.0):.3f}",
                    f"{section.get('average_separation', 0.0):.1f}", "",
                ))
                item_map[item_id] = (section, index + 1)
            tree.yview_moveto(0)
            page_label.configure(text=f"Showing {first + 1}-{min(first + page_size, len(bundled_sections))} of {len(bundled_sections)} sections")
            previous.configure(state="normal" if page else "disabled")
            next_button.configure(state="normal" if page < last_page else "disabled")
            tree.after_idle(ensure_buttons_positioned)

        next_button = ctk.CTkButton(navigation, text="Next", width=90, command=lambda: load_page(1))
        next_button.pack(side="right", padx=5)
        previous = ctk.CTkButton(navigation, text="Previous", width=90, command=lambda: load_page(-1))
        previous.pack(side="right", padx=5)
        load_page()

        summary_frame = ctk.CTkFrame(main_frame)
        summary_frame.pack(fill="x", pady=10)
        total_bundled = sum(float(s.get("bundled_length_miles", 0.0)) for s in bundled_sections)
        ctk.CTkLabel(
            summary_frame,
            text=f"Pairwise Bundled Length: {total_bundled:.3f} miles across {len(bundled_sections)} sections (not total mileage removed)",
            font=("Arial", 12, "bold"),
        ).pack()
        ctk.CTkLabel(summary_frame, text='Corridors are sampled approximations. KML descriptions identify rectangle fallbacks.',
                     wraplength=650).pack(pady=5)
    else:
        ctk.CTkLabel(
            main_frame,
            text="No bundled sections found with current parameters",
            font=("Arial", 12),
        ).pack(pady=20)

