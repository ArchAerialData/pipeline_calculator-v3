from __future__ import annotations

import customtkinter as ctk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES
from pipeline_calculator.gui.layout import ActionBar, WrappedLabel, parameter_fields


def show(
    root,
    *,
    title: str,
    detection_range_var,
    min_parallel_var,
    segment_length_var,
    angular_tolerance_var,
    on_browse,
    on_file_selected,
    retry_path=None,
) -> None:
    """Render the file selection screen."""

    for widget in root.winfo_children():
        widget.destroy()

    main_frame = ctk.CTkFrame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)

    WrappedLabel(main_frame, text=title, font=("Arial", 22, "bold")).pack(fill="x", padx=12, pady=(12, 4))
    drop_zone = ctk.CTkFrame(main_frame, fg_color="#202D38", border_color="#4B91C2", border_width=2)
    drop_zone.pack(fill="x", padx=12, pady=10)
    WrappedLabel(drop_zone, text="Drop your KMZ or KML file here", font=("Arial", 18, "bold")).pack(
        fill="x", padx=12, pady=(14, 0))
    WrappedLabel(drop_zone, text="Drag a file into this box, or choose Browse Files below.",
                 text_color="#B8C0CC").pack(fill="x", padx=12, pady=2)
    file_actions(drop_zone, on_browse, on_file_selected, retry_path)

    body = ctk.CTkScrollableFrame(main_frame, fg_color="#202020")
    body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    WrappedLabel(body, text="Analysis Settings", font=("Arial", 16, "bold"), anchor="w").pack(
        fill="x", padx=12, pady=(8, 2))
    parameter_fields(body, (detection_range_var, segment_length_var, min_parallel_var, angular_tolerance_var),
                     compact=True)

    def on_drop(event):
        try:
            paths = root.tk.splitlist(event.data)
            if len(paths) != 1:
                messagebox.showerror("Select One File", "Please drop one KMZ or KML file at a time.", parent=root)
                return
            file_path = paths[0]
            if file_path.lower().endswith((".kmz", ".kml")):
                on_file_selected(file_path)
            else:
                messagebox.showerror("Invalid File", "Please select a KMZ or KML file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")

    try:
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", on_drop)
        drop_zone.drop_target_register(DND_FILES)
        drop_zone.dnd_bind("<<Drop>>", on_drop)
    except Exception:
        # Drag/drop is best-effort; Browse works everywhere.
        pass


def file_actions(parent, on_browse, on_file_selected, retry_path=None):
    actions = [('Browse Files', on_browse)]
    if retry_path:
        actions.append(('Retry selected file', lambda: on_file_selected(retry_path)))
    frame = ActionBar(parent, actions)
    frame.pack(side='bottom', fill='x', padx=8, pady=8)
