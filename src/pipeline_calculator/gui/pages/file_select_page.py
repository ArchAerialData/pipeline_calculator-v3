from __future__ import annotations

import customtkinter as ctk
from tkinter import Canvas, messagebox
from tkinterdnd2 import DND_FILES
from pipeline_calculator.gui.layout import ActionBar, WrappedLabel, parameter_fields
from pipeline_calculator.gui.settings_panel import SettingsPanel


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
    browse_area = ctk.CTkFrame(main_frame, fg_color="#202D38", border_color="#4B91C2", border_width=2)
    browse_area.pack(fill="x", padx=12, pady=10)
    WrappedLabel(browse_area, text="Choose a KMZ or KML file", font=("Arial", 18, "bold")).pack(
        fill="x", padx=12, pady=(14, 0))
    WrappedLabel(browse_area, text="Use Browse Files to select a file from your computer.",
                 text_color="#B8C0CC").pack(fill="x", padx=12, pady=2)
    file_actions(browse_area, on_browse, on_file_selected, retry_path)

    # Reserve a usable drop target before settings; extra height goes here.
    drop_zone = ctk.CTkFrame(main_frame, height=80, fg_color="#3A3A3A",
                            corner_radius=0, border_width=0)
    drop_zone.pack(fill="both", expand=True, padx=12, pady=(0, 10))
    drop_zone.pack_propagate(False)
    # Native canvas supplies a dotted outline; CTk frames only support solid borders.
    drop_border = Canvas(drop_zone, background="#3A3A3A", highlightthickness=0, borderwidth=0)
    drop_border.place(x=0, y=0, relwidth=1, relheight=1)
    outline = drop_border.create_rectangle(0, 0, 1, 1, outline="#5FA5D5")

    def resize_border(event):
        scale = ctk.ScalingTracker.get_widget_scaling(drop_zone)
        inset = max(2, round(3 * scale))
        drop_border.coords(outline, inset, inset, max(inset, event.width - inset),
                           max(inset, event.height - inset))
        drop_border.itemconfigure(outline, width=max(1, round(2 * scale)),
                                  dash=(max(1, round(2 * scale)), max(2, round(4 * scale))))

    drop_border.bind("<Configure>", resize_border)
    instructions = ctk.CTkFrame(drop_zone, fg_color="transparent", height=64)
    instructions.pack(expand=True, fill="x", padx=8)
    WrappedLabel(instructions, text="Drag and drop your file here", font=("Arial", 18, "bold")).pack(fill="x")
    WrappedLabel(instructions, text="Drop one KMZ or KML file into this area to start analysis.",
                 text_color="#B8C0CC").pack(fill="x")

    body = SettingsPanel(main_frame)
    body.pack(side="bottom", fill="x")
    WrappedLabel(body, text="Analysis Settings", font=("Arial", 16, "bold"), anchor="w").pack(
        fill="x", padx=12, pady=(8, 2))
    parameter_fields(body, (detection_range_var, segment_length_var, min_parallel_var, angular_tolerance_var),
                     compact=True)

    browse_labels = [child for child in browse_area.winfo_children() if isinstance(child, WrappedLabel)]
    browse_actions = next(child for child in browse_area.winfo_children() if isinstance(child, ActionBar))
    compact = [False]

    def fit_short_window(event=None):
        height = event.height if event else main_frame.winfo_height()
        short = height / ctk.ScalingTracker.get_widget_scaling(main_frame) < 440
        if short == compact[0]:
            return
        compact[0] = short
        for index, label in enumerate(browse_labels):
            if short:
                label.pack_forget()
            else:
                label.pack(fill="x", padx=12, pady=(14, 0) if index == 0 else 2, before=browse_actions)

    main_frame.bind("<Configure>", fit_short_window, add="+")
    fit_short_window()

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
        drop_border.drop_target_register(DND_FILES)
        drop_border.dnd_bind("<<Drop>>", on_drop)
    except Exception:
        # Drag/drop is best-effort; Browse works everywhere.
        pass


def file_actions(parent, on_browse, on_file_selected, retry_path=None):
    actions = [('Browse Files', on_browse)]
    if retry_path:
        actions.append(('Retry selected file', lambda: on_file_selected(retry_path)))
    frame = ActionBar(parent, actions)
    frame.pack(side='bottom', fill='x', padx=8, pady=8)
