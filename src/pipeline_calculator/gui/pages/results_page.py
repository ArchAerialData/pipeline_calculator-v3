from __future__ import annotations

import os

import customtkinter as ctk

from pipeline_calculator.gui.tabs.overlap_tab import create as create_overlap_tab
from pipeline_calculator.gui.tabs.pipelines_tab import create as create_pipelines_tab
from pipeline_calculator.gui.tabs.placemarks_tab import create as create_placemarks_tab
from pipeline_calculator.gui.tabs.diagnostics_tab import create as create_diagnostics_tab
from pipeline_calculator.gui.tabs.summary_tab import create as create_summary_tab


def show(
    root,
    *,
    version: str,
    current_file: str | None,
    current_results: dict,
    on_export,
    on_reanalyze,
    on_new_file,
    on_exit,
    on_open_corridor,
) -> None:
    """Render the results screen."""

    root.deiconify()
    try:
        root.focus_force()
        root.attributes("-alpha", 1.0)
        root.lift()
        root.attributes("-topmost", False)
        root.configure(bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
    except Exception:
        pass

    root.update_idletasks()

    for widget in root.winfo_children():
        widget.destroy()

    root.title(f"Pipeline Calculator v{version} - Results")
    root.geometry("1200x800")

    header_frame = ctk.CTkFrame(root)
    header_frame.pack(fill="x", padx=10, pady=5)

    file_name = os.path.basename(current_file) if current_file else ""
    file_label = ctk.CTkLabel(header_frame, text=f"File: {file_name}", font=("Arial", 12))
    file_label.pack()

    tabview = ctk.CTkTabview(root)
    tabview.pack(fill="both", expand=True, padx=10, pady=5)

    summary_tab = tabview.add("Summary")
    create_summary_tab(summary_tab, current_results)

    if current_results.get("pipelines"):
        pipelines_tab = tabview.add("Pipelines")
        create_pipelines_tab(pipelines_tab, current_results)

    if current_results.get("overlap_analysis"):
        overlap_tab = tabview.add("Overlap Analysis")
        create_overlap_tab(overlap_tab, current_results, on_open_corridor=on_open_corridor)

    if current_results.get("placemarks"):
        placemark_tab = tabview.add("Placemarks")
        create_placemarks_tab(placemark_tab, current_results)

    if current_results.get("diagnostics"):
        diagnostics_tab = tabview.add("Diagnostics")
        create_diagnostics_tab(diagnostics_tab, current_results)

    button_frame = ctk.CTkFrame(root)
    button_frame.pack(fill="x", padx=10, pady=5)

    export_button = ctk.CTkButton(button_frame, text="Export Results", command=on_export)
    export_button.pack(side="left", padx=5)

    reanalyze_button = ctk.CTkButton(
        button_frame,
        text="Reanalyze with Different Parameters",
        command=on_reanalyze,
    )
    reanalyze_button.pack(side="left", padx=5)

    new_file_button = ctk.CTkButton(button_frame, text="Import New KMZ", command=on_new_file)
    new_file_button.pack(side="left", padx=5)

    close_button = ctk.CTkButton(button_frame, text="Exit", command=on_exit)
    close_button.pack(side="right", padx=5)

