from __future__ import annotations

import os

import customtkinter as ctk
from pipeline_calculator.gui.layout import ActionBar, ResultPages

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

    for widget in root.winfo_children():
        widget.destroy()

    root.title(f"Pipeline Calculator v{version} - Results")
    ActionBar(root, [("Export Results", on_export), ("Adjust Parameters", on_reanalyze),
                     ("Import New File", on_new_file), ("Exit", on_exit)]).pack(
                         side="bottom", fill="x", padx=10, pady=6)

    header_frame = ctk.CTkFrame(root)
    header_frame.pack(fill="x", padx=10, pady=5)

    file_name = os.path.basename(current_file) if current_file else ""
    ctk.CTkLabel(header_frame, text="File:").pack(side="left", padx=(8, 4))
    file_label = ctk.CTkEntry(header_frame)
    file_label.insert(0, file_name)
    file_label.configure(state="readonly")
    file_label.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4)

    tabview = ResultPages(root)
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

