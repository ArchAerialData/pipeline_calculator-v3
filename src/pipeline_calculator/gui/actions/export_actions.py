from __future__ import annotations

import json
import os
from typing import Any

from pipeline_calculator.export.xlsx import build_analysis_workbook


def default_export_filename(current_file: str | None) -> str:
    if not current_file:
        return "analysis.xlsx"
    base_name = os.path.splitext(os.path.basename(current_file))[0]
    return f"{base_name}_analysis.xlsx"


def export_results_to_path(current_results: dict[str, Any], save_path: str) -> None:
    if save_path.lower().endswith(".json"):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(current_results, f, indent=2, default=str)
        return

    wb = build_analysis_workbook(current_results)
    wb.save(save_path)


def export_with_dialog(current_results: dict[str, Any], current_file: str | None) -> str | None:
    """UI helper: pick a path and write results, showing messageboxes on success/failure."""
    from tkinter import filedialog, messagebox

    if isinstance(current_results.get("geography"), dict):
        return _export_package_with_dialog(current_results, current_file)

    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        initialfile=default_export_filename(current_file),
        filetypes=[("Excel Workbook", "*.xlsx"), ("JSON files", "*.json")],
    )
    if not save_path:
        return None

    try:
        export_results_to_path(current_results, save_path)
    except Exception as e:
        messagebox.showerror("Export Error", str(e))
        return None

    messagebox.showinfo("Export Complete", f"Results exported to:\n{save_path}")
    return save_path


def _export_package_with_dialog(current_results, current_file):
    """One modal package dialog; all file I/O runs outside Tk's event loop."""
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import customtkinter as ctk
    from pipeline_calculator.export.package import export_analysis_package
    from pipeline_calculator.gui.layout import WrappedLabel
    from pipeline_calculator.gui.window import fit_window

    parent = tk._default_root
    window = ctk.CTkToplevel(parent)
    window.title("Export Analysis Package")
    if parent is not None:
        window.transient(parent)
    content = ctk.CTkScrollableFrame(window)
    content.pack(fill="both", expand=True, padx=16, pady=16)
    WrappedLabel(content, text="Export every state and the combined analysis together.",
                 font=ctk.CTkFont(size=18, weight="bold"), justify="left").pack(fill="x", padx=8, pady=(8, 12))
    WrappedLabel(content, text="Excel workbook included. A new named folder will keep the selected files together.",
                 justify="left").pack(fill="x", padx=8, pady=(0, 14))
    maps = tk.BooleanVar(window, value=True)
    include_json = tk.BooleanVar(window, value=False)
    map_box = ctk.CTkCheckBox(content, text="Include KMZ maps", variable=maps)
    map_box.pack(anchor="w", padx=8, pady=8)
    WrappedLabel(content, text="Combined map and a folder for each state with interior geometry.",
                 justify="left").pack(fill="x", padx=8, pady=(0, 12))
    json_box = ctk.CTkCheckBox(content, text="Include JSON data", variable=include_json)
    json_box.pack(anchor="w", padx=8, pady=8)
    status = WrappedLabel(content, text="", justify="left")
    status.pack(fill="x", padx=8, pady=8)
    if not current_results["geography"].get("fragments"):
        maps.set(False)
        map_box.configure(state="disabled")
        status.configure(text="State map geometry is unavailable. The workbook includes available results and diagnostics.")
    progress = ctk.CTkProgressBar(content, mode="indeterminate")
    actions = ctk.CTkFrame(window, fg_color="transparent")
    actions.pack(side="bottom", fill="x", padx=24, pady=(0, 16))
    outcome = {"path": None, "error": None, "busy": False}

    def close():
        if not outcome["busy"]:
            window.destroy()

    def poll(done):
        if not window.winfo_exists():
            return
        if not done.is_set():
            window.after(100, lambda: poll(done))
            return
        outcome["busy"] = False
        progress.stop()
        progress.pack_forget()
        if outcome["error"] is not None:
            status.configure(text=f"Export failed: {outcome['error']}")
            export_button.configure(state="normal")
            cancel_button.configure(state="normal")
            json_box.configure(state="normal")
            if current_results["geography"].get("fragments"):
                map_box.configure(state="normal")
        else:
            window.destroy()

    def export():
        selected = filedialog.askdirectory(parent=window, title="Choose a folder for the analysis package", mustexist=True)
        if not selected:
            return
        options = {"include_maps": maps.get(), "include_json": include_json.get()}
        outcome.update(busy=True, error=None)
        for control in (export_button, cancel_button, map_box, json_box):
            control.configure(state="disabled")
        status.configure(text="Writing the workbook and selected files…")
        progress.pack(fill="x", padx=8, pady=8)
        progress.start()
        done = threading.Event()

        def worker():
            try:
                outcome["path"] = str(export_analysis_package(current_results, selected, current_file, **options))
            except Exception as error:
                outcome["error"] = str(error)
            finally:
                done.set()

        try:
            threading.Thread(target=worker, daemon=True).start()
        except RuntimeError as error:
            outcome["error"] = str(error)
            done.set()
        poll(done)

    cancel_button = ctk.CTkButton(actions, text="Cancel", command=close, width=95)
    cancel_button.pack(side="left")
    export_button = ctk.CTkButton(actions, text="Choose Folder & Export", command=export)
    export_button.pack(side="right", padx=(8, 0))
    fit_window(window, (560, 440), parent=parent)
    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Escape>", lambda event: close())
    window.wait_visibility()
    window.grab_set()
    window.wait_window()
    if outcome["path"] is not None:
        messagebox.showinfo("Export Complete", f"Analysis package exported to:\n{outcome['path']}", parent=parent)
    return outcome["path"]

