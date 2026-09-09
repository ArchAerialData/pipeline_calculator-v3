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

