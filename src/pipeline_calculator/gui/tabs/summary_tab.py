from __future__ import annotations

import customtkinter as ctk
from pipeline_calculator.gui.layout import WrappedLabel

from pipeline_calculator.core.constants import SEGMENT_LENGTH


def add_status_notice(parent, current_results: dict) -> None:
    errors = [d.get("message", "Analysis error") for d in current_results.get("diagnostics", [])
              if d.get("level") == "error"]
    if errors or current_results.get("analysis_complete") is False:
        messages = list(dict.fromkeys(errors))
        detail = "\n".join(messages[:3])
        if len(messages) > 3:
            detail += f"\n{len(messages) - 3} more issue(s); see Diagnostics or the exported workbook."
        WrappedLabel(
            parent,
            text="Analysis incomplete. Totals cover loaded, valid geometry only.\n" + detail,
            text_color="#FF8080", font=("Arial", 14, "bold"), wraplength=1000,
        ).pack(fill="x", pady=10)


def create(parent, current_results: dict) -> None:
    summary_frame = ctk.CTkScrollableFrame(parent)
    summary_frame.pack(fill="both", expand=True, padx=20, pady=20)

    WrappedLabel(summary_frame, text="Analysis Summary", font=("Arial", 20, "bold")).pack(fill="x", pady=10)
    add_status_notice(summary_frame, current_results)

    original_frame = ctk.CTkFrame(summary_frame)
    original_frame.pack(fill="x", pady=10)

    WrappedLabel(
        original_frame,
        text="Original Pipeline Totals",
        font=("Arial", 16, "bold"),
        text_color="#FFD700",
    ).pack(fill="x")

    total_miles = current_results.get("total_miles", 0.0)
    WrappedLabel(
        original_frame,
        text=f"Total Length: {total_miles:.3f} US Survey Miles",
        font=("Arial", 14),
    ).pack(fill="x")

    diagnostics = current_results.get("diagnostics", []) or []
    if diagnostics:
        warning_count = sum(1 for d in diagnostics if d.get("level") != "info")
        WrappedLabel(
            original_frame,
            text=f"Analysis Diagnostics: {len(diagnostics)} total, {warning_count} warning(s)/error(s)",
            font=("Arial", 14),
            text_color="#FFD700" if warning_count else "#AAAAAA",
        ).pack(fill="x")

    WrappedLabel(
        original_frame,
        text=f"Pipeline Count: {len(current_results.get('pipelines', []))}",
        font=("Arial", 14),
    ).pack(fill="x")

    if current_results.get("overlap_analysis"):
        overlap = current_results["overlap_analysis"]

        adjusted_frame = ctk.CTkFrame(summary_frame)
        adjusted_frame.pack(fill="x", pady=10)

        WrappedLabel(
            adjusted_frame,
            text="Adjusted for Overlaps",
            font=("Arial", 16, "bold"),
            text_color="#87CEEB",
        ).pack(fill="x")

        effective_miles = overlap.get("effective_total_miles", 0.0)
        WrappedLabel(
            adjusted_frame,
            text=f"Effective Survey Length (Adjusted Mileage): {effective_miles:.3f} US Survey Miles",
            font=("Arial", 14),
            text_color="#90EE90",
        ).pack(fill="x")

        savings_miles = overlap.get("savings_miles", 0.0)
        savings_pct = overlap.get("savings_percentage", 0.0)
        WrappedLabel(
            adjusted_frame,
            text=f"Mileage Removed: {savings_miles:.3f} miles ({savings_pct:.1f}%)",
            font=("Arial", 14),
            text_color="white",
        ).pack(fill="x")

        bundle_count = len(overlap.get("bundled_sections", []))
        WrappedLabel(
            adjusted_frame,
            text=f"Bundled Sections: {bundle_count}",
            font=("Arial", 14),
        ).pack(fill="x")

    sections = (current_results.get("overlap_analysis") or {}).get("bundled_sections") or []
    if sections:
        total_bundled = sum(float(section.get("bundled_length_miles", 0.0)) for section in sections)
        WrappedLabel(summary_frame, text=f"Pairwise bundled length: {total_bundled:.3f} miles "
                     "(not total mileage removed). Corridors are approximate visual guides; "
                     "KML descriptions identify rectangle fallbacks.", text_color="#B8C0CC").pack(fill="x", pady=8)

    params_frame = ctk.CTkFrame(summary_frame)
    params_frame.pack(fill="x", pady=10)

    WrappedLabel(params_frame, text="Analysis Parameters Used", font=("Arial", 16, "bold")).pack(fill="x")

    params = current_results.get("analysis_parameters", {})
    param_text = f"Detection Range: {params.get('detection_range', '')} m\n"
    param_text += f"Min Parallel Length: {params.get('min_parallel_length', '')} m\n"
    param_text += f"Segment Length: {params.get('segment_length', SEGMENT_LENGTH)} m\n"
    param_text += f"Angular Tolerance: {params.get('angular_tolerance', '')} deg"
    WrappedLabel(params_frame, text=param_text, font=("Arial", 12)).pack(fill="x")
