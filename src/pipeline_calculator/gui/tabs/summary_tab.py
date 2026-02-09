from __future__ import annotations

import customtkinter as ctk

from pipeline_calculator.core.constants import SEGMENT_LENGTH


def create(parent, current_results: dict) -> None:
    summary_frame = ctk.CTkScrollableFrame(parent)
    summary_frame.pack(fill="both", expand=True, padx=20, pady=20)

    ctk.CTkLabel(summary_frame, text="Analysis Summary", font=("Arial", 20, "bold")).pack(pady=10)

    original_frame = ctk.CTkFrame(summary_frame)
    original_frame.pack(fill="x", pady=10)

    ctk.CTkLabel(
        original_frame,
        text="Original Pipeline Totals",
        font=("Arial", 16, "bold"),
        text_color="#FFD700",
    ).pack()

    total_miles = current_results.get("total_miles", 0.0)
    ctk.CTkLabel(
        original_frame,
        text=f"Total Length: {total_miles:.3f} US Survey Miles",
        font=("Arial", 14),
    ).pack()

    ctk.CTkLabel(
        original_frame,
        text=f"Pipeline Count: {len(current_results.get('pipelines', []))}",
        font=("Arial", 14),
    ).pack()

    if current_results.get("overlap_analysis"):
        overlap = current_results["overlap_analysis"]

        adjusted_frame = ctk.CTkFrame(summary_frame)
        adjusted_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            adjusted_frame,
            text="Adjusted for Overlaps",
            font=("Arial", 16, "bold"),
            text_color="#87CEEB",
        ).pack()

        effective_miles = overlap.get("effective_total_miles", 0.0)
        ctk.CTkLabel(
            adjusted_frame,
            text=f"Effective Survey Length: {effective_miles:.3f} US Survey Miles",
            font=("Arial", 14),
        ).pack()

        savings_miles = overlap.get("savings_miles", 0.0)
        savings_pct = overlap.get("savings_percentage", 0.0)
        ctk.CTkLabel(
            adjusted_frame,
            text=f"Survey Savings: {savings_miles:.3f} miles ({savings_pct:.1f}%)",
            font=("Arial", 14),
            text_color="#90EE90",
        ).pack()

        bundle_count = len(overlap.get("bundled_sections", []))
        ctk.CTkLabel(
            adjusted_frame,
            text=f"Bundled Sections: {bundle_count}",
            font=("Arial", 14),
        ).pack()

    params_frame = ctk.CTkFrame(summary_frame)
    params_frame.pack(fill="x", pady=10)

    ctk.CTkLabel(params_frame, text="Analysis Parameters Used", font=("Arial", 16, "bold")).pack()

    params = current_results.get("analysis_parameters", {})
    param_text = f"Detection Range: {params.get('detection_range', '')} m\n"
    param_text += f"Min Parallel Length: {params.get('min_parallel_length', '')} m\n"
    param_text += f"Angular Tolerance: {params.get('angular_tolerance', '')}°"

    ctk.CTkLabel(params_frame, text=param_text, font=("Arial", 12)).pack()

    params2 = current_results.get("analysis_parameters", {})
    param_text2 = f"Segment Length: {params2.get('segment_length', SEGMENT_LENGTH)} m\n"
    param_text2 += f"Angular Tolerance: {params2.get('angular_tolerance', '')} deg"
    ctk.CTkLabel(params_frame, text=param_text2, font=("Arial", 12), text_color="#AAAAAA").pack()

