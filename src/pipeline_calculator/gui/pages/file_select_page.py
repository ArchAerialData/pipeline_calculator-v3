from __future__ import annotations

import customtkinter as ctk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES


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

    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=("Arial", 24, "bold"),
    )
    title_label.pack(pady=20)

    instructions = ctk.CTkLabel(
        main_frame,
        text="Drag and drop a KMZ or KML file here\n\nOR\n\nClick Browse to select a file",
        font=("Arial", 14),
        justify="center",
    )
    instructions.pack(pady=20)

    params_frame = ctk.CTkFrame(main_frame)
    params_frame.pack(pady=20, padx=40, fill="x")

    ctk.CTkLabel(params_frame, text="Analysis Parameters", font=("Arial", 16, "bold")).pack(pady=10)

    detection_frame = ctk.CTkFrame(params_frame)
    detection_frame.pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(detection_frame, text="Detection Range (m):").pack(side="left", padx=10)
    ctk.CTkEntry(detection_frame, textvariable=detection_range_var, width=100).pack(side="left")
    ctk.CTkLabel(
        detection_frame,
        text="(Max centerline separation to bundle)",
        text_color="#888888",
    ).pack(side="left", padx=10)

    seglen_frame = ctk.CTkFrame(params_frame)
    seglen_frame.pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(seglen_frame, text="Segment Length (m):").pack(side="left", padx=10)
    ctk.CTkEntry(seglen_frame, textvariable=segment_length_var, width=100).pack(side="left")
    ctk.CTkLabel(
        seglen_frame,
        text="(Resolution; smaller = slower, finer)",
        text_color="#888888",
    ).pack(side="left", padx=10)

    parallel_frame = ctk.CTkFrame(params_frame)
    parallel_frame.pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(parallel_frame, text="Min Parallel Length (m):").pack(side="left", padx=10)
    ctk.CTkEntry(parallel_frame, textvariable=min_parallel_var, width=100).pack(side="left")
    ctk.CTkLabel(
        parallel_frame,
        text="(Min bundled section)",
        text_color="#888888",
    ).pack(side="left", padx=10)

    angular_frame = ctk.CTkFrame(params_frame)
    angular_frame.pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(angular_frame, text="Angular Tolerance (°):").pack(side="left", padx=10)
    ctk.CTkEntry(angular_frame, textvariable=angular_tolerance_var, width=100).pack(side="left")
    ctk.CTkLabel(
        angular_frame,
        text="(Max angle difference)",
        text_color="#888888",
    ).pack(side="left", padx=10)

    file_actions(main_frame, on_browse, on_file_selected, retry_path)

    def on_drop(event):
        try:
            file_path = event.data.strip("{}").strip('"')
            if file_path.lower().endswith((".kmz", ".kml")):
                on_file_selected(file_path)
            else:
                messagebox.showerror("Invalid File", "Please select a KMZ or KML file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")

    try:
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", on_drop)
    except Exception:
        # Drag/drop is best-effort; Browse works everywhere.
        pass


def file_actions(parent, on_browse, on_file_selected, retry_path=None):
    frame = ctk.CTkFrame(parent)
    frame.pack(pady=20)
    ctk.CTkButton(frame, text='Browse Files', command=on_browse, width=200, height=40).pack(side='left', padx=5)
    if retry_path:
        ctk.CTkButton(frame, text='Retry selected file', command=lambda: on_file_selected(retry_path),
                     width=200, height=40).pack(side='left', padx=5)

