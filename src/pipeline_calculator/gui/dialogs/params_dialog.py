from __future__ import annotations

import customtkinter as ctk


class ParamsDialog:
    def __init__(
        self,
        root,
        *,
        detection_range_var,
        segment_length_var,
        min_parallel_var,
        angular_tolerance_var,
        on_apply,
        on_cancel=None,
    ) -> None:
        self._root = root
        self._on_apply = on_apply
        self._on_cancel = on_cancel

        self.frame = ctk.CTkFrame(root, corner_radius=10)

        ctk.CTkLabel(
            self.frame,
            text="Adjust Analysis Parameters",
            font=("Arial", 16, "bold"),
        ).pack(pady=10, padx=20)

        detection_frame = ctk.CTkFrame(self.frame)
        detection_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(detection_frame, text="Detection Range (m):").pack(side="left", padx=10)
        ctk.CTkEntry(detection_frame, textvariable=detection_range_var).pack(side="left")

        seglen_frame = ctk.CTkFrame(self.frame)
        seglen_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(seglen_frame, text="Segment Length (m):").pack(side="left", padx=10)
        ctk.CTkEntry(seglen_frame, textvariable=segment_length_var).pack(side="left")

        parallel_frame = ctk.CTkFrame(self.frame)
        parallel_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(parallel_frame, text="Min Parallel Length (m):").pack(side="left", padx=10)
        ctk.CTkEntry(parallel_frame, textvariable=min_parallel_var).pack(side="left")

        angular_frame = ctk.CTkFrame(self.frame)
        angular_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(angular_frame, text="Angular Tolerance (°):").pack(side="left", padx=10)
        ctk.CTkEntry(angular_frame, textvariable=angular_tolerance_var).pack(side="left")

        button_frame = ctk.CTkFrame(self.frame)
        button_frame.pack(pady=20)

        ctk.CTkButton(button_frame, text="Apply & Reanalyze", command=self._apply).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=self._cancel).pack(side="left", padx=5)

    def show(self) -> None:
        self.frame.place(relx=0.5, rely=0.5, anchor="center")

    def close(self) -> None:
        try:
            self.frame.destroy()
        except Exception:
            pass

    def _apply(self) -> None:
        self.close()
        try:
            self._on_apply()
        except Exception:
            pass

    def _cancel(self) -> None:
        self.close()
        if self._on_cancel is None:
            return
        try:
            self._on_cancel()
        except Exception:
            pass

