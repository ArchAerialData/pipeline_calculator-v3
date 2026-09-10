from __future__ import annotations

import customtkinter as ctk
from pipeline_calculator.gui.layout import ActionBar, WrappedLabel, parameter_fields


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

        ActionBar(self.frame, [("Apply & Reanalyze", self._apply), ("Cancel", self._cancel)]).pack(
            side="bottom", fill="x", padx=8, pady=8)
        body = ctk.CTkScrollableFrame(self.frame)
        body.pack(fill="both", expand=True, padx=8, pady=8)
        WrappedLabel(body, text="Adjust Analysis Parameters", font=("Arial", 16, "bold")).pack(fill="x", pady=10)
        parameter_fields(body, (detection_range_var, segment_length_var, min_parallel_var, angular_tolerance_var))

    def show(self) -> None:
        self.frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.94, relheight=0.9)

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

