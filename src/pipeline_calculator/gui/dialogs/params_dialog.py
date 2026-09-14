from __future__ import annotations

import customtkinter as ctk
from pipeline_calculator.gui.layout import ActionBar, WrappedLabel, parameter_fields
from pipeline_calculator.gui.modal import ModalSurface, ModalBody, TEXT


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

        self.surface = ModalSurface(root)
        self.frame = self.surface.card

        ActionBar(self.frame, [("Apply & Reanalyze", self._apply), ("Cancel", self._cancel)]).pack(
            side="bottom", fill="x", padx=8, pady=8)
        body = ModalBody(self.frame)
        body.pack(fill="both", expand=True, padx=20, pady=20)
        WrappedLabel(body, text="Adjust analysis parameters", text_color=TEXT, anchor='w', justify='left',
                     font=ctk.CTkFont(size=22, weight='bold')).pack(fill="x", padx=12, pady=(4, 14))
        parameter_fields(body, (detection_range_var, segment_length_var, min_parallel_var, angular_tolerance_var))

    def show(self) -> None:
        self.surface.show(preferred_size=(760, 560))

    def close(self) -> None:
        try:
            self.surface.destroy()
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

