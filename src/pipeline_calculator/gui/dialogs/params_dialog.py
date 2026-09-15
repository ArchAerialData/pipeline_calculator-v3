from __future__ import annotations

import customtkinter as ctk
from tkinter import BooleanVar, messagebox
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
        state_preference=None,
    ) -> None:
        self._root = root
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        self._state_preference = state_preference
        self._state_draft = (BooleanVar(root, value=state_preference.variable.get())
                             if state_preference is not None else None)

        self.surface = ModalSurface(root)
        self.frame = self.surface.card

        ActionBar(self.frame, [("Apply & Reanalyze", self._apply), ("Cancel", self._cancel)]).pack(
            side="bottom", fill="x", padx=8, pady=8)
        body = ModalBody(self.frame)
        body.pack(fill="both", expand=True, padx=20, pady=20)
        WrappedLabel(body, text="Adjust analysis parameters", text_color=TEXT, anchor='w', justify='left',
                     font=ctk.CTkFont(size=22, weight='bold')).pack(fill="x", padx=12, pady=(4, 14))
        if state_preference is not None:
            state_preference.add_control(body, draft=self._state_draft)
        parameter_fields(body, (detection_range_var, segment_length_var, min_parallel_var, angular_tolerance_var))

    def show(self) -> None:
        self.surface.show(preferred_size=(760, 560))

    def close(self) -> None:
        try:
            self.surface.destroy()
        except Exception as error:
            messagebox.showerror('Could not close parameters', str(error), parent=self._root)

    def _apply(self) -> None:
        if self._state_preference is not None:
            self._state_preference.commit(self._state_draft.get())
        self.close()
        try:
            self._on_apply()
        except Exception as error:
            messagebox.showerror('Could not start analysis', str(error), parent=self._root)

    def _cancel(self) -> None:
        self.close()
        if self._on_cancel is None:
            return
        try:
            self._on_cancel()
        except Exception as error:
            messagebox.showerror('Could not return to results', str(error), parent=self._root)

