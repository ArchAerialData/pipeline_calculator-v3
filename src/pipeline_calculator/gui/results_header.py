"""Responsive results context: scope selection beside persistent input notices."""
from __future__ import annotations

from tkinter import ttk

import customtkinter as ctk

from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.styles import scope_style


class ResultsContextHeader(ctk.CTkFrame):
    """Keep context outside the replaceable result tabs and collapse empty space."""

    def __init__(self, parent, *, after, scopes, selection, on_select,
                 repair_workflow=None, state_preference=None):
        super().__init__(parent, fg_color='transparent', width=1, height=1)
        self._pack_after = after
        self._callback_host = self.winfo_toplevel()
        self._layout_id = None
        self._layout_key = None
        self._scale = None
        self._preference_trace = None
        self._preference_variable = None
        self.scope = None
        self.selector = None
        self.view_label = None
        self.notices = ctk.CTkFrame(self, fg_color='transparent', width=1, height=1)
        self.repair_notice = None

        if scopes is not None:
            self.scope = ctk.CTkFrame(self, fg_color='transparent', width=1, height=1)
            self.scope.grid_columnconfigure(1, weight=1)
            self.view_label = ctk.CTkLabel(self.scope, text='View:', text_color='#A9D9FF',
                                          font=('Arial', 14, 'bold'))
            self.view_label.grid(row=0, column=0, padx=(0, 8), sticky='w')
            self.selector = ttk.Combobox(self.scope, textvariable=selection, values=list(scopes),
                                         state='readonly', width=25, takefocus=True)
            self.selector.grid(row=0, column=1, sticky='ew', padx=(0, 8), pady=4)
            WrappedLabel(self.scope, text='Use the dropdown to view individual state statistics.',
                         text_color='#B8C8D8', font=('Arial', 13), anchor='w', justify='left',
                         wrap_padding=8).grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 4))
            self.selector.bind('<<ComboboxSelected>>', lambda event: on_select(selection.get()))

        # Each notice owns its packed contents; only these outer buckets use grid.
        if repair_workflow is not None:
            self.repair_notice = repair_workflow.add_notice(self.notices)
        if state_preference is not None:
            state_preference.add_notice(self.notices, pady=(4, 0))
            self._preference_variable = state_preference.notice
            self._preference_trace = self._preference_variable.trace_add('write', self._queue_layout)
        self.bind('<Configure>', self._queue_layout, add='+')
        self.notices.bind('<Configure>', self._queue_layout, add='+')
        self._arrange()

    def _queue_layout(self, *args):
        if self._layout_id is None:
            self._layout_id = self._callback_host.after(20, self._arrange)

    def _style_selector(self, factor):
        if self.selector is None or self._scale == factor:
            return
        self._scale = factor
        font = ('Arial', -round(14 * factor))
        self.selector.configure(font=font, style=scope_style(self.selector, factor))
        popup = self.selector.tk.call('ttk::combobox::PopdownWindow', self.selector)
        self.selector.tk.call(f'{popup}.f.l', 'configure', '-font', font,
                              '-background', '#242424', '-foreground', '#F1F4F8',
                              '-selectbackground', '#1F538D', '-selectforeground', '#FFFFFF')

    def _arrange(self):
        self._layout_id = None
        factor = ctk.ScalingTracker.get_widget_scaling(self)
        self._style_selector(factor)
        has_notices = any(child.winfo_manager() for child in self.notices.winfo_children())
        if self.scope is None and not has_notices:
            self.pack_forget()
            self._layout_key = None
            return
        if not self.winfo_manager():
            # The file row lives for this entire results page, unlike replaced tabs.
            self.pack(after=self._pack_after, fill='x', padx=18, pady=(4, 4))
        available = self.winfo_width() / factor
        scope_width = (max(360, (self.view_label.winfo_reqwidth() +
                                self.selector.winfo_reqwidth()) / factor + 16)
                       if self.scope is not None else 0)
        side_by_side = self.scope is not None and has_notices and available >= scope_width + 24 + 440
        key = (has_notices, side_by_side, factor, round(scope_width))
        if key == self._layout_key:
            return
        self._layout_key = key
        self.grid_columnconfigure(0, weight=0 if side_by_side else 1,
                                  minsize=round(scope_width * factor) if side_by_side else 0)
        self.grid_columnconfigure(1, weight=1 if side_by_side else 0, minsize=0)
        if self.scope is not None:
            self.scope.grid(row=0, column=0, sticky='new' if side_by_side else 'nw',
                            padx=0, pady=0)
        if has_notices:
            self.notices.grid(row=0 if side_by_side or self.scope is None else 1,
                              column=1 if side_by_side else 0, sticky='new',
                              padx=(24, 0) if side_by_side else 0,
                              pady=(0, 0) if side_by_side or self.scope is None else (4, 0))
        else:
            self.notices.grid_remove()

    def destroy(self):
        if self._layout_id is not None:
            self._callback_host.after_cancel(self._layout_id)
            self._layout_id = None
        if self._preference_trace is not None:
            self._preference_variable.trace_remove('write', self._preference_trace)
            self._preference_trace = None
        super().destroy()
