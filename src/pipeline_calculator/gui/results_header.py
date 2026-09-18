"""Responsive results context: scope selection beside persistent input notices."""
from __future__ import annotations

from tkinter import Misc, TclError, messagebox, ttk

import customtkinter as ctk

from pipeline_calculator.gui.bindings import ConfigureBinding
from pipeline_calculator.gui.layout import ResultPages, WrappedLabel
from pipeline_calculator.gui.repair_ui import _keyboard_button
from pipeline_calculator.gui.styles import scope_style


class _StatusButton(ctk.CTkButton):
    def focus_set(self):
        # Modal restoration must return to the widget owning the key bindings;
        # CTkButton otherwise focuses its internal, mouse-only text label.
        Misc.focus_set(self)


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
        self._repair_workflow = repair_workflow
        self._repair_source = repair_workflow.source if repair_workflow is not None else None
        # This header describes the displayed result, which can outlive the
        # workflow source while another import starts behind a modal panel.
        self._repair_rules = tuple(self._repair_source.report.get('rules') or ()) if self._repair_source is not None else ()
        self._repair_analysis_state = repair_workflow.analysis_state if repair_workflow is not None else None
        self._compact = False
        self._root_binding = ConfigureBinding(self._callback_host, self._root_configured)
        self.scope = None
        self.selector = None
        self.view_label = None
        self.scope_helper = None
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
            self.scope_helper = WrappedLabel(self.scope, text='Use the dropdown to view individual state statistics.',
                         text_color='#B8C8D8', font=('Arial', 13), anchor='w', justify='left',
                         wrap_padding=8)
            self.scope_helper.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 4))
            self.selector.bind('<<ComboboxSelected>>', lambda event: on_select(selection.get()))

        # Each notice owns its packed contents; only these outer buckets use grid.
        if repair_workflow is not None:
            self.repair_notice = repair_workflow.add_notice(self.notices)
        if state_preference is not None:
            state_preference.add_notice(self.notices, pady=(4, 0))
            self._preference_variable = state_preference.notice
            self._preference_trace = self._preference_variable.trace_add('write', self._queue_layout)
        self.compact_notices = ctk.CTkFrame(self, fg_color='transparent', width=1, height=1)
        self.repair_button = None
        if self.repair_notice is not None:
            self.repair_button = _StatusButton(self.compact_notices, text='', width=1, height=28,
                font=ctk.CTkFont(size=12), fg_color='#304D3E', hover_color='#3D634F',
                text_color='#B7E7C6', command=self._show_repair_details)
            _keyboard_button(self.repair_button)
        self.preference_button = _StatusButton(self.compact_notices, text='Setting not saved',
            width=1, height=28, font=ctk.CTkFont(size=12), fg_color='#5A4525',
            hover_color='#705732', text_color='#FFE1A6', command=self._show_preference_warning)
        _keyboard_button(self.preference_button)
        self.bind('<Configure>', self._queue_layout, add='+')
        self.notices.bind('<Configure>', self._queue_layout, add='+')
        self._arrange()

    def _root_configured(self, event):
        if event.widget is self._callback_host:
            self._queue_layout()

    def _show_preference_warning(self):
        if self._preference_variable is not None and self._preference_variable.get():
            messagebox.showwarning('State breakdown setting', self._preference_variable.get(),
                                   parent=self._callback_host)

    def _repair_status(self):
        recognized = self._repair_rules == ('filename_format_mismatch_v1',)
        root_scale = ctk.ScalingTracker.get_window_scaling(self._callback_host)
        short = self._callback_host.winfo_height() / root_scale < 280
        text = ('Recognized' if recognized else 'Repaired') if short else ('File type recognized' if recognized else 'File repaired')
        suffix = {'failed': 'Analysis failed', 'cancelled': 'Analysis cancelled'}
        return text + ' · ' + suffix.get(self._repair_analysis_state, 'Details')

    def _repair_available(self):
        workflow = self._repair_workflow
        return (workflow is not None and not workflow.closed and self._repair_source is not None
                and workflow.source is self._repair_source and workflow.verified)

    def _show_repair_details(self):
        if self._repair_available():
            self._repair_workflow.show_details()

    def _context_budget(self, factor):
        """Reserve result navigation/content after the other fixed root rows."""
        fixed = 0
        for sibling in self.master.winfo_children():
            if sibling is self or isinstance(sibling, ResultPages) or sibling.winfo_manager() != 'pack':
                continue
            padding = sibling.pack_info().get('pady', 0)
            padding = (padding,) if isinstance(padding, (int, float)) else self.tk.splitlist(padding)
            fixed += sibling.winfo_reqheight() + (2 * int(padding[0]) if len(padding) == 1
                                                  else sum(map(int, padding)))
        # Eight logical pixels are our pack padding; ten belong to ResultPages.
        return self.master.winfo_height() - fixed - (100 + 18) * factor

    @staticmethod
    def _contains(parent, widget):
        while widget is not None:
            if widget is parent:
                return True
            widget = widget.master
        return False

    @staticmethod
    def _first_button(parent):
        for child in parent.winfo_children():
            if isinstance(child, ctk.CTkButton):
                return child
            found = ResultsContextHeader._first_button(child)
            if found is not None:
                return found
        return None

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
        # Measure the expanded content even while its original widgets are hidden.
        # Root pack otherwise clips these fixed rows before results get any space.
        scope_height = (max(self.view_label.winfo_reqheight(), self.selector.winfo_reqheight() + 8 * factor)
                        + self.scope_helper.winfo_reqheight() + 4 * factor if self.scope is not None else 0)
        full_scope_width = max(360, (self.view_label.winfo_reqwidth() +
                                    self.selector.winfo_reqwidth()) / factor + 16) if self.scope else 0
        full_beside = self.scope is not None and has_notices and available >= full_scope_width + 24 + 440
        notice_height = self.notices.winfo_reqheight() if has_notices else 0
        expanded_height = (max(scope_height, notice_height) if full_beside else
                           scope_height + notice_height + (4 * factor if self.scope and has_notices else 0))
        compact = expanded_height > self._context_budget(factor)
        warning = self._preference_variable.get() if self._preference_variable is not None else ''
        repair_text = self._repair_status() if self.repair_button is not None else ''
        if self.repair_button is not None:
            self.repair_button.configure(state='normal' if self._repair_available() else 'disabled')
        if self.selector is not None:
            self.selector.configure(width=16 if compact else 25)
        scope_width = ((self.view_label.winfo_reqwidth() + self.selector.winfo_reqwidth()) / factor + 16
                       if self.scope is not None else 0)
        if not compact:
            scope_width = max(360, scope_width) if self.scope else 0
        compact_width = 0
        for button, text in ((self.repair_button, repair_text), (self.preference_button, warning)):
            if button is not None and text:
                if button is self.repair_button:
                    button.configure(text=text)
                width = button._text_label.winfo_reqwidth() / factor + 24
                button.configure(width=width)
                compact_width += width + 6
        side_by_side = self.scope is not None and has_notices and available >= scope_width + (12 + compact_width if compact else 24 + 440)
        stack_buttons = compact_width > available
        key = (has_notices, side_by_side, compact, stack_buttons, factor, round(scope_width), warning, repair_text)
        if key == self._layout_key:
            return
        self._layout_key = key
        try:
            focused = self.focus_get()
        except (KeyError, TclError):
            focused = None
        was_compact = self._compact
        self._compact = compact
        self.grid_columnconfigure(0, weight=0 if side_by_side else 1,
                                  minsize=round(scope_width * factor) if side_by_side else 0)
        self.grid_columnconfigure(1, weight=1 if side_by_side else 0, minsize=0)
        if self.scope is not None:
            if compact:
                self.scope_helper.grid_remove()
            else:
                self.scope_helper.grid()
            self.scope.grid(row=0, column=0, sticky='new' if side_by_side else 'nw',
                            padx=0, pady=0)
        self.notices.grid_remove()
        self.compact_notices.grid_remove()
        if has_notices:
            target = self.compact_notices if compact else self.notices
            if self.repair_button is not None:
                self.repair_button.pack(side='top' if stack_buttons else 'left', anchor='w',
                                        padx=(0, 6), pady=(0, 4) if stack_buttons else 0)
            if warning:
                self.preference_button.pack(side='top' if stack_buttons else 'left', anchor='w')
            else:
                self.preference_button.pack_forget()
            target.grid(row=0 if side_by_side or self.scope is None else 1,
                              column=1 if side_by_side else 0, sticky='new',
                              padx=(12 if compact else 24, 0) if side_by_side else 0,
                              pady=(0, 0) if side_by_side or self.scope is None else (4, 0))
        if compact != was_compact and focused is not None:
            hidden = self.notices if compact else self.compact_notices
            if self._contains(hidden, focused):
                replacement = ((self.repair_button or self.preference_button) if compact else
                               self._first_button(self.notices) or self.selector or self._first_button(self.master))
                if replacement is not None:
                    Misc.focus_set(replacement)

    def destroy(self):
        self._root_binding.close()
        if self._layout_id is not None:
            self._callback_host.after_cancel(self._layout_id)
            self._layout_id = None
        if self._preference_trace is not None:
            self._preference_variable.trace_remove('write', self._preference_trace)
            self._preference_trace = None
        super().destroy()
