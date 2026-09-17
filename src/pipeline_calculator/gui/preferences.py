"""Small, versioned user preferences; analysis never depends on disk writes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import weakref


def preferences_path() -> Path:
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA') or Path.home() / 'AppData' / 'Roaming')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'PipelineCalculator' / 'preferences.json'


def load_state_breakdown(path: Path | None = None) -> bool:
    try:
        data = json.loads((path or preferences_path()).read_text(encoding='utf-8'))
        return (isinstance(data, dict) and type(data.get('schema_version')) is int and data['schema_version'] == 1
                and data.get('state_breakdown') is True)
    except (OSError, ValueError, UnicodeError):
        return False


def save_state_breakdown(enabled: bool, path: Path | None = None) -> None:
    destination = path or preferences_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent,
                                         prefix='.preferences-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({'schema_version': 1, 'state_breakdown': bool(enabled)}, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class StateBreakdownPreference:
    """Shared Tk binding for both desktop entrypoints and parameter drafts."""

    def __init__(self, root, *, path=None):
        from tkinter import BooleanVar, StringVar
        self.path = path
        self.variable = BooleanVar(root, value=load_state_breakdown(path))
        self.notice = StringVar(root, value='')
        self._controls = weakref.WeakSet()
        self._busy = False

    def commit(self, enabled=None):
        if enabled is not None:
            self.variable.set(bool(enabled))
        try:
            save_state_breakdown(self.variable.get(), self.path)
        except OSError:
            self.notice.set('This setting is active for this session, but could not be saved.')
        else:
            self.notice.set('')

    def snapshot(self):
        from pipeline_calculator.core.options import AnalysisOptions
        return AnalysisOptions(state_breakdown=bool(self.variable.get()))

    def set_busy(self, busy):
        self._busy = bool(busy)
        for control in tuple(self._controls):
            if control.winfo_exists():
                control.configure(state='disabled' if busy else 'normal')

    def add_control(self, parent, *, draft=None):
        import customtkinter as ctk
        from pipeline_calculator.gui.layout import WrappedLabel
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        frame.pack(fill='x', padx=12, pady=(8, 10))
        row = ctk.CTkFrame(frame, fg_color='transparent')
        row.pack(anchor='w', pady=(0, 4))
        variable = self.variable if draft is None else draft
        switch = ctk.CTkSwitch(row, text='State breakdown',
                              variable=variable,
                              command=self.commit if draft is None else None,
                              state='disabled' if self._busy else 'normal')
        switch.pack(side='left')

        class StatusLabel(ctk.CTkLabel):
            preference_trace = None

            def destroy(label):
                if label.preference_trace is not None:
                    variable.trace_remove('write', label.preference_trace)
                    label.preference_trace = None
                super().destroy()

        switch_font = switch.cget('font')
        font = ctk.CTkFont(family=switch_font.cget('family'),
                          size=switch_font.cget('size'), weight='bold')
        status = StatusLabel(row, text='', font=font, width=32,
                             height=switch.cget('height'), anchor='w')
        status.pack(side='left', padx=(8, 0))

        def update_status(*args):
            enabled = variable.get()
            status.configure(text='ON' if enabled else 'OFF',
                             text_color='#8CD8A8' if enabled else '#FF8080')

        status.preference_trace = variable.trace_add('write', update_status)
        update_status()
        # CTk's canvas switch otherwise lacks native keyboard activation.
        switch._canvas.configure(takefocus=True, highlightthickness=1,
                                  highlightcolor='#9CC8EB', highlightbackground='#202020')
        switch._canvas.bind('<space>', lambda event: self._activate(switch))
        switch._canvas.bind('<Return>', lambda event: self._activate(switch))
        self._controls.add(switch)
        WrappedLabel(frame, text='Split mileage and overlap results by U.S. state.',
                     text_color='#B8C0CC', anchor='w', justify='left').pack(fill='x')
        self.add_notice(frame)
        return switch

    def add_notice(self, parent, **pack_options):
        """Keep failed-save notices visible after leaving the parameter dialog."""
        from pipeline_calculator.gui.layout import WrappedLabel
        variable = self.notice
        class NoticeLabel(WrappedLabel):
            preference_trace = None

            def destroy(label):
                if label.preference_trace is not None:
                    variable.trace_remove('write', label.preference_trace)
                    label.preference_trace = None
                super().destroy()

        notice_label = NoticeLabel(parent, textvariable=variable, text_color='#E5C783',
                                  anchor='w', justify='left')
        def show_notice(*args):
            if self.notice.get():
                notice_label.pack(fill='x', **pack_options)
            else:
                notice_label.pack_forget()
        notice_label.preference_trace = variable.trace_add('write', show_notice)
        show_notice()
        return notice_label

    @staticmethod
    def _activate(switch):
        if switch.cget('state') != 'disabled':
            switch.toggle()
        return 'break'
