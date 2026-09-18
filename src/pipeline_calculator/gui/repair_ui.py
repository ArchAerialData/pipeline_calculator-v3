"""Shared repair decisions and snapshot ownership for both desktop entrypoints.

Only this module's worker reads/writes a repaired copy. All widget work stays on
the Tk thread, and an approval always resumes the captured source and request.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.gui.layout import ActionBar, WrappedLabel
from pipeline_calculator.gui.modal import ModalBody, ModalSurface, MUTED, TEXT

DETAIL_LIMIT = 24000


def _details(value):
    """Plain text only; cap rendering independently of the retained report."""
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    return text if len(text) <= DETAIL_LIMIT else text[:DETAIL_LIMIT] + '\n[Display shortened.]'


def _keyboard_button(button):
    # CTk buttons do not opt into native keyboard traversal themselves.
    tk.Misc.configure(button, takefocus=1)
    tk.Misc.bind(button, '<Return>', lambda event: (button.invoke(), 'break')[1], add='+')
    tk.Misc.bind(button, '<space>', lambda event: (button.invoke(), 'break')[1], add='+')
    tk.Misc.bind(button, '<FocusIn>', lambda event: button.configure(border_width=2,
                 border_color='#9DCBF0'), add='+')
    tk.Misc.bind(button, '<FocusOut>', lambda event: button.configure(border_width=0), add='+')


class RepairActionFooter(ctk.CTkFrame):
    """Keep repair approval below the reversible choices at every window size."""

    def __init__(self, parent, actions, primary_action):
        super().__init__(parent, fg_color='transparent')
        self.secondary = ActionBar(self, actions)
        self.secondary.pack(fill='x')
        for button in self.secondary.buttons:
            button.configure(fg_color='#394553', hover_color='#4A5A6C')
        label, command = primary_action
        self.primary_button = ctk.CTkButton(self, text=label, command=command,
            height=38, font=ctk.CTkFont(size=14, weight='bold'),
            fg_color='#237A45', hover_color='#195C33', text_color='white')
        self.primary_button.pack(fill='x', padx=5, pady=(8, 4))
        # Preserve the approval-first action interface used by both entrypoints;
        # keyboard traversal follows the actual visual order independently.
        self.buttons = [self.primary_button, *self.secondary.buttons]
        self.visual_buttons = [*self.secondary.buttons, self.primary_button]


class RepairPanel:
    """One scrollable body and a permanently reachable responsive action footer."""

    def __init__(self, root, *, title, message, filename='', details='', actions=(),
                 primary_action=None, on_cancel=None):
        self.root = root
        self.closed = False
        self.on_cancel = on_cancel
        self.bindings = []
        self.resize_id = None
        self.previous_focus = root.focus_get()
        self.surface = ModalSurface(root)
        self.card = self.surface.card
        self.footer = (RepairActionFooter(self.card, actions, primary_action)
                       if primary_action is not None else ActionBar(self.card, actions))
        self.footer.pack(side='bottom', fill='x', padx=14, pady=(8, 14))
        self.body = ModalBody(self.card)
        self.body.pack(fill='both', expand=True, padx=22, pady=(20, 4))
        WrappedLabel(self.body, text=title, text_color=TEXT, anchor='w', justify='left',
                     font=ctk.CTkFont(size=22, weight='bold')).pack(fill='x', pady=(0, 8))
        if filename:
            WrappedLabel(self.body, text=filename, text_color=MUTED, anchor='w',
                         justify='left').pack(fill='x', pady=(0, 10))
        self.message = WrappedLabel(self.body, text=message, text_color=TEXT,
                                    anchor='w', justify='left', font=ctk.CTkFont(size=14))
        self.message.pack(fill='x', pady=(0, 12))
        self.detail_text = _details(details) if details else ''
        self.detail_box = None
        self.detail_button = None
        self.focus_controls = list(self.footer.visual_buttons if primary_action is not None
                                   else self.footer.buttons)
        self.initial_focus = self.focus_controls[0] if self.focus_controls else None
        if self.detail_text:
            self.detail_button = ctk.CTkButton(self.body, text='Show details', width=150,
                                              fg_color='#394553', command=self.toggle_details)
            self.detail_button.pack(anchor='w', pady=(0, 8))
            self.focus_controls.append(self.detail_button)
        if primary_action is None:
            for index, button in enumerate(self.footer.buttons):
                if index:
                    button.configure(fg_color='#394553', hover_color='#4A5A6C')
        for button in self.focus_controls:
            _keyboard_button(button)
        self.bindings.append(('<Escape>', root.bind('<Escape>', self._escape, add='+')))
        self.bindings.append(('<Tab>', root.bind('<Tab>', self._tab, add='+')))
        self.bindings.append(('<Shift-Tab>', root.bind('<Shift-Tab>', lambda e: self._tab(e, -1), add='+')))
        self.bindings.append(('<Configure>', root.bind('<Configure>', self._queue_resize, add='+')))
        self.body.bind('<Configure>', self._queue_resize, add='+')
        self.surface.show((720, 340))
        self._queue_resize()
        self.surface.grab_set()
        if self.initial_focus is not None:
            tk.Misc.focus_set(self.initial_focus)

    def _escape(self, event=None):
        if self.on_cancel is not None:
            self.on_cancel()
        return 'break'

    def _queue_resize(self, event=None):
        if not self.closed and self.resize_id is None:
            self.resize_id = self.root.after(30, self._resize)

    def _resize(self):
        self.resize_id = None
        if self.closed:
            return
        scale = ctk.ScalingTracker.get_widget_scaling(self.surface)
        # Size short decisions to their content; only the body scrolls once
        # expanded details or the screen size exceed the available height.
        natural = (self.body.winfo_reqheight() + self.footer.winfo_reqheight()) / scale + 48
        preferred = (720, min(650, max(230, round(natural))))
        if preferred != self.surface._preferred_size:
            self.surface._preferred_size = preferred
            self.surface._fit_card()

    def _tab(self, event=None, direction=1):
        controls = [w for w in self.focus_controls if w.winfo_exists() and w.winfo_viewable()
                    and (not isinstance(w, ctk.CTkButton) or w.cget('state') != 'disabled')]
        if controls:
            focus = self.root.focus_get()
            index = next((i for i, w in enumerate(controls) if focus is w or
                          str(focus).startswith(str(w) + '.')), -1)
            tk.Misc.focus_set(controls[(index + direction) % len(controls)])
        return 'break'

    def toggle_details(self):
        if self.detail_box is None:
            self.detail_box = ctk.CTkTextbox(self.body, height=190, wrap='word')
            self.detail_box.insert('1.0', self.detail_text)
            self.detail_box.configure(state='disabled')
            self.focus_controls.append(self.detail_box._textbox)
        if self.detail_box.winfo_manager():
            self.detail_box.pack_forget()
            self.detail_button.configure(text='Show details')
        else:
            self.detail_box.pack(fill='x', pady=(0, 12))
            self.detail_button.configure(text='Hide details')

    def copy_text(self, text, *, client=True):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.message.configure(text='Copied. You can paste this request into your message to the client.'
                                   if client else 'Diagnostic report copied for application support.')
        except tk.TclError:
            self.message.configure(text='Could not copy. Select the text in Details and copy it manually.')
            if self.detail_box is None or not self.detail_box.winfo_manager():
                self.toggle_details()

    def suspend(self):
        self.surface.grab_release()
        self.surface.place_forget()

    def restore(self):
        self.surface.show(self.surface._preferred_size)
        self.surface.grab_set()
        if self.initial_focus is not None:
            tk.Misc.focus_set(self.initial_focus)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.resize_id is not None:
            self.root.after_cancel(self.resize_id)
            self.resize_id = None
        for event, binding in self.bindings:
            self.root.unbind(event, binding)
        self.bindings.clear()
        try:
            self.surface.grab_release()
            self.surface.destroy()
            if self.previous_focus is not None and self.previous_focus.winfo_exists() and self.previous_focus.winfo_viewable():
                self.previous_focus.focus_set()
        except tk.TclError:
            pass


class RepairWorkflow:
    """Own the active source across preparation, analysis, retry, and optional save."""

    def __init__(self, root, *, set_busy, on_return, on_replace, on_resume):
        self.root = root
        self.set_busy = set_busy
        self.on_return = on_return
        self.on_replace = on_replace
        self.on_resume = on_resume
        self.source = None
        self.display_name = ''
        self.panel = None
        self.closed = False
        self.analysis_state = None
        self.save_context = None
        self.save_poll = None

    @property
    def verified(self):
        return self.source is not None and bool(getattr(self.source, 'verified', False))

    def session_for(self, path):
        if self.verified and Path(path).resolve() == Path(self.source.original_path).resolve():
            return self.source
        self.retire_source()
        return None

    def retire_source(self):
        # Worker closures retain their own reference; dropping the UI owner must
        # never invalidate a snapshot that a cancelling worker still consumes.
        self.source = None
        self.analysis_state = None

    def handle_done(self, job):
        self.display_name = Path(getattr(job, 'file_path', '')).name
        source = getattr(job, 'source_session', None)
        if source is not None:
            self.source = source
        self.analysis_state = job.state
        if job.state == 'repair_required':
            self.offer(job)
            return True
        error = job.error
        if error is not None and hasattr(error, 'findings'):
            self.set_busy(False)
            self.on_return()
            self.set_busy(True)
            self.show_failure(error)
            return True
        return False

    def _close_panel(self):
        if self.panel is not None:
            self.panel.close()
            self.panel = None

    def cancel_decision(self):
        self._close_panel()
        if not self.verified:
            self.retire_source()
        self.set_busy(False)
        self.on_return()

    def choose_another(self):
        panel = self.panel
        panel.suspend()
        try:
            path = filedialog.askopenfilename(parent=self.root, title='Choose another KML or KMZ',
                         filetypes=[('KML / KMZ files', '*.kml *.kmz'), ('All files', '*.*')])
        except Exception as error:
            panel.message.configure(text=f'Could not open the file picker: {error}')
            panel.restore()
            return
        if not path:
            panel.restore()
            return
        self._close_panel()
        self.retire_source()
        self.set_busy(False)
        self.on_replace(path)

    def offer(self, job):
        self.set_busy(True)
        source = self.source

        def approve():
            if self.panel is None or self.closed:
                return
            self._close_panel()
            self.on_resume(job.file_path, job.params, options=job.options,
                           source_session=source, approve_repair=True)

        report = source.report
        format_only = report.get('rules') == ['filename_format_mismatch_v1']
        self.panel = RepairPanel(self.root, title=("This file's contents do not match its extension"
                                                 if format_only else 'This file may be safely repairable'),
            filename=source.display_name,
            message=('We’ll verify that its geometry is unchanged before analyzing it.' if format_only else
                     'This file has a formatting error that may be safely repairable. We’ll verify that its geometry is unchanged before analyzing it.'),
            details=report, actions=[('Cancel', self.cancel_decision),
                                    ('Choose another file', self.choose_another)],
            primary_action=('Repair & analyze', approve), on_cancel=self.cancel_decision)

    def show_failure(self, error):
        category = getattr(error, 'category', '')
        findings = getattr(error, 'findings', [])
        source_problem = category not in ('verification', 'operation', 'limit') and any(
            f.get('category', 'source') == 'source' or
            (f.get('category') == 'policy' and f.get('action')) for f in findings)
        client_request = getattr(error, 'client_request', '') if source_problem else ''
        if callable(client_request):
            client_request = client_request()
        title = 'This file cannot be repaired safely' if client_request else 'Could not prepare this file'
        if category == 'verification':
            title = 'We could not verify this repair'
        if category == 'coverage':
            title = 'Formatting fixed; analysis blocked'
        diagnostic = _details({'category': category, 'message': str(error), 'findings': findings})
        details = (client_request + '\n\nTechnical details\n' if client_request else '') + diagnostic
        actions = []
        if client_request:
            actions.append(('Copy client request', lambda: self.panel.copy_text(client_request)))
        else:
            actions.append(('Copy diagnostic report', lambda: self.panel.copy_text(diagnostic, client=False)))
        actions.extend([('Choose another file', self.choose_another), ('Cancel', self.cancel_decision)])
        self.panel = RepairPanel(self.root, title=title,
            filename=self.source.display_name if self.source is not None else self.display_name,
            message=str(error), details=details, actions=actions, on_cancel=self.cancel_decision)

    def add_notice(self, parent, **pack_options):
        if not self.verified:
            return None
        report = self.source.report
        if report.get('status') in ('not_needed', 'unchanged'):
            return None
        frame = ctk.CTkFrame(parent, fg_color='#23372F', border_color='#456A59', border_width=1)
        frame.pack(fill='x', **pack_options)
        text = 'File formatting repaired. Geometry verified unchanged. Original preserved.'
        rules = report.get('rules', [])
        if rules == ['filename_format_mismatch_v1']:
            text = 'File type recognized. Geometry verified unchanged. Original preserved.'
        if self.analysis_state in ('failed', 'cancelled'):
            text += ' Analysis could not finish.' if self.analysis_state == 'failed' else ' Analysis cancelled.'
        WrappedLabel(frame, text=text, text_color='#B7E7C6', anchor='w', justify='left',
                     font=ctk.CTkFont(size=13)).pack(fill='x', padx=12, pady=(8, 2))
        controls = ctk.CTkFrame(frame, fg_color='transparent')
        controls.pack(fill='x', padx=12, pady=(0, 8))
        buttons = [ctk.CTkButton(controls, text=label, command=command, width=width, height=28,
                    font=ctk.CTkFont(size=12), fg_color='#304D3E', hover_color='#3D634F')
                   for label, command, width in [('Details', self.show_details, 88),
                                                ('Save repaired copy…', self.save_copy, 168)]]
        def arrange(event=None):
            available = controls.winfo_width() / ctk.ScalingTracker.get_widget_scaling(controls)
            columns = 2 if available >= 272 else 1
            for index, button in enumerate(buttons):
                button.grid(row=index // columns, column=index % columns, sticky='w',
                            padx=(0, 8), pady=(2, 0))
        controls.bind('<Configure>', arrange, add='+')
        arrange()
        if not self.source.can_save:
            buttons[1].configure(state='disabled')
        for button in buttons:
            _keyboard_button(button)
        return frame

    def show_details(self):
        if self.panel is not None or not self.verified:
            return
        self.set_busy(True)

        def done():
            self._close_panel()
            self.set_busy(False)

        actions = [('Close', done)]
        if self.source.can_save:
            actions.insert(0, ('Save repaired copy…', lambda: (done(), self.save_copy())))
        message = 'The original was preserved. The approved formatting changes passed geometry verification.'
        if not self.source.can_save:
            message += '\n\nSaving a copy is unavailable: ' + str(self.source.save_unavailable_reason)
        self.panel = RepairPanel(self.root, title='Verified file repair', filename=self.source.display_name,
            message=message, details=self.source.report, actions=actions, on_cancel=done)

    def save_copy(self):
        if self.closed or self.panel is not None or not self.verified or not self.source.can_save:
            return
        source = self.source
        extension = '.' + source.effective_format.lower().lstrip('.')
        stem = Path(source.display_name).stem + '_repaired'
        original_parent = Path(source.original_path).parent
        proposed = original_parent / (stem + extension)
        index = 2
        while proposed.exists():
            proposed = original_parent / f'{stem}_{index}{extension}'
            index += 1
        self.set_busy(True)
        try:
            path = filedialog.asksaveasfilename(parent=self.root, title='Save verified repaired copy',
                initialfile=proposed.name, initialdir=str(original_parent), defaultextension=extension,
                filetypes=[(extension[1:].upper(), '*' + extension)])
        except Exception as error:
            self.set_busy(False)
            self._save_feedback(error=error)
            return
        if not path:
            self.set_busy(False)
            return
        context = self.save_context = ExecutionContext()
        done = threading.Event()
        outcome = {}

        def cancel():
            context.cancel()
            if self.panel is not None:
                self.panel.message.configure(text='Cancelling the save. Your analysis and verified input remain available.')
                for button in self.panel.footer.buttons:
                    button.configure(state='disabled')

        self.panel = RepairPanel(self.root, title='Saving repaired copy', filename=source.display_name,
            message='Writing and independently verifying the copy. Your original file will not be overwritten.',
            actions=[('Cancel', cancel)], on_cancel=cancel)

        def worker():
            try:
                outcome['receipt'] = source.save(path, context=context)
            except BaseException as error:
                outcome['error'] = error
            finally:
                done.set()

        def poll():
            self.save_poll = None
            if self.closed:
                return
            if not done.is_set():
                self.save_poll = self.root.after(80, poll)
                return
            self.save_context = None
            self._close_panel()
            self.set_busy(False)
            error = outcome.get('error')
            if isinstance(error, AnalysisCancelled):
                return
            self._save_feedback(path=path, error=error, receipt=outcome.get('receipt', ''))

        try:
            threading.Thread(target=worker, daemon=True).start()
        except RuntimeError as error:
            outcome['error'] = error
            done.set()
        poll()

    def _save_feedback(self, *, path=None, error=None, receipt=''):
        self.set_busy(True)

        def finish():
            self._close_panel()
            self.set_busy(False)

        actions = [('Close', finish)]
        if error is not None:
            actions.insert(0, ('Retry save', lambda: (finish(), self.save_copy())))
        self.panel = RepairPanel(self.root,
            title='The repaired copy could not be saved' if error else 'Repaired copy saved',
            message=('Your analysis and verified input remain available.\n\n' + str(error)) if error else str(path),
            details=str(error) if error else receipt, actions=actions, on_cancel=finish)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.save_context is not None:
            self.save_context.cancel()
        if self.save_poll is not None:
            self.root.after_cancel(self.save_poll)
            self.save_poll = None
        self._close_panel()
        self.retire_source()
