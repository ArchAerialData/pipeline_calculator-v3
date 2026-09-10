"""Recoverable asynchronous KML launch, using Tk only on the calling thread."""
from __future__ import annotations

from pathlib import Path
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from pipeline_calculator.gui.actions.open_kml_action import create_and_launch_corridor, launch_saved_corridor


class CorridorDialog:
    def __init__(self, root, section, index):
        self.section, self.index = section, index
        self.path = None
        self.closed = False
        self.poll_id = None
        self.window = ctk.CTkToplevel(root)
        self.window.title('View Corridor')
        self.window.transient(root)
        self.label = ctk.CTkLabel(self.window, text='Preparing corridor...', wraplength=480)
        self.label.pack(padx=20, pady=20)
        self.path_entry = ctk.CTkEntry(self.window, width=480)
        self.path_entry.pack(padx=20, pady=5)
        self.buttons = []
        for text, command in [('Copy Path', self.copy_path), ('Save As', self.save_as), ('Retry', self.retry)]:
            button = ctk.CTkButton(self.window, text=text, command=command, state='disabled')
            button.pack(padx=20, pady=5)
            self.buttons.append(button)
        ctk.CTkButton(self.window, text='Close', command=self.close).pack(pady=15)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.bind('<Destroy>', self._destroyed, add='+')
        self.retry()

    def _destroyed(self, event):
        if event.widget is self.window:
            self._detach()

    def retry(self):
        if self.closed or (hasattr(self, 'done') and not self.done.is_set()):
            return
        self.done = threading.Event()
        self.outcome = self.error = None
        for button in self.buttons:
            button.configure(state='disabled')
        self.label.configure(text='Requesting opening...')
        # Capture inputs; the worker neither calls Tk nor schedules callbacks.
        path, section, index = self.path, self.section, self.index
        def worker():
            try:
                self.outcome = launch_saved_corridor(path) if path else create_and_launch_corridor(section, index)
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.done.set()
        try:
            threading.Thread(target=worker, daemon=True).start()
        except RuntimeError as exc:
            self.error = str(exc)
            self.done.set()
        self._poll()

    def _poll(self):
        self.poll_id = None
        if self.closed:
            return
        if not self.done.is_set():
            self.poll_id = self.window.after(100, self._poll)
            return
        if self.error is not None:
            self.label.configure(text=f'Could not prepare or open the corridor:\n{self.error}')
            self.buttons[-1].configure(state='normal')
            return
        self.path = self.outcome.path
        self.path_entry.delete(0, 'end')
        self.path_entry.insert(0, self.path)
        for button in self.buttons:
            button.configure(state='normal')
        text = 'Opening requested.' if self.outcome.status == 'requested' else f'KML was saved, but could not be opened.\n{self.outcome.error}'
        self.label.configure(text=text + '\nThis is a temporary file. Use Save As to keep a copy; the operating system may clean temporary files.')

    def copy_path(self):
        if self.path:
            self.window.clipboard_clear()
            self.window.clipboard_append(self.path)

    def save_as(self):
        if not self.path:
            return
        destination = filedialog.asksaveasfilename(parent=self.window, defaultextension='.kml',
            initialfile=Path(self.path).name, filetypes=[('KML', '*.kml')])
        if destination:
            try:
                if Path(destination).resolve() != Path(self.path).resolve():
                    shutil.copyfile(self.path, destination)
            except OSError as exc:
                messagebox.showerror('Save Error', f'{exc}\nOriginal KML remains at {self.path}', parent=self.window)
            else:
                self.label.configure(text=f'Copy saved to {destination}')

    def _detach(self):
        self.closed = True
        if self.poll_id is not None:
            try:
                self.window.after_cancel(self.poll_id)
            except tk.TclError:
                pass
            self.poll_id = None

    def close(self):
        self._detach()
        self.window.destroy()
