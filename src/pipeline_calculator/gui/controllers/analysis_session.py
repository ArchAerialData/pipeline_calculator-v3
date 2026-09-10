"""Shared Tk-thread presentation/cleanup for both application entrypoints."""
from __future__ import annotations

import time
from pathlib import Path
import tkinter as tk
import customtkinter as ctk

from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController


class AnalysisSession:
    def __init__(self, root, on_done, controller=None):
        self.root = root
        self.on_done = on_done
        self.controller = controller or AnalysisController()
        self.job = None
        self.frame = None
        self.poll_id = None
        self.closed = False
        self.last_sequence = 0
        self.warning_visible = False

    def start(self, path, params):
        if self.closed or self.job is not None:
            raise RuntimeError('Analysis session already started or closed')
        self.frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.9, relheight=0.85)
        self.frame.lift()
        # Reserve controls first; long filenames/warnings scroll above them.
        footer = ctk.CTkFrame(self.frame, fg_color='transparent')
        footer.pack(side='bottom', fill='x', padx=10, pady=10)
        self.bar = ctk.CTkProgressBar(footer, width=200, mode='indeterminate')
        self.bar.pack(padx=20, pady=10)
        self.bar.start()
        self.controls = ctk.CTkFrame(footer, fg_color='transparent')
        self.controls.pack()
        self.cancel_button = ctk.CTkButton(self.controls, text='Cancel', command=self.cancel)
        self.cancel_button.pack(side='left', padx=5)
        self.continue_button = ctk.CTkButton(self.controls, text='Continue anyway', command=self.continue_workload)
        content = ctk.CTkScrollableFrame(self.frame, height=160)
        content.pack(fill='both', expand=True, padx=10, pady=10)
        self.label = ctk.CTkLabel(content, text='Starting analysis...', wraplength=380)
        self.label.pack(padx=10, pady=10)
        self.filename_label = ctk.CTkLabel(content, text=Path(path).name, wraplength=380,
                                         text_color='#B8C0CC')
        self.filename_label.pack(padx=10, pady=(0, 10))
        content.bind('<Configure>', self._resize_text, add='+')
        try:
            self.job = self.controller.start(path, params)
            self._poll(self.job.job_id)
        except BaseException:
            self.close()
            raise

    def _resize_text(self, event):
        if not self.closed:
            scale = ctk.ScalingTracker.get_widget_scaling(self.label)
            width = max(60, event.width / scale - 32)
            self.label.configure(wraplength=width)
            self.filename_label.configure(wraplength=width)

    def cancel(self):
        if not self.closed and self.job is not None:
            self.job.cancel()
            self.cancel_button.configure(state='disabled')
            self.label.configure(text='Cancelling... Waiting for the current operation to stop.')
            self.continue_button.pack_forget()

    def continue_workload(self):
        if not self.closed and self.job is not None:
            self.job.context.accept_workload()
            self.continue_button.pack_forget()
            self.warning_visible = False
            self.bar.start()
            self.label.configure(text='Continuing analysis...')

    def _poll(self, job_id):
        self.poll_id = None
        if self.closed or self.job is None or self.job.job_id != job_id:
            return
        job = self.job
        if job.done.is_set():
            self._cleanup()
            self.closed = True
            self.on_done(job)
            return
        warning = job.context.workload_warning()
        if job.state == 'cancellation_requested':
            self.label.configure(text='Cancelling... Waiting for the current operation to stop.')
        elif warning is not None:
            self.label.configure(text=warning)
            if not self.warning_visible:
                self.warning_visible = True
                self.bar.stop()
                self.continue_button.pack(side='left', padx=5)
        else:
            snapshot = job.context.snapshot()
            if snapshot is not None and snapshot.job_id == job_id and snapshot.sequence >= self.last_sequence:
                self.last_sequence = snapshot.sequence
                count = f'{snapshot.completed:,}'
                if snapshot.total is not None:
                    count += f' of {snapshot.total:,}'
                elapsed = max(0, time.monotonic() - job.context.started_at)
                self.label.configure(text=f'{snapshot.stage}: {count}\nElapsed: {elapsed:.0f} s')
        self.poll_id = self.root.after(100, lambda: self._poll(job_id))

    def _cleanup(self):
        if self.poll_id is not None:
            try:
                self.root.after_cancel(self.poll_id)
            except tk.TclError:
                pass
            self.poll_id = None
        if self.frame is not None:
            try:
                if getattr(self, 'bar', None) is not None:
                    self.bar.stop()
                self.frame.destroy()
            except tk.TclError:
                pass
            self.frame = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.job is not None:
            self.job.cancel()
        self._cleanup()
