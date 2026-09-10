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
        self.frame.place(relx=0.5, rely=0.5, anchor='center')
        self.frame.lift()
        ctk.CTkLabel(self.frame, text=Path(path).name, wraplength=380).pack(padx=20, pady=(10, 0))
        self.label = ctk.CTkLabel(self.frame, text='Starting analysis...', wraplength=380)
        self.label.pack(padx=20, pady=20)
        self.bar = ctk.CTkProgressBar(self.frame, width=300, mode='indeterminate')
        self.bar.pack(padx=20, pady=10)
        self.bar.start()
        self.cancel_button = ctk.CTkButton(self.frame, text='Cancel', command=self.cancel)
        self.cancel_button.pack(pady=(0, 20))
        self.continue_button = ctk.CTkButton(self.frame, text='Continue anyway', command=self.continue_workload)
        try:
            self.job = self.controller.start(path, params)
            self._poll(self.job.job_id)
        except BaseException:
            self.close()
            raise

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
        if job.state == 'cancellation_requested':
            self.label.configure(text='Cancelling... Waiting for the current operation to stop.')
        elif job.context.workload_warning() is not None:
            self.label.configure(text=job.context.workload_warning())
            if not self.warning_visible:
                self.warning_visible = True
                self.bar.stop()
                self.continue_button.pack(pady=(0, 20))
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
