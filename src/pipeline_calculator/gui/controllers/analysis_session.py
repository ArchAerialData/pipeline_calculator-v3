"""Shared Tk-thread presentation/cleanup for both application entrypoints."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
import customtkinter as ctk

from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController
from pipeline_calculator.core.workload import WorkloadWarning
from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.modal import ModalSurface, ModalBody, TEXT, MUTED, OUTLINE


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
        self.resize_binding = None
        self.resize_id = None
        self.panel_size = None
        self.surface = None
        self.presented_warning = None

    def start(self, path, params, *, options=None, source_session=None, approve_repair=False):
        if self.closed or self.job is not None:
            raise RuntimeError('Analysis session already started or closed')
        self.surface = ModalSurface(self.root)
        self.frame = self.surface.card
        self.frame.pack_propagate(False)
        self.surface.show()
        # Reserve the action footer first so it cannot scroll out of reach.
        self.footer = ctk.CTkFrame(self.frame, fg_color='transparent')
        self.footer.pack(side='bottom', fill='x', padx=24, pady=(0, 20))
        ctk.CTkFrame(self.footer, height=1, fg_color=OUTLINE, corner_radius=0).pack(fill='x', pady=(0, 12))
        self.footer_note = WrappedLabel(self.footer, text='You can cancel at any time.',
                                        text_color=MUTED, font=ctk.CTkFont(size=12), anchor='w', justify='left')
        self.footer_note.pack(fill='x', pady=(0, 12))
        self.progress_area = ctk.CTkFrame(self.footer, fg_color='transparent')
        self.progress_area.pack(fill='x', pady=(0, 16))
        progress_labels = ctk.CTkFrame(self.progress_area, fg_color='transparent')
        progress_labels.pack(fill='x', pady=(0, 6))
        ctk.CTkLabel(progress_labels, text='Overall progress', anchor='w', text_color=MUTED,
                     font=ctk.CTkFont(size=13), height=24).pack(side='left')
        self.percent_label = ctk.CTkLabel(progress_labels, text='0%', anchor='e',
            text_color='#A6E5BB', font=ctk.CTkFont(size=15, weight='bold'), height=24)
        self.percent_label.pack(side='right')
        self.bar = ctk.CTkProgressBar(self.progress_area, height=12, mode='determinate',
                                    fg_color='#3A444F', progress_color='#40B873')
        self.bar.pack(fill='x')
        self.bar.set(0)
        self.controls = ctk.CTkFrame(self.footer, fg_color='transparent')
        self.controls.pack(fill='x')
        self.cancel_button = ctk.CTkButton(self.controls, text='Cancel', command=self.cancel,
            width=112, height=38, fg_color='#E5B94F', hover_color='#CFA13A',
            text_color='#211B0E', font=ctk.CTkFont(size=14, weight='bold'))
        self.cancel_button.pack(side='left')
        self.continue_button = ctk.CTkButton(self.controls, text='Continue anyway', command=self.continue_workload,
            width=160, height=38, fg_color='#237A45', hover_color='#195C33', text_color='white',
            font=ctk.CTkFont(size=14, weight='bold'))
        self.content = ModalBody(self.frame)
        self.content.pack(fill='both', expand=True, padx=24, pady=(24, 18))
        self.badge = ctk.CTkLabel(self.content, text='ANALYZING', font=ctk.CTkFont(size=11, weight='bold'),
                                 text_color='#9CC8EB', fg_color='#253C50', corner_radius=6, height=26, width=118)
        self.badge.pack(anchor='w', pady=(0, 12))
        self.title_font = ctk.CTkFont(size=26, weight='bold')
        self.title = WrappedLabel(self.content, text='Analyzing your pipelines', anchor='w', justify='left',
                                  text_color=TEXT, font=self.title_font)
        self.title.pack(fill='x', pady=(0, 4))
        self.filename_label = WrappedLabel(self.content, text=Path(path).name, text_color=MUTED,
                                           font=ctk.CTkFont(size=13), anchor='w', justify='left')
        self.filename_label.pack(fill='x', pady=(0, 20))
        self.body_font = ctk.CTkFont(size=15)
        self.label = WrappedLabel(self.content, text='Starting analysis...', font=self.body_font,
                                  text_color=MUTED, anchor='w', justify='left')
        self.label.pack(fill='x', pady=(0, 18))
        self.warning_sections = ctk.CTkFrame(self.content, fg_color='transparent')
        estimate = ctk.CTkFrame(self.warning_sections, fg_color='#353126', border_width=1,
                                border_color='#665535', corner_radius=10)
        estimate.pack(fill='x', pady=(0, 22))
        ctk.CTkLabel(estimate, text='ESTIMATED RUN TIME', text_color='#DABB7A',
                     font=ctk.CTkFont(size=11, weight='bold'), anchor='w').pack(fill='x', padx=18, pady=(12, 2))
        self.estimate_label = WrappedLabel(estimate, text='', text_color='#F2D79C',
            font=ctk.CTkFont(size=19, weight='bold'), anchor='w', justify='left')
        self.estimate_label.pack(fill='x', padx=18, pady=(0, 14))
        self.guidance = ctk.CTkFrame(self.warning_sections, fg_color='transparent')
        self.guidance.pack(fill='x')
        self.guidance_cards = []
        for heading, detail in [('What to expect', WorkloadWarning.impact),
                                ('For a smaller run', WorkloadWarning.recommendation)]:
            section = ctk.CTkFrame(self.guidance, fg_color='transparent')
            self.guidance_cards.append(section)
            WrappedLabel(section, text=heading, text_color=TEXT, anchor='w', justify='left',
                         font=ctk.CTkFont(size=15, weight='bold')).pack(fill='x', pady=(0, 6))
            WrappedLabel(section, text=detail, text_color=MUTED, font=self.body_font,
                         anchor='w', justify='left').pack(fill='x')
        self.guidance_columns = None
        self.content.bind('<Configure>', self._queue_resize, add='+')
        self.resize_binding = self.root.bind('<Configure>', self._queue_resize, add='+')
        self._resize_panel()
        try:
            kwargs = {'options': options} if options is not None else {}
            if source_session is not None:
                kwargs['source_session'] = source_session
            if approve_repair:
                kwargs['approve_repair'] = True
            self.job = self.controller.start(path, params, **kwargs)
            self._poll(self.job.job_id)
        except BaseException:
            self.close()
            raise

    def _show_warning(self, warning):
        if self.presented_warning == warning:
            return
        self.presented_warning = warning
        self.warning_visible = True
        self.badge.configure(text='LONGER ANALYSIS', text_color='#E5C783', fg_color='#453B26')
        self.title.configure(text='Review before continuing')
        self.label.configure(text=WorkloadWarning.introduction)
        self.estimate_label.configure(text=warning.detail if isinstance(warning, WorkloadWarning) else str(warning))
        self.warning_sections.pack(fill='x')
        self.footer_note.configure(text='Safety limits remain active during analysis.')
        self.progress_area.pack_forget()
        self.continue_button.pack(side='right')

    def _show_progress(self, text):
        if self.warning_visible:
            self.warning_visible = False
            self.presented_warning = None
            self.warning_sections.pack_forget()
            self.continue_button.pack_forget()
            self.badge.configure(text='ANALYZING', text_color='#9CC8EB', fg_color='#253C50')
            self.title.configure(text='Analyzing your pipelines')
            self.footer_note.configure(text='You can cancel at any time.')
            self.progress_area.pack(fill='x', pady=(0, 16), before=self.controls)
        self.label.configure(text=text)

    def _queue_resize(self, event=None):
        if not self.closed and self.resize_id is None:
            self.resize_id = self.root.after(30, self._resize_panel)

    def _resize_panel(self):
        self.resize_id = None
        if self.closed or self.frame is None:
            return
        scale = ctk.ScalingTracker.get_widget_scaling(self.frame)
        width = min(760, self.root.winfo_width() / scale * .94)
        font_size = 15 if width >= 600 else 14
        if self.body_font.cget('size') != font_size:
            self.body_font.configure(size=font_size)
            self.title_font.configure(size=26 if width >= 600 else 22)
        columns = 2 if width >= 660 else 1
        if columns != self.guidance_columns:
            self.guidance_columns = columns
            for index in range(2):
                # An empty column in a uniform group still reserves a share of
                # the width. At narrow/high-DPI sizes that feeds label wrapping
                # back into grid's natural widths and keep Tk recomputing layout.
                self.guidance.grid_columnconfigure(index, weight=int(index < columns),
                                                   uniform='guidance' if index < columns else '')
            for index, section in enumerate(self.guidance_cards):
                section.grid(row=index // columns, column=index % columns, sticky='new',
                             padx=(0, 24) if columns == 2 and index == 0 else 0,
                             pady=(0, 16) if columns == 1 and index == 0 else 0)
        # Measure the content with all its section spacing; keep actions fixed
        # while only the body scrolls on short or high-DPI windows.
        natural_height = (self.content.winfo_reqheight() + self.footer.winfo_reqheight()) / scale + 64
        height = min(max(260, natural_height), self.root.winfo_height() / scale * .90)
        size = (round(width), round(height))
        if size != self.panel_size:
            self.panel_size = size
            self.frame.configure(width=size[0], height=size[1])

    def cancel(self):
        if not self.closed and self.job is not None:
            self.job.cancel()
            self.cancel_button.configure(state='disabled')
            self._show_progress('Cancelling... Waiting for the current operation to stop.')

    def continue_workload(self):
        if not self.closed and self.job is not None:
            self.job.context.accept_workload()
            self._show_progress('Continuing analysis...')

    def _poll(self, job_id):
        self.poll_id = None
        if self.closed or self.job is None or self.job.job_id != job_id:
            return
        job = self.job
        if job.done.is_set():
            if job.state == 'completed':
                self.bar.set(1)
                self.percent_label.configure(text='100%')
            self._cleanup()
            self.closed = True
            self.on_done(job)
            return
        warning = job.context.workload_warning()
        if job.state == 'cancellation_requested':
            self._show_progress('Cancelling... Waiting for the current operation to stop.')
        elif warning is not None:
            self._show_warning(warning)
        else:
            snapshot = job.context.snapshot()
            if snapshot is not None and snapshot.job_id == job_id and snapshot.sequence >= self.last_sequence:
                self.last_sequence = snapshot.sequence
                count = f'{snapshot.completed:,}'
                if snapshot.total is not None:
                    count += f' of {snapshot.total:,}'
                elapsed = job.context.elapsed_seconds()
                self.bar.set(snapshot.fraction)
                self.percent_label.configure(text=f'{int(snapshot.fraction*100)}%')
                detail = f'{snapshot.stage}: {count}' if snapshot.total is not None or snapshot.completed else snapshot.stage
                self._show_progress(f'{detail}\nElapsed: {elapsed:.0f} s')
        self._queue_resize()
        self.poll_id = self.root.after(100, lambda: self._poll(job_id))

    def _cleanup(self):
        if self.resize_binding is not None:
            self.root.unbind('<Configure>', self.resize_binding)
            self.resize_binding = None
        if self.resize_id is not None:
            self.root.after_cancel(self.resize_id)
            self.resize_id = None
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
                self.surface.destroy()
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
