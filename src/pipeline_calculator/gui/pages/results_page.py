from __future__ import annotations

import os
import logging
from tkinter import StringVar

import customtkinter as ctk
from pipeline_calculator.gui.layout import ActionBar, ResultPages, WrappedLabel
from pipeline_calculator.gui.results_header import ResultsContextHeader

from pipeline_calculator.gui.tabs.overlap_tab import create as create_overlap_tab
from pipeline_calculator.gui.tabs.pipelines_tab import create as create_pipelines_tab
from pipeline_calculator.gui.tabs.placemarks_tab import create as create_placemarks_tab
from pipeline_calculator.gui.tabs.diagnostics_tab import create as create_diagnostics_tab
from pipeline_calculator.gui.tabs.summary_tab import create as create_summary_tab


def show(
    root,
    *,
    version: str,
    current_file: str | None,
    current_results: dict,
    on_export,
    on_reanalyze,
    on_new_file,
    on_exit,
    on_open_corridor,
    state_preference=None,
    repair_workflow=None,
) -> None:
    """Render the results screen."""

    for widget in root.winfo_children():
        widget.destroy()

    root.title(f"Pipeline Calculator v{version} - Results")
    ActionBar(root, [("Export Results", on_export), ("Adjust Parameters", on_reanalyze),
                     ("Import New File", on_new_file), ("Exit", on_exit)],
              compact_labels=('Export', 'Adjust', 'Import', 'Exit')).pack(
                         side="bottom", fill="x", padx=10, pady=6)

    header_frame = ctk.CTkFrame(root)
    header_frame.pack(fill="x", padx=10, pady=5)

    file_name = os.path.basename(current_file) if current_file else ""
    ctk.CTkLabel(header_frame, text="File:").pack(side="left", padx=(8, 4))
    file_label = ctk.CTkEntry(header_frame)
    file_label.insert(0, file_name)
    file_label.configure(state="readonly")
    file_label.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4)
    geography = current_results.get('geography')
    states = sorted((geography or {}).get('states') or [], key=lambda row: row['state_name'])
    scopes = {'Combined': current_results, **{row['state_name']: row for row in states}}
    selection = StringVar(root, value='Combined')
    tabview = None
    current_scope = None
    selector = None
    failed_scope = [None]
    error_frame = ctk.CTkFrame(root)
    error_label = WrappedLabel(error_frame, text='', text_color='#FFB993', justify='left')
    error_label.pack(fill='x', padx=12, pady=8)
    ctk.CTkButton(error_frame, text='Retry display', command=lambda: select_scope(failed_scope[0])).pack(pady=(0, 8))

    def populate_scope(tabview, name):
        # A presentation copy leaves the complete analysis snapshot available to export.
        displayed = dict(scopes[name])
        displayed['_geography'] = geography
        displayed.setdefault('analysis_parameters', current_results.get('analysis_parameters'))
        displayed.setdefault('application_version', current_results.get('application_version'))
        state_view = name != 'Combined'
        summary_tab = tabview.add('Summary')
        create_summary_tab(summary_tab, displayed, on_select_state=select_scope)
        if displayed.get('pipelines'):
            create_pipelines_tab(tabview.add('Pipelines'), displayed)
        overlap_failed = any(item.get('code') == 'overlap_analysis_failed'
                             for item in (displayed.get('diagnostics') or []))
        if displayed.get('overlap_analysis') or state_view or overlap_failed:
            create_overlap_tab(tabview.add('Overlap Analysis'), displayed, on_open_corridor=on_open_corridor)
        if current_results.get('placemarks'):
            placemark_tab = tabview.add('Placemarks')
            if state_view:
                WrappedLabel(placemark_tab, text='Point placemarks are available in the Combined view only.',
                             font=('Arial', 14)).pack(fill='x', padx=20, pady=20)
            else:
                create_placemarks_tab(placemark_tab, displayed)
        diagnostics = list(displayed.get('diagnostics') or [])
        if not state_view:
            diagnostics += (geography or {}).get('diagnostics') or []
        if diagnostics:
            create_diagnostics_tab(tabview.add('Diagnostics'), {'diagnostics': diagnostics})
    def select_scope(name):
        nonlocal tabview, current_scope
        if name not in scopes:
            return
        if name == current_scope:
            selection.set(current_scope)
            error_frame.pack_forget()
            return
        previous = tabview.selector.get() if tabview is not None else 'Summary'
        replacement = ResultPages(root)
        try:
            populate_scope(replacement, name)
            if previous in replacement.pages:
                replacement.set(previous)
        except Exception:
            replacement.destroy()
            logging.getLogger(__name__).exception('Could not render results scope %s', name)
            failed_scope[0] = name
            selection.set(current_scope or 'Combined')
            error_label.configure(text=f'Could not display {name}. ' +
                                  ('Your previous results are still available.' if tabview is not None else
                                   'Your analysis is still available to export.'))
            error_frame.pack(fill='x', padx=18, pady=4, **({'before': tabview} if tabview else {}))
            return
        error_frame.pack_forget()
        if tabview is not None:
            tabview.pack_forget()
        # Do not retain pack(before=old_view): CTk replays pack options on DPI
        # changes, after old_view has been destroyed.
        replacement.pack(fill='both', expand=True, padx=10, pady=5)
        if tabview is not None:
            tabview.destroy()
            if selector is not None:
                selector.focus_set()
        tabview = replacement
        current_scope = name
        selection.set(name)

    context_header = ResultsContextHeader(root, after=header_frame,
        scopes=scopes if geography is not None else None, selection=selection, on_select=select_scope,
        repair_workflow=repair_workflow, state_preference=state_preference)
    selector = context_header.selector
    select_scope('Combined')

