from __future__ import annotations

import os
from tkinter import StringVar, ttk

import customtkinter as ctk
from pipeline_calculator.gui.layout import ActionBar, ResultPages, WrappedLabel

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
) -> None:
    """Render the results screen."""

    for widget in root.winfo_children():
        widget.destroy()

    root.title(f"Pipeline Calculator v{version} - Results")
    ActionBar(root, [("Export Results", on_export), ("Adjust Parameters", on_reanalyze),
                     ("Import New File", on_new_file), ("Exit", on_exit)]).pack(
                         side="bottom", fill="x", padx=10, pady=6)

    header_frame = ctk.CTkFrame(root)
    header_frame.pack(fill="x", padx=10, pady=5)

    file_name = os.path.basename(current_file) if current_file else ""
    ctk.CTkLabel(header_frame, text="File:").pack(side="left", padx=(8, 4))
    file_label = ctk.CTkEntry(header_frame)
    file_label.insert(0, file_name)
    file_label.configure(state="readonly")
    file_label.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4)
    if state_preference is not None:
        state_preference.add_notice(root, padx=18, pady=(0, 4))

    geography = current_results.get('geography')
    states = sorted((geography or {}).get('states') or [], key=lambda row: row['state_name'])
    scopes = {'Combined': current_results, **{row['state_name']: row for row in states}}
    selection = StringVar(root, value='Combined')
    tabview = None

    def select_scope(name):
        nonlocal tabview
        if name not in scopes:
            return
        previous = tabview.selector.get() if tabview is not None else 'Summary'
        if tabview is not None:
            tabview.destroy()
        selection.set(name)
        # A presentation copy leaves the complete analysis snapshot available to export.
        displayed = dict(scopes[name])
        displayed['_geography'] = geography
        displayed.setdefault('analysis_parameters', current_results.get('analysis_parameters'))
        state_view = name != 'Combined'
        tabview = ResultPages(root)
        tabview.pack(fill='both', expand=True, padx=10, pady=5)
        summary_tab = tabview.add('Summary')
        create_summary_tab(summary_tab, displayed, on_select_state=select_scope)
        if displayed.get('pipelines'):
            create_pipelines_tab(tabview.add('Pipelines'), displayed)
        if displayed.get('overlap_analysis') or state_view:
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
        if previous in tabview.pages:
            tabview.set(previous)

    if geography is not None:
        scope_bar = ctk.CTkFrame(root, fg_color='transparent')
        scope_bar.pack(fill='x', padx=18, pady=(4, 0))
        ctk.CTkLabel(scope_bar, text='View:').pack(side='left', padx=(0, 8))
        selector = ttk.Combobox(scope_bar, textvariable=selection, values=list(scopes),
                                state='readonly', width=25, takefocus=True)
        style = ttk.Style(selector)
        style.theme_use('clam')
        style_name = f'StateScope{id(selector)}.TCombobox'
        style.configure(style_name, foreground='#F1F4F8', fieldbackground='#242424',
                        background='#343434', arrowcolor='#F1F4F8', bordercolor='#535B65')
        style.map(style_name, fieldbackground=[('readonly', '#242424')],
                  foreground=[('readonly', '#F1F4F8')], selectbackground=[('readonly', '#1F538D')])
        selector.configure(style=style_name)
        last_scale = [None]
        def scale_selector(event=None):
            factor = ctk.ScalingTracker.get_widget_scaling(scope_bar)
            if factor != last_scale[0]:
                last_scale[0] = factor
                font = ('Arial', -round(14 * factor))
                selector.configure(font=font)
                style.configure(style_name, padding=round(5 * factor), arrowsize=round(14 * factor))
                root.option_add('*TCombobox*Listbox.font', font)
        scope_bar.bind('<Configure>', scale_selector, add='+')
        scale_selector()
        selector.pack(side='left', padx=(0, 8), pady=4)
        selector.bind('<<ComboboxSelected>>', lambda event: select_scope(selection.get()))
    select_scope('Combined')

