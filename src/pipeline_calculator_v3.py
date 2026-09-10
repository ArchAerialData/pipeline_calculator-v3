#!/usr/bin/env python3
"""
Pipeline Calculator with Overlap Analysis - KMZ/KML Pipeline Calculator
Compatibility entrypoint for Pipeline Calculator v4
Version: derived from Git or embedded build metadata
"""

import subprocess
import sys
import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import KDTree
from collections import defaultdict
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk, StringVar, DoubleVar
from tkinterdnd2 import TkinterDnD, DND_FILES
import platform
import traceback
import threading
from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
import json
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')
import tempfile
from PIL import Image, ImageTk
import math
from xml.sax.saxutils import escape as _xml_escape

# Version info
try:
    from pipeline_calculator import __version__ as __version__
except Exception:
    __version__ = "4.0-dev.unknown"
__author__ = "Pipeline Calculator Team"

# Default analysis parameters
DEFAULT_DETECTION_RANGE = 15  # meters
MIN_PARALLEL_LENGTH = 200  # meters
SEGMENT_LENGTH = 5  # meters
ANGULAR_TOLERANCE = 15  # degrees
GAP_TOLERANCE = 5  # meters

class PipelineAnalyzer:
    """Combined pipeline length and overlap analyzer."""
    
    def __init__(self):
        self.geod = Geod(ellps='GRS80')  # US standard
        self.survey_mile = 1609.347218694
        self.detection_range = DEFAULT_DETECTION_RANGE
        self.min_parallel_length = MIN_PARALLEL_LENGTH
        self.segment_length = SEGMENT_LENGTH
        self.angular_tolerance = ANGULAR_TOLERANCE
        
    def extract_features_from_file(self, file_path, progress_callback=None):
        """Extract all features from KMZ/KML file in a memory-efficient way."""
        from pipeline_calculator.parsers.kml_kmz import (
            extract_features_from_file as _extract_features_from_file,
        )

        return _extract_features_from_file(file_path, progress_callback=progress_callback)
    
    
    def calculate_pipeline_lengths(self, pipelines):
        """Calculate individual pipeline lengths."""
        from pipeline_calculator.core.analyzer import PipelineAnalyzer as _CoreAnalyzer

        core = _CoreAnalyzer(
            geod=self.geod,
            survey_mile=self.survey_mile,
            detection_range=self.detection_range,
            min_parallel_length=self.min_parallel_length,
            segment_length=self.segment_length,
            angular_tolerance=self.angular_tolerance,
        )
        return core.calculate_pipeline_lengths(pipelines)
    
    def segment_pipeline(self, coordinates):
        """Break pipeline into fixed-length segments for analysis."""
        from pipeline_calculator.core.segmentation import segment_pipeline as _segment_pipeline

        return _segment_pipeline(self.geod, coordinates, self.segment_length)
    
    def find_parallel_segments(self, pipelines, progress_callback=None):
        """Identify pipeline segments that run parallel within detection range."""
        from pipeline_calculator.core.overlap import find_parallel_segments as _find_parallel_segments

        return _find_parallel_segments(
            pipelines,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
        )
    
    def calculate_overlap_results(self, pipelines, parallel_groups, progress_callback=None):
        """Calculate bundled lengths and overlap statistics."""
        from pipeline_calculator.core.overlap import calculate_overlap_results as _calculate_overlap_results

        return _calculate_overlap_results(
            pipelines,
            parallel_groups,
            geod=self.geod,
            survey_mile_m=self.survey_mile,
            segment_length=self.segment_length,
            min_parallel_length=self.min_parallel_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
        )

    def compute_effective_length_by_clusters(self, pipelines, per_pipeline_total_meters, progress_callback=None):
        """Compute effective length from qualified, mutually compatible groups."""
        from pipeline_calculator.core.effective_length import (
            compute_effective_length_by_clusters as _compute_effective_length_by_clusters,
        )

        return _compute_effective_length_by_clusters(
            pipelines,
            per_pipeline_total_meters,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
            min_parallel_length=self.min_parallel_length,
        )
    
    def analyze_complete(self, file_path, progress_callback=None, *, context=None):
        """Complete analysis of KMZ/KML file."""
        from pipeline_calculator.core.analyzer import PipelineAnalyzer as _CoreAnalyzer

        core = _CoreAnalyzer(
            geod=self.geod,
            survey_mile=self.survey_mile,
            detection_range=self.detection_range,
            min_parallel_length=self.min_parallel_length,
            segment_length=self.segment_length,
            angular_tolerance=self.angular_tolerance,
        )
        return core.analyze_complete(file_path, progress_callback=progress_callback, context=context)


def build_analysis_workbook(current_results):
    """Build an XLSX workbook (openpyxl) from an analysis results dict.

    This is separated from the GUI flow so it can be unit-tested and used in CI.
    """
    # Delegated to the refactor package module; kept as a wrapper for backwards
    # compatibility with existing callers and packaging entrypoints.
    from pipeline_calculator.export.xlsx import build_analysis_workbook as _build_analysis_workbook

    return _build_analysis_workbook(current_results)


def build_overlap_corridor_kml(section, index):
    """Build a KML document for a bundled corridor section.

    Separated from GUI side-effects (tempfile + open) so it can be unit-tested.
    """
    from pipeline_calculator.export.corridor_kml import (
        build_overlap_corridor_kml as _build_overlap_corridor_kml,
    )

    return _build_overlap_corridor_kml(section, index)


class PipelineCalculatorGUI:
    """Main GUI application for pipeline calculator with overlap analysis."""
    
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self._set_app_icon()
        self.analyzer = PipelineAnalyzer()
        self.current_results = None
        self.current_file = None
        
        # Analysis parameter variables
        # NOTE: CTkEntry's internal textvariable trace calls `.get()` while the user is typing.
        # If a DoubleVar is used, tkinter attempts to convert transient values ("" / "q" / etc.)
        # to float and raises TclError. Use StringVar and parse when needed.
        self.detection_range_var = StringVar(value=str(DEFAULT_DETECTION_RANGE))
        self.min_parallel_var = StringVar(value=str(MIN_PARALLEL_LENGTH))
        self.segment_length_var = StringVar(value=str(SEGMENT_LENGTH))
        self.angular_tolerance_var = StringVar(value=str(ANGULAR_TOLERANCE))
        
        self._processing = False
        self._closing = False
        self._analysis_session = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.setup_gui()

    def _get_float_var(self, var, default):
        """Safely parse a float from a Tk variable (StringVar/DoubleVar) with fallback to default."""
        try:
            raw = var.get()
        except Exception:
            raw = ""

        # Allow temporary empty strings while the user is editing.
        if raw is None:
            raw_str = ""
        else:
            raw_str = str(raw).strip()

        if raw_str == "":
            try:
                var.set(str(default))
            except Exception:
                pass
            return float(default)

        # Accept common formatting like "1,234.5"
        raw_str = raw_str.replace(",", "")
        try:
            value = float(raw_str)
        except Exception:
            try:
                var.set(str(default))
            except Exception:
                pass
            return float(default)

        return value

    def _set_app_icon(self):
        """Configure window icon for supported platforms."""
        try:
            # In PyInstaller builds, resources are unpacked under `sys._MEIPASS`.
            if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                base_dir = getattr(sys, "_MEIPASS")
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            system = platform.system()
            if system == "Windows":
                icon_path = os.path.join(base_dir, "icon.ico")
                if os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
            else:
                icon_name = "icon.icns" if system == "Darwin" else "icon.ico"
                icon_path = os.path.join(base_dir, icon_name)
                if os.path.exists(icon_path):
                    img = Image.open(icon_path)
                    self._icon_image = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, self._icon_image)
        except Exception:
            pass
    
    def setup_gui(self):
        """Initialize the main GUI."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root.title(f"Pipeline Calculator v{__version__}")
        self.root.geometry("800x600")
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.show_file_selection()
    
    def show_file_selection(self):
        if getattr(self, "_processing", False):
            return
        """Display file selection interface."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, 
                                  text="Pipeline Calculator with Overlap Analysis", 
                                  font=("Arial", 24, "bold"))
        title_label.pack(pady=20)
        
        # Instructions
        instructions = ctk.CTkLabel(main_frame, 
                                   text="Drag and drop a KMZ or KML file here\n\nOR\n\nClick Browse to select a file",
                                   font=("Arial", 14),
                                   justify="center")
        instructions.pack(pady=20)
        
        # Parameters frame
        params_frame = ctk.CTkFrame(main_frame)
        params_frame.pack(pady=20, padx=40, fill="x")
        
        ctk.CTkLabel(params_frame, text="Analysis Parameters", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        # Detection range
        detection_frame = ctk.CTkFrame(params_frame)
        detection_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(detection_frame, text="Detection Range (m):").pack(side="left", padx=10)
        ctk.CTkEntry(detection_frame, textvariable=self.detection_range_var, width=100).pack(side="left")
        ctk.CTkLabel(detection_frame, text="(Max centerline separation to bundle)", 
                    text_color="#888888").pack(side="left", padx=10)

        # Segment length (analysis resolution)
        seglen_frame = ctk.CTkFrame(params_frame)
        seglen_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(seglen_frame, text="Segment Length (m):").pack(side="left", padx=10)
        ctk.CTkEntry(seglen_frame, textvariable=self.segment_length_var, width=100).pack(side="left")
        ctk.CTkLabel(seglen_frame, text="(Resolution; smaller = slower, finer)",
                    text_color="#888888").pack(side="left", padx=10)

        # Min parallel length
        parallel_frame = ctk.CTkFrame(params_frame)
        parallel_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(parallel_frame, text="Min Parallel Length (m):").pack(side="left", padx=10)
        ctk.CTkEntry(parallel_frame, textvariable=self.min_parallel_var, width=100).pack(side="left")
        ctk.CTkLabel(parallel_frame, text="(Min bundled section)", 
                    text_color="#888888").pack(side="left", padx=10)
        
        # Angular tolerance
        angular_frame = ctk.CTkFrame(params_frame)
        angular_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(angular_frame, text="Angular Tolerance (deg):").pack(side="left", padx=10)
        ctk.CTkEntry(angular_frame, textvariable=self.angular_tolerance_var, width=100).pack(side="left")
        ctk.CTkLabel(angular_frame, text="(Max angle difference)", 
                    text_color="#888888").pack(side="left", padx=10)
        
        from pipeline_calculator.gui.pages.file_select_page import file_actions
        file_actions(main_frame, self.browse_file, self.process_file, self.current_file)

        # Drag and drop
        def on_drop(event):
            try:
                file_path = event.data.strip('{}').strip('"')
                if file_path.lower().endswith(('.kmz', '.kml')):
                    self.process_file(file_path)
                else:
                    messagebox.showerror("Invalid File", "Please select a KMZ or KML file.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")
        
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', on_drop)
    
    def browse_file(self):
        if getattr(self, "_processing", False):
            return
        """Handle file browsing."""
        self.root.withdraw()  # Hide main window temporarily
        
        try:
            filetypes = [
                ("All supported", "*.kmz *.kml"),
                ("KMZ files", "*.kmz"),
                ("KML files", "*.kml"),
                ("All files", "*.*")
            ]
            file_path = filedialog.askopenfilename(filetypes=filetypes)
            
            if file_path:
                self.process_file(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to browse file: {str(e)}")
        finally:
            self.root.deiconify()  # Show main window again
    
    def process_file(self, file_path):
        if getattr(self, '_processing', False) or getattr(self, '_closing', False):
            return
        self._processing = True
        try:
            self.current_results = None
            self.current_file = file_path
            from pipeline_calculator.gui.state import AnalysisParameters
            params, corrections = AnalysisParameters.from_strings(
                self.detection_range_var.get(), self.min_parallel_var.get(),
                self.segment_length_var.get(), self.angular_tolerance_var.get())
            for name, variable in (("detection_range", self.detection_range_var),
                                   ("min_parallel_length", self.min_parallel_var),
                                   ("segment_length", self.segment_length_var),
                                   ("angular_tolerance", self.angular_tolerance_var)):
                if name in corrections:
                    variable.set(corrections[name])
                setattr(self.analyzer, name, getattr(params, name))
            self._analysis_session = AnalysisSession(self.root, self._analysis_done)
            self._analysis_session.start(file_path, params)
        except Exception as e:
            self._processing = False
            if getattr(self, '_analysis_session', None) is not None:
                self._analysis_session.close()
            messagebox.showerror("Processing Error", str(e))
            self.show_file_selection()

    def _analysis_done(self, job):
        if getattr(self, '_closing', False):
            return
        if self._analysis_session is None or self._analysis_session.job is not job:
            return
        self._processing = False
        if job.state == 'completed':
            self.current_results = job.result
            self.show_results()
        else:
            if job.error is not None:
                messagebox.showerror("Processing Error", str(job.error))
            self.show_file_selection()

    def close(self):
        self._closing = True
        if self._analysis_session is not None:
            self._analysis_session.close()
        self.root.destroy()

    def show_results(self):
        """Display analysis results."""
        try:
            # Ensure window is stable and visible
            self.root.deiconify()
            self.root.focus_force()
            try:
                self.root.attributes("-alpha", 1.0)
                self.root.lift()
                self.root.attributes("-topmost", False)  # Prevent flashing on other monitors
                # Set a solid background to avoid transparency artifacts
                self.root.configure(bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
            except Exception:
                pass
            
            # Force window to stay on current monitor
            self.root.update_idletasks()

            # Clear window
            for widget in self.root.winfo_children():
                widget.destroy()

            self.root.title(f"Pipeline Calculator v{__version__} - Results")
            self.root.geometry("1200x800")
            
            # Header
            header_frame = ctk.CTkFrame(self.root)
            header_frame.pack(fill="x", padx=10, pady=5)
            
            file_label = ctk.CTkLabel(header_frame, 
                                     text=f"File: {os.path.basename(self.current_file)}", 
                                     font=("Arial", 12))
            file_label.pack()
            
            # Create tabbed view
            tabview = ctk.CTkTabview(self.root)
            tabview.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Summary tab
            summary_tab = tabview.add("Summary")
            self.create_summary_tab(summary_tab)
            
            # Pipelines tab
            if self.current_results['pipelines']:
                pipeline_tab = tabview.add("Pipelines")
                self.create_pipeline_tab(pipeline_tab)
            
            # Overlap Analysis tab
            if self.current_results['overlap_analysis']:
                overlap_tab = tabview.add("Overlap Analysis")
                self.create_overlap_tab(overlap_tab)
            
            # Placemarks tab
            if self.current_results['placemarks']:
                placemark_tab = tabview.add("Placemarks")
                self.create_placemark_tab(placemark_tab)
            
            # Button frame
            button_frame = ctk.CTkFrame(self.root)
            button_frame.pack(fill="x", padx=10, pady=5)
            
            # Export button
            export_button = ctk.CTkButton(button_frame, text="Export Results", 
                                         command=self.export_results)
            export_button.pack(side="left", padx=5)
            
            # Reanalyze button
            reanalyze_button = ctk.CTkButton(button_frame, 
                                            text="Reanalyze with Different Parameters", 
                                            command=self.reanalyze)
            reanalyze_button.pack(side="left", padx=5)
            
            # New file button
            new_file_button = ctk.CTkButton(button_frame, text="Import New KMZ", 
                                           command=self.show_file_selection)
            new_file_button.pack(side="left", padx=5)
            
            # Close button
            close_button = ctk.CTkButton(button_frame, text="Exit", 
                                        command=self.close)
            close_button.pack(side="right", padx=5)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")
            self.show_file_selection()
    
    def create_summary_tab(self, parent):
        from pipeline_calculator.gui.tabs.summary_tab import create
        create(parent, self.current_results)

    def create_pipeline_tab(self, parent):
        """Create pipeline details tab."""
        # Create treeview
        columns = ("OBJECTID", "Name", "Length (m)", "Length (miles)")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        tree.heading("OBJECTID", text="Object ID")
        tree.heading("Name", text="Name")
        tree.heading("Length (m)", text="Length (meters)")
        tree.heading("Length (miles)", text="Length (miles)")
        
        tree.column("OBJECTID", width=100)
        tree.column("Name", width=300)
        tree.column("Length (m)", width=150)
        tree.column("Length (miles)", width=150)
        
        # Style
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", 
                       background="#2b2b2b", 
                       foreground="white", 
                       fieldbackground="#2b2b2b")
        
        # Populate
        for pipeline in self.current_results['pipelines']:
            tree.insert("", "end", values=(
                pipeline['OBJECTID'],
                pipeline['Name'],
                f"{pipeline['Shape_Length']:.3f}",
                f"{pipeline['pipelinelength']:.6f}"
            ))
        
        # Add total row
        tree.insert("", "end", values=(
            "TOTAL",
            "TOTAL",
            f"{self.current_results['total_meters']:.3f}",
            f"{self.current_results['total_miles']:.6f}"
        ))
        
        tree.pack(fill="both", expand=True, padx=10, pady=10)

    def view_overlap_kml(self, section, index):
        from pipeline_calculator.gui.dialogs.corridor_dialog import CorridorDialog
        try:
            CorridorDialog(self.root, section, index)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open KML file: {str(e)}")

    def create_overlap_tab(self, parent):
        from pipeline_calculator.gui.tabs.overlap_tab import create
        create(parent, self.current_results, on_open_corridor=self.view_overlap_kml)

    def create_placemark_tab(self, parent):
        """Create placemark details tab."""
        # Create treeview
        columns = ("ID", "Name", "Count")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        tree.heading("ID", text="Placemark ID")
        tree.heading("Name", text="Name")
        tree.heading("Count", text="Count")
        
        tree.column("ID", width=150)
        tree.column("Name", width=400)
        tree.column("Count", width=100)
        
        # Populate
        for placemark in self.current_results['placemarks']:
            tree.insert("", "end", values=(
                placemark['Placemark_ID'],
                placemark['Name'],
                placemark['Count']
            ))
        
        tree.pack(fill="both", expand=True, padx=10, pady=10)
    
    def reanalyze(self):
        if getattr(self, "_processing", False):
            return
        """Show in-window parameter editor and reanalyze."""
        try:
            # Clean up any existing parameter frame
            if hasattr(self, 'param_frame') and self.param_frame is not None:
                try:
                    self.param_frame.destroy()
                    self.param_frame = None
                except Exception:
                    pass

            self.param_frame = ctk.CTkFrame(self.root, corner_radius=10)
            self.param_frame.place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(self.param_frame, text="Adjust Analysis Parameters",
                        font=("Arial", 16, "bold")).pack(pady=10, padx=20)

            # Detection range
            detection_frame = ctk.CTkFrame(self.param_frame)
            detection_frame.pack(fill="x", padx=20, pady=10)
            ctk.CTkLabel(detection_frame, text="Detection Range (m):").pack(side="left", padx=10)
            ctk.CTkEntry(detection_frame, textvariable=self.detection_range_var).pack(side="left")

            # Segment length
            seglen_frame = ctk.CTkFrame(self.param_frame)
            seglen_frame.pack(fill="x", padx=20, pady=10)
            ctk.CTkLabel(seglen_frame, text="Segment Length (m):").pack(side="left", padx=10)
            ctk.CTkEntry(seglen_frame, textvariable=self.segment_length_var).pack(side="left")

            # Min parallel
            parallel_frame = ctk.CTkFrame(self.param_frame)
            parallel_frame.pack(fill="x", padx=20, pady=10)
            ctk.CTkLabel(parallel_frame, text="Min Parallel Length (m):").pack(side="left", padx=10)
            ctk.CTkEntry(parallel_frame, textvariable=self.min_parallel_var).pack(side="left")

            # Angular tolerance
            angular_frame = ctk.CTkFrame(self.param_frame)
            angular_frame.pack(fill="x", padx=20, pady=10)
            ctk.CTkLabel(angular_frame, text="Angular Tolerance (deg):").pack(side="left", padx=10)
            ctk.CTkEntry(angular_frame, textvariable=self.angular_tolerance_var).pack(side="left")

            # Buttons
            button_frame = ctk.CTkFrame(self.param_frame)
            button_frame.pack(pady=20)

            def apply_and_analyze():
                if hasattr(self, 'param_frame') and self.param_frame is not None:
                    try:
                        self.param_frame.destroy()
                        self.param_frame = None
                    except Exception:
                        pass
                self.process_file(self.current_file)
            
            def cancel_dialog():
                if hasattr(self, 'param_frame') and self.param_frame is not None:
                    try:
                        self.param_frame.destroy()
                        self.param_frame = None
                    except Exception:
                        pass

            ctk.CTkButton(button_frame, text="Apply & Reanalyze",
                         command=apply_and_analyze).pack(side="left", padx=5)
            ctk.CTkButton(button_frame, text="Cancel",
                         command=cancel_dialog).pack(side="left", padx=5)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show parameter dialog: {str(e)}")
    
    def export_results(self):
        if getattr(self, "_processing", False):
            return
        """Export analysis results.

        Creates a single XLSX workbook with two sheets when `.xlsx` is selected:
        - Pipeline Length Analysis
        - Pipeline Overlap Analysis

        JSON export remains available as an alternative.
        """
        try:
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            
            # Ask for save location
            save_path = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                initialfile=f"{base_name}_analysis.xlsx",
                filetypes=[('Excel Workbook', '*.xlsx'), ('JSON files', '*.json')]
            )
            
            if not save_path:
                return
            
            if save_path.endswith('.json'):
                # Export as JSON
                with open(save_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
            else:
                # Export as XLSX workbook with two sheets
                try:
                    wb = build_analysis_workbook(self.current_results)
                except Exception as ex:
                    messagebox.showerror("Export Error", str(ex))
                    return
                wb.save(save_path)
            
            messagebox.showinfo("Export Complete", f"Results exported to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def run(self):
        """Start the application."""
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Application Error", f"Application error: {str(e)}")


def main():
    """Main entry point."""
    if len(sys.argv) == 3 and sys.argv[1] == '--smoke-test':
        from pipeline_calculator.smoke import run
        return run(sys.argv[2], implementation='legacy')
    try:
        print(f"Pipeline Calculator v{__version__}")
        print(f"Running on {platform.system()} {platform.machine()}")
        print("-" * 50)

        # macOS 26 compatibility: the Apple CommandLineTools Python (3.9) ships Tk 8.5 which
        # aborts when creating a root window. Fail fast with an actionable message instead
        # of crashing deep in Tk initialization.
        if platform.system() == "Darwin":
            try:
                import tkinter as _tk
            except Exception as exc:
                print("ERROR: tkinter is not available in this Python environment.", file=sys.stderr)
                print(f"Details: {exc}", file=sys.stderr)
                print("Fix: on macOS, prefer Homebrew python@3.11 + python-tk@3.11.", file=sys.stderr)
                print("Tip: run `bash scripts/macos/setup_macos.sh` from the repo root.", file=sys.stderr)
                sys.exit(1)

            try:
                _tk_ver = float(getattr(_tk, "TkVersion", 0.0))
            except Exception:
                _tk_ver = 0.0

            if _tk_ver < 8.6:
                print(f"ERROR: Tcl/Tk {_tk_ver} detected. Tk 8.6+ is required for macOS 26 compatibility.", file=sys.stderr)
                print("Fix: install Homebrew python@3.11 + python-tk@3.11 and re-run.", file=sys.stderr)
                print("Tip: run `bash scripts/macos/setup_macos.sh` from the repo root.", file=sys.stderr)
                sys.exit(1)
        
        # Check for required packages (only if not frozen)
        if not getattr(sys, 'frozen', False):
            required = ['pyproj', 'pandas', 'numpy', 'scipy', 'customtkinter', 'tkinterdnd2']
            missing = []
            
            for package in required:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing.append(package)
            
            if missing:
                print(f"Missing packages: {', '.join(missing)}")
                print("Installing...")
                for package in missing:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # Run GUI
        app = PipelineCalculatorGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        try:
            messagebox.showerror("Fatal Error", f"A fatal error occurred:\n\n{str(e)}\n\nThe application will now exit.")
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
