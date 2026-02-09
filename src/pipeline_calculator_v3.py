#!/usr/bin/env python3
"""
Pipeline Calculator with Overlap Analysis - KMZ/KML Pipeline Calculator
Enhanced version with overlap detection and bundling analysis
Version: 3.0.0 - Fixed
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
__version__ = "3.0.0-fixed"
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
        """Compute effective length using per-segment clustering across pipelines.

        For each segment midpoint, find nearby parallel segments on other pipelines
        within detection range. If k pipelines share that neighborhood, attribute
        only 1/k of that segment length to the effective total. This avoids
        double-counting and naturally handles 3+ parallel pipelines.
        """
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
        )
    
    def analyze_complete(self, file_path, progress_callback=None):
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
        return core.analyze_complete(file_path, progress_callback=progress_callback)


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
        ctk.CTkLabel(angular_frame, text="Angular Tolerance (°):").pack(side="left", padx=10)
        ctk.CTkEntry(angular_frame, textvariable=self.angular_tolerance_var, width=100).pack(side="left")
        ctk.CTkLabel(angular_frame, text="(Max angle difference)", 
                    text_color="#888888").pack(side="left", padx=10)
        
        # Browse button
        browse_button = ctk.CTkButton(main_frame, text="Browse Files", 
                                     command=self.browse_file, 
                                     width=200, height=40)
        browse_button.pack(pady=20)
        
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
        """Process selected file with progress indication."""
        try:
            self.current_file = file_path
            
            # Validate parameter values
            detection_range = max(1, self._get_float_var(self.detection_range_var, DEFAULT_DETECTION_RANGE))
            min_parallel = max(10, self._get_float_var(self.min_parallel_var, MIN_PARALLEL_LENGTH))
            segment_length = max(1, self._get_float_var(self.segment_length_var, SEGMENT_LENGTH))
            angular_tolerance = max(1, min(90, self._get_float_var(self.angular_tolerance_var, ANGULAR_TOLERANCE)))
            
            # Update analyzer parameters
            self.analyzer.detection_range = detection_range
            self.analyzer.min_parallel_length = min_parallel
            self.analyzer.segment_length = segment_length
            self.analyzer.angular_tolerance = angular_tolerance
            
            # Ensure window is stable and visible
            self.root.focus_force()
            self.root.update_idletasks()
            
            # Create in-window progress overlay
            progress_frame = ctk.CTkFrame(self.root, corner_radius=10)
            progress_frame.place(relx=0.5, rely=0.5, anchor="center")
            
            # Ensure overlay is on top
            progress_frame.lift()

            status_label = ctk.CTkLabel(progress_frame,
                                       text="Analyzing pipelines and overlaps...",
                                       font=("Arial", 14))
            status_label.pack(pady=20, padx=20)

            progress_bar = ctk.CTkProgressBar(progress_frame, width=300, mode="indeterminate")
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Force UI update
            self.root.update()

            # Worker thread
            result_holder = {}
            analysis_complete = threading.Event()

            def worker():
                try:
                    result_holder['result'] = self.analyzer.analyze_complete(file_path)
                except Exception as e:
                    result_holder['error'] = e
                finally:
                    analysis_complete.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            # Check thread completion with better error handling
            def check_thread():
                if not analysis_complete.is_set():
                    self.root.after(100, check_thread)
                else:
                    try:
                        progress_bar.stop()
                        progress_frame.destroy()
                    except Exception:
                        pass
                        
                    if 'error' in result_holder:
                        error_msg = str(result_holder['error'])
                        messagebox.showerror("Processing Error", 
                                           f"Failed to process file:\n\n{error_msg}\n\nPlease check that the file is a valid KMZ/KML file.")
                        self.show_file_selection()
                    else:
                        self.current_results = result_holder['result']
                        self.show_results()

            check_thread()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {str(e)}")
            self.show_file_selection()
    
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
                                        command=self.root.quit)
            close_button.pack(side="right", padx=5)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")
            self.show_file_selection()
    
    def create_summary_tab(self, parent):
        """Create summary tab with key metrics."""
        summary_frame = ctk.CTkScrollableFrame(parent)
        summary_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(summary_frame, text="Analysis Summary", 
                    font=("Arial", 20, "bold")).pack(pady=10)
        
        # Original totals
        original_frame = ctk.CTkFrame(summary_frame)
        original_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(original_frame, text="Original Pipeline Totals", 
                    font=("Arial", 16, "bold"), 
                    text_color="#FFD700").pack()
        
        total_miles = self.current_results['total_miles']
        ctk.CTkLabel(original_frame, 
                    text=f"Total Length: {total_miles:.3f} US Survey Miles",
                    font=("Arial", 14)).pack()
        
        ctk.CTkLabel(original_frame, 
                    text=f"Pipeline Count: {len(self.current_results['pipelines'])}",
                    font=("Arial", 14)).pack()
        
        # Overlap analysis results
        if self.current_results['overlap_analysis']:
            overlap = self.current_results['overlap_analysis']
            
            # Adjusted totals
            adjusted_frame = ctk.CTkFrame(summary_frame)
            adjusted_frame.pack(fill="x", pady=10)
            
            ctk.CTkLabel(adjusted_frame, text="Adjusted for Overlaps", 
                        font=("Arial", 16, "bold"), 
                        text_color="#87CEEB").pack()
            
            effective_miles = overlap['effective_total_miles']
            ctk.CTkLabel(adjusted_frame, 
                        text=f"Effective Survey Length: {effective_miles:.3f} US Survey Miles",
                        font=("Arial", 14)).pack()
            
            savings_miles = overlap['savings_miles']
            savings_pct = overlap['savings_percentage']
            ctk.CTkLabel(adjusted_frame, 
                        text=f"Survey Savings: {savings_miles:.3f} miles ({savings_pct:.1f}%)",
                        font=("Arial", 14), 
                        text_color="#90EE90").pack()
            
            # Bundled sections count
            bundle_count = len(overlap['bundled_sections'])
            ctk.CTkLabel(adjusted_frame, 
                        text=f"Bundled Sections: {bundle_count}",
                        font=("Arial", 14)).pack()
        
        # Analysis parameters
        params_frame = ctk.CTkFrame(summary_frame)
        params_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(params_frame, text="Analysis Parameters Used", 
                    font=("Arial", 16, "bold")).pack()
        
        params = self.current_results['analysis_parameters']
        param_text = f"Detection Range: {params['detection_range']} m\n"
        param_text += f"Min Parallel Length: {params['min_parallel_length']} m\n"
        param_text += f"Angular Tolerance: {params['angular_tolerance']}°"
        
        ctk.CTkLabel(params_frame, text=param_text, 
                    font=("Arial", 12)).pack()
        # Additional parameters for clarity
        params2 = self.current_results['analysis_parameters']
        param_text2 = f"Segment Length: {params2.get('segment_length', self.analyzer.segment_length)} m\n"
        param_text2 += f"Angular Tolerance: {params2['angular_tolerance']} deg"
        ctk.CTkLabel(params_frame, text=param_text2,
                    font=("Arial", 12), text_color="#AAAAAA").pack()
    
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
        """Generate a temporary KML with polygon corridor for the bundled section and open it."""
        try:
            kml = build_overlap_corridor_kml(section, index)

            with tempfile.NamedTemporaryFile('w', suffix=f'_corridor_{index:03d}.kml', delete=False, encoding='utf-8') as tmp:
                tmp.write(kml)
                path = tmp.name

            try:
                if sys.platform.startswith('win'):
                    os.startfile(path)  # nosec - temporary path
                elif sys.platform == 'darwin':
                    subprocess.run(['open', path], check=False)
                else:
                    subprocess.run(['xdg-open', path], check=False)
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open KML file: {str(e)}")

    def create_overlap_tab(self, parent):
        """Create Overlap Analysis tab using a proper table with aligned columns.

        Replaces the free-form row layout with a ttk.Treeview so that every
        cell aligns with its column header, similar to a spreadsheet.
        """
        overlap = self.current_results['overlap_analysis']

        # Main container
        main_frame = ctk.CTkFrame(parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        ctk.CTkLabel(
            main_frame,
            text="Bundled Pipeline Sections",
            font=("Arial", 16, "bold"),
        ).pack(pady=(10, 0))

        if overlap['bundled_sections']:
            # Frame to host the tree and its scrollbar
            table_frame = ctk.CTkFrame(main_frame)
            table_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Configure a dark style for Treeview to match the app theme
            style = ttk.Style()
            try:
                style.theme_use("default")
            except Exception:
                pass
            style.configure(
                "Overlap.Treeview",
                background="#2b2b2b",
                foreground="white",
                fieldbackground="#2b2b2b",
                rowheight=26,
            )
            style.configure("Overlap.Treeview.Heading", font=("Arial", 12, "bold"))

            # Define columns
            columns = ("Pipeline Pair", "Length (miles)", "Avg Sep (m)", "Action")
            tree = ttk.Treeview(
                table_frame,
                columns=columns,
                show="headings",
                height=20,
                style="Overlap.Treeview",
            )
            # Ensure the implicit '#0' column has zero width so bbox math lines up
            try:
                tree.column('#0', width=0, stretch=False)
            except Exception:
                pass

            # Scrollbar
            vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)

            # Configure column headings
            tree.heading("Pipeline Pair", text="Pipeline Pair")
            tree.heading("Length (miles)", text="Length (miles)")
            tree.heading("Avg Sep (m)", text="Avg Sep (m)")
            tree.heading("Action", text="Action")

            # Column widths
            tree.column("Pipeline Pair", width=700, anchor="w")
            tree.column("Length (miles)", width=150, anchor="center")
            tree.column("Avg Sep (m)", width=120, anchor="center")
            tree.column("Action", width=110, anchor="center")

            # Determine which sections to show (keep previous top-20 behavior)
            sections_to_display = (
                overlap['bundled_sections'][:20]
                if len(overlap['bundled_sections']) > 20
                else overlap['bundled_sections']
            )

            # Map of item-id -> (section, index) for event handlers
            item_map = {}
            for idx, section in enumerate(sections_to_display, start=1):
                pair_text = f"{section['pipeline_1']} + {section['pipeline_2']}"
                item_id = tree.insert(
                    "",
                    "end",
                    values=(
                        pair_text,
                        f"{section['bundled_length_miles']:.3f}",
                        f"{section['average_separation']:.1f}",
                        "",  # leave cell blank; real button is overlaid
                    ),
                )
                item_map[item_id] = (section, idx)

            # Event handlers for action clicks / double-click anywhere on a row
            def _open_for_item(item_id):
                try:
                    section, idx = item_map[item_id]
                    self.view_overlap_kml(section, idx)
                except Exception:
                    pass

            def on_click(event):
                # Trigger only when clicking the Action column
                region = tree.identify("region", event.x, event.y)
                if region != "cell":
                    return
                row_id = tree.identify_row(event.y)
                col = tree.identify_column(event.x)  # '#1' .. '#n'
                if row_id and col == "#4":  # Action column
                    _open_for_item(row_id)

            def on_double_click(event):
                row_id = tree.identify_row(event.y)
                if row_id:
                    _open_for_item(row_id)

            tree.bind("<ButtonRelease-1>", on_click)
            tree.bind("<Double-1>", on_double_click)

            # Overlay real blue buttons inside the Action column (Treeview doesn't natively support widgets per cell)
            action_buttons = {}

            def ensure_buttons_positioned(event=None):
                try:
                    # Place a button for each visible row
                    for item_id in tree.get_children(""):
                        # Use column identifier to avoid off-by-one with hidden '#0'
                        bbox = tree.bbox(item_id, column="Action")
                        btn = action_buttons.get(item_id)
                        if not bbox:
                            # Item is not visible; hide any existing button
                            if btn:
                                btn.place_forget()
                            continue
                        x, y, w, h = bbox
                        if btn is None:
                            # Create button lazily
                            section, idx = item_map[item_id]
                            def make_cmd(s=section, i=idx):
                                return lambda: self.view_overlap_kml(s, i)
                            btn = ctk.CTkButton(
                                table_frame,
                                text="View Corridor",
                                width=min(110, max(80, w - 8)),
                                height=min(26, max(22, h - 6)),
                                command=make_cmd(),
                            )
                            action_buttons[item_id] = btn
                        # Convert tree-relative bbox to parent coords
                        btn.place(x=tree.winfo_x() + x + (w // 2),
                                  y=tree.winfo_y() + y + (h // 2),
                                  anchor="center")
                except Exception:
                    pass

            # Keep buttons aligned on scroll/resize
            def on_tree_scroll(first, last):
                try:
                    vsb.set(first, last)
                finally:
                    ensure_buttons_positioned()

            tree.configure(yscrollcommand=on_tree_scroll)
            tree.bind("<Configure>", ensure_buttons_positioned)
            tree.bind("<ButtonRelease-1>", ensure_buttons_positioned, add="+")
            tree.bind("<Motion>", lambda e: None)  # keep events active on Windows
            try:
                tree.after(100, ensure_buttons_positioned)
            except Exception:
                pass

            # Layout the tree + scrollbar
            tree.pack(side="left", fill="both", expand=True)
            vsb.pack(side="right", fill="y")

            # Info if truncated to top 20
            if len(overlap['bundled_sections']) > 20:
                ctk.CTkLabel(
                    main_frame,
                    text=(
                        f"Showing top 20 of {len(overlap['bundled_sections'])} bundled sections "
                        "(sorted by length). Double-click a row, or click 'View' in the Action column "
                        "to open its corridor in Google Earth."
                    ),
                    font=("Arial", 10),
                    text_color="#888888",
                    justify="center",
                    wraplength=1100,
                ).pack(pady=(4, 6))

            # Summary statistics at bottom
            summary_frame = ctk.CTkFrame(main_frame)
            summary_frame.pack(fill="x", pady=10)
            total_bundled = sum(s['bundled_length_miles'] for s in overlap['bundled_sections'])
            ctk.CTkLabel(
                summary_frame,
                text=(
                    f"Total Bundled Length: {total_bundled:.3f} miles across "
                    f"{len(overlap['bundled_sections'])} sections"
                ),
                font=("Arial", 12, "bold"),
            ).pack()
        else:
            ctk.CTkLabel(
                main_frame,
                text="No bundled sections found with current parameters",
                font=("Arial", 12),
            ).pack(pady=20)
    
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
            ctk.CTkLabel(angular_frame, text="Angular Tolerance (°):").pack(side="left", padx=10)
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
    try:
        print(f"Pipeline Calculator v{__version__}")
        print(f"Running on {platform.system()} {platform.machine()}")
        print("-" * 50)
        
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
    main()
