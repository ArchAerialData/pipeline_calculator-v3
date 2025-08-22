#!/usr/bin/env python3
"""
Pipeline Calculator with Overlap Analysis - KMZ/KML Pipeline Calculator
Enhanced version with overlap detection and bundling analysis
Version: 3.0.0
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
warnings.filterwarnings('ignore')

# Version info
__version__ = "3.0.0"
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
        """Extract all features from KMZ/KML file."""
        # Extract KML content
        if file_path.lower().endswith('.kmz'):
            with zipfile.ZipFile(file_path, 'r') as kmz:
                kml_files = [f for f in kmz.namelist() if f.lower().endswith('.kml')]
                if not kml_files:
                    raise ValueError("No KML file found in KMZ archive")
                with kmz.open(kml_files[0]) as kml:
                    kml_content = kml.read().decode('utf-8', errors='ignore')
        else:
            with open(file_path, 'r', encoding='utf-8') as kml:
                kml_content = kml.read()
        
        # Parse XML
        root = ET.fromstring(kml_content)
        
        # Find namespace
        namespace_patterns = [
            {'kml': 'http://www.opengis.net/kml/2.2'},
            {'kml': 'http://earth.google.com/kml/2.2'},
            {'kml': 'http://earth.google.com/kml/2.1'},
            None
        ]
        
        placemarks = []
        namespace = None
        
        for ns in namespace_patterns:
            if ns:
                found_placemarks = root.findall('.//kml:Placemark', ns)
            else:
                found_placemarks = root.findall('.//Placemark')
            
            if found_placemarks:
                placemarks = found_placemarks
                namespace = ns
                break
        
        if not placemarks:
            raise ValueError("No Placemarks found in KML file")
        
        pipelines = []
        placemark_data = []
        pipeline_count = 0
        placemark_count = 0
        
        total_placemarks = len(placemarks)
        
        for i, placemark in enumerate(placemarks):
            if progress_callback:
                progress = (i + 1) / total_placemarks * 0.5  # First half of progress
                progress_callback(progress)
            
            # Extract name
            if namespace:
                name_elem = placemark.find('kml:name', namespace)
            else:
                name_elem = placemark.find('name')
            name = name_elem.text.strip() if name_elem is not None and name_elem.text else f'Item_{i+1}'
            
            # Extract OBJECTID
            objectid = self._extract_objectid(placemark, namespace)
            
            # Extract coordinates
            coords = self._extract_coordinates(placemark, namespace)
            
            if coords:
                # Check geometry type
                has_linestring = self._has_linestring(placemark, namespace)
                has_point = self._has_point(placemark, namespace)
                
                # Process as pipeline if LineString or multiple coords
                if has_linestring or (len(coords) >= 2 and not has_point):
                    pipeline_count += 1
                    pipelines.append({
                        'id': pipeline_count - 1,
                        'objectid': objectid,
                        'name': name,
                        'coordinates': coords
                    })
                    
                # Process as placemark if Point or single coord
                elif has_point or len(coords) == 1:
                    placemark_count += 1
                    placemark_data.append({
                        'Placemark_ID': objectid if objectid != 'N/A' else f'PM_{placemark_count}',
                        'Name': name,
                        'Count': 1
                    })
        
        return pipelines, placemark_data
    
    def _extract_objectid(self, placemark, namespace):
        """Extract OBJECTID from placemark."""
        objectid = 'N/A'
        if namespace:
            objectid_elem = placemark.find('.//kml:Data[@name="OBJECTID"]/kml:value', namespace)
            if objectid_elem is None:
                objectid_elem = placemark.find('.//kml:SimpleData[@name="OBJECTID"]', namespace)
        else:
            objectid_elem = placemark.find('.//Data[@name="OBJECTID"]/value')
            if objectid_elem is None:
                objectid_elem = placemark.find('.//SimpleData[@name="OBJECTID"]')
        
        if objectid_elem is not None and objectid_elem.text:
            objectid = objectid_elem.text.strip()
        return objectid
    
    def _has_linestring(self, placemark, namespace):
        """Check if placemark has LineString geometry."""
        if namespace:
            return placemark.find('.//kml:LineString', namespace) is not None
        return placemark.find('.//LineString') is not None
    
    def _has_point(self, placemark, namespace):
        """Check if placemark has Point geometry."""
        if namespace:
            return placemark.find('.//kml:Point', namespace) is not None
        return placemark.find('.//Point') is not None
    
    def _extract_coordinates(self, placemark, namespace):
        """Extract and parse coordinates from placemark."""
        if namespace:
            coords_elem = placemark.find('.//kml:coordinates', namespace)
        else:
            coords_elem = placemark.find('.//coordinates')
        
        coords = []
        if coords_elem is not None and coords_elem.text:
            coords_text = coords_elem.text.strip()
            
            for coord_str in coords_text.replace('\n', ' ').replace('\t', ' ').split():
                if coord_str.strip():
                    try:
                        parts = coord_str.split(',')
                        if len(parts) >= 2:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                coords.append((lon, lat))
                    except (ValueError, IndexError):
                        continue
        return coords
    
    def calculate_pipeline_lengths(self, pipelines):
        """Calculate individual pipeline lengths."""
        pipeline_data = []
        total_length_meters = 0
        total_length_miles = 0
        
        for pipeline in pipelines:
            length_meters = 0
            coords = pipeline['coordinates']
            
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]
                _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
                length_meters += abs(distance)
            
            length_miles = length_meters / self.survey_mile
            
            pipeline_data.append({
                'OBJECTID': pipeline['objectid'],
                'Name': pipeline['name'],
                'Shape_Length': length_meters,
                'pipelinelength': length_miles
            })
            
            total_length_meters += length_meters
            total_length_miles += length_miles
        
        return pipeline_data, total_length_meters, total_length_miles
    
    def segment_pipeline(self, coordinates):
        """Break pipeline into fixed-length segments for analysis."""
        segments = []
        accumulated_distance = 0
        
        for i in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]
            
            azimuth, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
            accumulated_distance += distance
            
            while accumulated_distance >= self.segment_length:
                ratio = (self.segment_length - (accumulated_distance - distance)) / distance
                mid_lon = lon1 + ratio * (lon2 - lon1)
                mid_lat = lat1 + ratio * (lat2 - lat1)
                
                segments.append({
                    'midpoint': (mid_lon, mid_lat),
                    'bearing': azimuth,
                    'length': self.segment_length,
                    'segment_index': len(segments)
                })
                
                accumulated_distance -= self.segment_length
                lon1, lat1 = mid_lon, mid_lat
        
        return segments
    
    def find_parallel_segments(self, pipelines, progress_callback=None):
        """Identify pipeline segments that run parallel within detection range."""
        # Segment all pipelines
        for idx, pipeline in enumerate(pipelines):
            if progress_callback:
                progress = 0.5 + (idx / len(pipelines)) * 0.25  # 50-75% progress
                progress_callback(progress)
            pipeline['segments'] = self.segment_pipeline(pipeline['coordinates'])
        
        # Build spatial index
        all_segments = []
        segment_to_pipeline = {}
        
        for p_idx, pipeline in enumerate(pipelines):
            for seg in pipeline['segments']:
                seg_idx = len(all_segments)
                all_segments.append(seg['midpoint'])
                segment_to_pipeline[seg_idx] = (p_idx, seg)
        
        if not all_segments:
            return {}
        
        points = np.array([(lon, lat) for lon, lat in all_segments])
        tree = KDTree(points)
        
        # Find parallel segments
        parallel_groups = defaultdict(list)
        
        for seg_idx, (p_idx, segment) in segment_to_pipeline.items():
            nearby_indices = tree.query_ball_point(points[seg_idx], 
                                                  self.detection_range / 111000)
            
            for near_idx in nearby_indices:
                if near_idx == seg_idx:
                    continue
                
                near_p_idx, near_segment = segment_to_pipeline[near_idx]
                
                if p_idx == near_p_idx:
                    continue
                
                bearing_diff = abs(segment['bearing'] - near_segment['bearing'])
                bearing_diff = min(bearing_diff, 360 - bearing_diff)
                
                if bearing_diff <= self.angular_tolerance:
                    lon1, lat1 = segment['midpoint']
                    lon2, lat2 = near_segment['midpoint']
                    _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
                    
                    if distance <= self.detection_range:
                        key = tuple(sorted([p_idx, near_p_idx]))
                        parallel_groups[key].append({
                            'pipeline_1_segment': segment['segment_index'],
                            'pipeline_2_segment': near_segment['segment_index'],
                            'distance': distance
                        })
        
        return parallel_groups
    
    def calculate_overlap_results(self, pipelines, parallel_groups, progress_callback=None):
        """Calculate bundled lengths and overlap statistics."""
        results = {
            'bundled_sections': [],
            'pipeline_overlaps': {},
            'total_bundled_length': 0,
            'effective_total_length': 0,
            'savings_meters': 0,
            'savings_miles': 0,
            'savings_percentage': 0,
            'parameter_impacts': {}
        }
        
        bundled_segments = defaultdict(set)
        
        # Process parallel groups
        for (p1_idx, p2_idx), segments in parallel_groups.items():
            if not segments:
                continue
            
            segments.sort(key=lambda x: x['pipeline_1_segment'])
            
            # Find continuous sections
            continuous_sections = []
            current_section = [segments[0]]
            
            for seg in segments[1:]:
                if (seg['pipeline_1_segment'] - current_section[-1]['pipeline_1_segment'] <= 2 and
                    seg['pipeline_2_segment'] - current_section[-1]['pipeline_2_segment'] <= 2):
                    current_section.append(seg)
                else:
                    if len(current_section) * self.segment_length >= self.min_parallel_length:
                        continuous_sections.append(current_section)
                    current_section = [seg]
            
            if len(current_section) * self.segment_length >= self.min_parallel_length:
                continuous_sections.append(current_section)
            
            for section in continuous_sections:
                bundled_length = len(section) * self.segment_length
                avg_distance = np.mean([s['distance'] for s in section])
                
                for seg in section:
                    bundled_segments[p1_idx].add(seg['pipeline_1_segment'])
                    bundled_segments[p2_idx].add(seg['pipeline_2_segment'])
                
                results['bundled_sections'].append({
                    'pipeline_1': pipelines[p1_idx]['name'],
                    'pipeline_2': pipelines[p2_idx]['name'],
                    'bundled_length_meters': bundled_length,
                    'bundled_length_miles': bundled_length / self.survey_mile,
                    'average_separation': avg_distance,
                    'segment_count': len(section)
                })
        
        # Calculate per-pipeline overlaps
        for p_idx, pipeline in enumerate(pipelines):
            bundled_count = len(bundled_segments[p_idx])
            bundled_length = bundled_count * self.segment_length
            
            results['pipeline_overlaps'][pipeline['name']] = {
                'bundled_segments': bundled_count,
                'bundled_length_meters': bundled_length,
                'bundled_length_miles': bundled_length / self.survey_mile
            }
        
        # Calculate totals
        total_bundled = sum(section['segment_count'] * self.segment_length
                            for section in results['bundled_sections'])
        results['total_bundled_length'] = total_bundled
        
        if progress_callback:
            progress_callback(1.0)  # 100% complete
        
        return results
    
    def analyze_complete(self, file_path, progress_callback=None):
        """Complete analysis of KMZ/KML file."""
        # Extract features
        pipelines, placemarks = self.extract_features_from_file(file_path, progress_callback)
        
        # Calculate basic lengths
        pipeline_data, total_meters, total_miles = self.calculate_pipeline_lengths(pipelines)
        
        # Perform overlap analysis if multiple pipelines
        overlap_results = None
        if len(pipelines) >= 2:
            parallel_groups = self.find_parallel_segments(pipelines, progress_callback)
            overlap_results = self.calculate_overlap_results(pipelines, parallel_groups, progress_callback)
            
            # Calculate effective total
            total_savings = sum(section['bundled_length_meters'] * 0.5 
                              for section in overlap_results['bundled_sections'])
            overlap_results['effective_total_meters'] = total_meters - total_savings
            overlap_results['effective_total_miles'] = overlap_results['effective_total_meters'] / self.survey_mile
            overlap_results['savings_meters'] = total_savings
            overlap_results['savings_miles'] = total_savings / self.survey_mile
            overlap_results['savings_percentage'] = (total_savings / total_meters * 100) if total_meters > 0 else 0
        
        return {
            'pipelines': pipeline_data,
            'placemarks': placemarks,
            'total_meters': total_meters,
            'total_miles': total_miles,
            'overlap_analysis': overlap_results,
            'analysis_parameters': {
                'detection_range': self.detection_range,
                'min_parallel_length': self.min_parallel_length,
                'segment_length': self.segment_length,
                'angular_tolerance': self.angular_tolerance
            }
        }


class PipelineCalculatorGUI:
    """Main GUI application for pipeline calculator with overlap analysis."""
    
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.analyzer = PipelineAnalyzer()
        self.current_results = None
        self.current_file = None
        
        # Analysis parameter variables
        self.detection_range_var = DoubleVar(value=DEFAULT_DETECTION_RANGE)
        self.min_parallel_var = DoubleVar(value=MIN_PARALLEL_LENGTH)
        self.segment_length_var = DoubleVar(value=SEGMENT_LENGTH)
        self.angular_tolerance_var = DoubleVar(value=ANGULAR_TOLERANCE)
        
        self.setup_gui()
    
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
        ctk.CTkLabel(detection_frame, text="(Survey swath width)", 
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
            file_path = event.data.strip('{}').strip('"')
            if file_path.lower().endswith(('.kmz', '.kml')):
                self.process_file(file_path)
            else:
                messagebox.showerror("Invalid File", "Please select a KMZ or KML file.")
        
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', on_drop)
    
    def browse_file(self):
        """Handle file browsing."""
        self.root.withdraw()  # Hide main window temporarily
        
        filetypes = [
            ("All supported", "*.kmz *.kml"),
            ("KMZ files", "*.kmz"),
            ("KML files", "*.kml"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        self.root.deiconify()  # Show main window again
        
        if file_path:
            self.process_file(file_path)
    
    def process_file(self, file_path):
        """Process selected file with progress indication."""
        self.current_file = file_path
        
        # Update analyzer parameters
        self.analyzer.detection_range = self.detection_range_var.get()
        self.analyzer.min_parallel_length = self.min_parallel_var.get()
        self.analyzer.segment_length = self.segment_length_var.get()
        self.analyzer.angular_tolerance = self.angular_tolerance_var.get()
        
        # Create progress window
        progress_window = ctk.CTkToplevel(self.root)
        progress_window.title("Processing...")
        progress_window.geometry("450x220")
        progress_window.resizable(False, False)
        
        # Center progress window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
        y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
        progress_window.geometry(f"+{x}+{y}")
        
        # Progress content
        progress_frame = ctk.CTkFrame(progress_window)
        progress_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        status_label = ctk.CTkLabel(progress_frame, 
                                   text="Analyzing pipelines and overlaps...", 
                                   font=("Arial", 14))
        status_label.pack(pady=20)
        
        progress_bar = ctk.CTkProgressBar(progress_frame, width=300)
        progress_bar.pack(pady=10)
        progress_bar.set(0)
        
        file_label = ctk.CTkLabel(progress_frame, 
                                 text=f"File: {os.path.basename(file_path)}", 
                                 font=("Arial", 10), 
                                 text_color="#CCCCCC")
        file_label.pack(pady=10)
        
        progress_window.update()
        
        # Progress callback
        def update_progress(progress):
            self.root.after(0, lambda: progress_bar.set(progress))
        
        # Worker thread
        result_holder = {}
        
        def worker():
            try:
                result = self.analyzer.analyze_complete(file_path, update_progress)
                result_holder['result'] = result
            except Exception as e:
                result_holder['error'] = e
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        # Check thread completion
        def check_thread():
            if thread.is_alive():
                self.root.after(100, check_thread)
            else:
                progress_window.destroy()
                if 'error' in result_holder:
                    messagebox.showerror("Processing Error", str(result_holder['error']))
                    self.show_file_selection()
                else:
                    self.current_results = result_holder['result']
                    self.show_results()
        
        check_thread()
    
    def show_results(self):
        """Display analysis results."""
        # Ensure window is visible and fully opaque
        self.root.deiconify()
        try:
            self.root.attributes("-alpha", 1.0)
            self.root.lift()
            # Set a solid background to avoid transparency artifacts
            self.root.configure(bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        except Exception:
            pass

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
    
    def create_overlap_tab(self, parent):
        """Create overlap analysis details tab."""
        overlap = self.current_results['overlap_analysis']
        
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bundled sections
        ctk.CTkLabel(scroll_frame, text="Bundled Pipeline Sections", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        if overlap['bundled_sections']:
            for section in overlap['bundled_sections']:
                section_frame = ctk.CTkFrame(scroll_frame)
                section_frame.pack(fill="x", padx=20, pady=5)
                
                text = f"{section['pipeline_1']} ↔ {section['pipeline_2']}\n"
                text += f"Length: {section['bundled_length_miles']:.3f} miles | "
                text += f"Avg Separation: {section['average_separation']:.1f} m"
                
                ctk.CTkLabel(section_frame, text=text, 
                           font=("Arial", 12)).pack(pady=5)
        else:
            ctk.CTkLabel(scroll_frame, 
                        text="No bundled sections found with current parameters",
                        font=("Arial", 12)).pack()
    
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
        """Show parameter dialog and reanalyze."""
        # Create parameter dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Adjust Analysis Parameters")
        dialog.geometry("500x400")
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Parameters
        ctk.CTkLabel(dialog, text="Adjust Analysis Parameters", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        # Detection range
        detection_frame = ctk.CTkFrame(dialog)
        detection_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(detection_frame, text="Detection Range (m):").pack(side="left", padx=10)
        detection_entry = ctk.CTkEntry(detection_frame, textvariable=self.detection_range_var)
        detection_entry.pack(side="left")
        
        # Min parallel
        parallel_frame = ctk.CTkFrame(dialog)
        parallel_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(parallel_frame, text="Min Parallel Length (m):").pack(side="left", padx=10)
        parallel_entry = ctk.CTkEntry(parallel_frame, textvariable=self.min_parallel_var)
        parallel_entry.pack(side="left")
        
        # Angular tolerance
        angular_frame = ctk.CTkFrame(dialog)
        angular_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(angular_frame, text="Angular Tolerance (°):").pack(side="left", padx=10)
        angular_entry = ctk.CTkEntry(angular_frame, textvariable=self.angular_tolerance_var)
        angular_entry.pack(side="left")
        
        # Buttons
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(pady=20)
        
        def apply_and_analyze():
            dialog.destroy()
            self.process_file(self.current_file)
        
        ctk.CTkButton(button_frame, text="Apply & Reanalyze", 
                     command=apply_and_analyze).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", 
                     command=dialog.destroy).pack(side="left", padx=5)
    
    def export_results(self):
        """Export analysis results."""
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            initialfile=f"{base_name}_analysis",
            filetypes=[('CSV files', '*.csv'), ('JSON files', '*.json')]
        )
        
        if not save_path:
            return
        
        try:
            if save_path.endswith('.json'):
                # Export as JSON
                with open(save_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
            else:
                # Export as CSV
                # Pipeline data
                pipeline_df = pd.DataFrame(self.current_results['pipelines'])
                pipeline_df.to_csv(save_path, index=False)
                
                # Overlap data
                if self.current_results['overlap_analysis']:
                    overlap_path = save_path.replace('.csv', '_overlaps.csv')
                    overlap_df = pd.DataFrame(self.current_results['overlap_analysis']['bundled_sections'])
                    overlap_df.to_csv(overlap_path, index=False)
                    
                    # Summary
                    summary_path = save_path.replace('.csv', '_summary.txt')
                    with open(summary_path, 'w') as f:
                        f.write("Pipeline Analysis Summary\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Total Original Length: {self.current_results['total_miles']:.3f} miles\n")
                        f.write(f"Effective Survey Length: {self.current_results['overlap_analysis']['effective_total_miles']:.3f} miles\n")
                        f.write(f"Survey Savings: {self.current_results['overlap_analysis']['savings_miles']:.3f} miles\n")
                        f.write(f"Savings Percentage: {self.current_results['overlap_analysis']['savings_percentage']:.1f}%\n")
            
            messagebox.showinfo("Export Complete", f"Results exported to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
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


if __name__ == "__main__":
    main()