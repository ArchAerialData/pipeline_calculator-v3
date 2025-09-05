# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Pipeline Calculator v3.0, a Python GUI application for analyzing KMZ/KML pipeline files. The application calculates pipeline lengths, detects overlapping parallel sections for aerial survey optimization, and exports results. Built with CustomTkinter for modern UI, uses geospatial libraries for accurate calculations, and includes GitHub Actions for cross-platform executable builds.

## Development Commands

### Running the Application
```bash
# Run from source
python src/pipeline_calculator_v3.py

# Run test KML generator
python generate_test_kml.py
```

### Environment Setup
```bash
# Unix/macOS setup
./setup.sh

# Windows setup  
setup.bat

# Manual setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
```

### Building Executables
```bash
# Build standalone executable
pyinstaller --onefile --windowed --name Pipeline_Calculator_v3 \
  --add-data "README.md:." \
  --hidden-import scipy.spatial \
  --hidden-import scipy._lib.messagestream \
  --hidden-import tkinterdnd2 \
  --hidden-import PIL \
  src/pipeline_calculator_v3.py
```

### GitHub Actions
- Builds automatically trigger on pushes to `main` or tagged releases
- Creates Windows `.exe` and macOS `.dmg` artifacts
- Tagged releases (e.g., `v3.0.0`) create GitHub releases with binaries

## Code Architecture

### Core Components

**PipelineAnalyzer** (`src/pipeline_calculator_v3.py`):
- Main analysis engine combining length calculation and overlap detection
- Uses pyproj/Geod for geodesic calculations with US Survey Miles (1609.347218694m)
- Implements spatial indexing via KDTree for efficient overlap detection
- Segments pipelines into 5m chunks for granular analysis

**PipelineCalculatorGUI** (`src/pipeline_calculator_v3.py`):
- CustomTkinter-based modern dark theme interface
- Tabbed layout: Summary, Pipelines, Overlap Analysis, Placemarks
- Drag-and-drop support via tkinterdnd2
- Real-time parameter adjustment with live re-analysis

### Key Algorithms

**Overlap Detection Process**:
1. Pipeline segmentation into 5-meter segments
2. KD-tree spatial indexing for proximity queries  
3. Bearing comparison within angular tolerance
4. Continuous parallel section identification above minimum length
5. Deduplication to prevent double-counting

**Analysis Parameters**:
- Detection Range: Survey swath width (default: 15m)
- Minimum Parallel Length: Minimum bundled section (default: 200m) 
- Angular Tolerance: Maximum angle difference (default: 15°)
- Segment Length: Fixed at 5m for analysis granularity

### File Structure

```
src/pipeline_calculator_v3.py  # Main application (2000+ lines)
generate_test_kml.py           # KML test file generator
requirements.txt               # Python dependencies
setup.sh / setup.bat          # Environment setup scripts
.github/workflows/build.yaml  # CI/CD pipeline
test_data/                    # Sample KMZ/KML files
icon.ico / icon.icns          # Application icons
```

### Dependencies

**Core Libraries**:
- `pyproj>=3.4.0` - Geospatial coordinate transformations
- `pandas>=1.5.0` - Data manipulation and CSV export
- `numpy==1.24.3` - Numerical computations (pinned version)
- `scipy>=1.10.0` - KDTree spatial indexing
- `customtkinter>=5.0.0` - Modern UI framework

**GUI/Integration**:
- `tkinterdnd2` - Drag-and-drop file support
- `Pillow>=9.0.0` - Image processing for icons
- `pyinstaller>=5.0.0` - Executable building

### Data Flow

1. **Input**: KMZ/KML files via drag-drop or file browser
2. **Parsing**: XML parsing to extract LineString coordinates
3. **Segmentation**: Convert pipelines to 5m segments with bearings
4. **Spatial Analysis**: KDTree indexing and proximity queries
5. **Overlap Detection**: Parallel section identification and bundling
6. **Output**: Tabular results with CSV/JSON export options

### Important Constants

```python
DEFAULT_DETECTION_RANGE = 15      # meters - survey swath width
MIN_PARALLEL_LENGTH = 200         # meters - minimum bundle length  
SEGMENT_LENGTH = 5                # meters - analysis granularity
ANGULAR_TOLERANCE = 15            # degrees - parallelism threshold
GAP_TOLERANCE = 5                 # meters - gap bridging
```

## Claude Code Permissions

This repository contains a legitimate geospatial analysis application for pipeline survey optimization. Claude Code instances are authorized to:

- Read, modify, and create Python files for feature development and bug fixes
- Install and manage Python dependencies via pip/requirements.txt
- Run the application and execute build/test commands
- Create and modify KML/KMZ test data files
- Update documentation, configuration files, and GitHub Actions workflows
- Access and process geospatial coordinate data and pipeline geometry
- Generate temporary files for Google Earth integration and exports

The codebase is defensive in nature, designed for legitimate GIS and aerial survey applications in the energy sector.

## Development Notes

- Application uses threading for file processing to prevent UI freezing
- Error handling includes comprehensive try/catch with user-friendly messages
- Google Earth integration via temporary KML file generation
- Cross-platform compatibility handled via platform-specific PyInstaller builds
- Memory optimization for large files (1000+ pipelines tested)
- @CLAUDE.md