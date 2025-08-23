# Pipeline Calculator v3.0 - With Overlap Analysis

A comprehensive GUI application for calculating pipeline lengths and analyzing overlaps from KMZ/KML files. Designed for GIS professionals and aerial survey planning to optimize flight paths by identifying bundled pipeline sections.

## 🚀 Key Features

### New in v3.0
- **Overlap Detection**: Automatically identifies parallel pipeline sections that can be surveyed in a single pass
- **Adjustable Parameters**: Real-time parameter adjustment for different survey altitudes and equipment
- **Cost Optimization**: Calculates effective survey length accounting for bundled sections
- **Detailed Analytics**: Provides comprehensive breakdown of overlap statistics
- **Google Earth Preview**: Open bundled sections directly in Google Earth with a single click
- **Native App Icons**: Displays `.ico` on Windows and `.icns` on macOS for proper branding

### Core Features
- Drag-and-drop file support for KMZ/KML files
- Calculates pipeline lengths in meters and US Survey Miles
- Identifies and counts placemarks (point features)
- Tabbed interface for organized data viewing
- Export results to CSV and JSON formats
- Dark mode interface for reduced eye strain
- Progress indicators for large file processing

## 📊 Overlap Analysis Capabilities

The overlap analysis feature helps optimize aerial survey planning by:
- Identifying pipelines within detection range that run parallel
- Calculating bundled sections that can be captured in one survey pass
- Providing adjusted total lengths for accurate cost estimation
- Showing percentage savings from bundled surveying

### Adjustable Parameters

- **Detection Range**: Survey swath width (default: 15m)
  - Adjust based on flight altitude and sensor capabilities
- **Minimum Parallel Length**: Minimum bundled section (default: 200m)
  - Filter out short overlaps that aren't worth bundling
- **Angular Tolerance**: Maximum angle difference (default: 15°)
  - Define how parallel pipelines need to be

## 🔧 Installation

### Option 1: Download Pre-built Executables
Download the latest release from the GitHub releases page:
- **Windows**: `Pipeline_Calculator_v3.exe`
- **macOS**: `Pipeline_Calculator_v3_macOS.dmg`

### Option 2: Run from Source
Requires Python 3.8 or higher.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pipeline-calculator-v3.git
   cd pipeline-calculator-v3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python src/pipeline_calculator_v3.py
   ```

## 📝 Usage Guide

### Basic Workflow

1. **Launch the application**
   - Double-click the executable, or
   - Run `python src/pipeline_calculator_v3.py`

2. **Configure analysis parameters** (optional)
   - Adjust Detection Range for your survey altitude
   - Set Minimum Parallel Length based on project requirements
   - Modify Angular Tolerance for pipeline alignment sensitivity

3. **Load a KMZ/KML file**
   - Drag and drop onto the application window, or
   - Click "Browse Files" to select

4. **Review results across tabs**:
   - **Summary**: Key metrics including original and adjusted totals
   - **Pipelines**: Individual pipeline lengths and details
   - **Overlap Analysis**: Bundled sections and savings
   - **Placemarks**: Point features if present

5. **Export or reanalyze**
   - Export results to CSV/JSON
   - Reanalyze with different parameters
   - Import new KMZ file

### Understanding the Results

- **Original Total Length**: Sum of all individual pipeline lengths
- **Effective Survey Length**: Adjusted length accounting for overlaps
- **Survey Savings**: Reduction in survey distance from bundling
- **Bundled Sections**: Specific pipeline pairs that can be surveyed together

## 🎯 Use Cases

### Aerial Survey Planning
- Optimize flight paths for methane detection surveys
- Calculate accurate survey distances for cost estimation
- Identify opportunities for multi-pipeline capture

### Pipeline Management
- Inventory pipeline networks
- Analyze corridor density
- Plan maintenance surveys

### GIS Integration
- Compatible with Google Earth exports
- Works with ArcGIS and QGIS KML files
- CSV exports for further analysis

## 🏗️ GitHub Repository Structure

To set up this project in GitHub for successful builds:

```
pipeline-calculator-v3/
├── .github/
│   └── workflows/
│       └── build.yaml          # GitHub Actions workflow
├── src/
│   └── pipeline_calculator_v3.py   # Main application
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
└── icon.icns                   # macOS app icon (optional)
```

### Setup Instructions:

1. **Create a new GitHub repository**

2. **Add the files**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Pipeline Calculator v3.0"
   git branch -M main
   git remote add origin https://github.com/yourusername/pipeline-calculator-v3.git
   git push -u origin main
   ```

3. **Enable GitHub Actions**:
   - Go to Settings → Actions → General
   - Select "Allow all actions and reusable workflows"

4. **Create a release** to trigger builds:
   ```bash
   git tag v3.0.0
   git push origin v3.0.0
   ```

The GitHub Actions workflow will automatically build executables for Windows and macOS when you push to main or create a tagged release.

## 🔬 Technical Details

### Overlap Detection Algorithm

The application uses a sophisticated algorithm to detect overlaps:

1. **Segmentation**: Pipelines are divided into 5-meter segments
2. **Spatial Indexing**: KD-tree structure for efficient proximity queries
3. **Parallel Detection**: Bearing comparison within angular tolerance
4. **Bundling**: Continuous parallel sections above minimum length
5. **Deduplication**: Prevents double-counting of bundled segments

### Data Processing

- **Coordinate System**: WGS84 geodesic calculations
- **Distance Units**: US Survey Miles (1609.347218694 meters)
- **Performance**: Optimized for files with 1000+ pipelines

## 🐛 Troubleshooting

### Common Issues

1. **"No overlaps detected"**
   - Try increasing Detection Range
   - Reduce Minimum Parallel Length
   - Check if pipelines are actually close together

2. **File explorer freezes**
   - Application now properly manages window states
   - Use Browse button if drag-and-drop causes issues

3. **Memory issues with large files**
   - Files are processed with progress indication
   - Consider splitting very large KMZ files (>100MB)

4. **Incorrect overlap calculations**
   - Verify Angular Tolerance setting
   - Check that pipelines are properly formatted LineStrings

## 📊 Parameter Impact Guide

### Detection Range (altitude-dependent)
- **10m**: Low altitude, high-resolution surveys
- **15m**: Standard aerial surveys (default)
- **25m**: Higher altitude, wider coverage
- **50m**: Satellite or high-altitude surveillance

### Minimum Parallel Length
- **50m**: Aggressive bundling (may increase false positives)
- **200m**: Conservative bundling (default)
- **500m**: Only long continuous sections

### Angular Tolerance
- **5°**: Strictly parallel pipelines only
- **15°**: Reasonably parallel (default)
- **30°**: Lenient, includes diverging pipelines
- **45°**: Very lenient, may over-bundle

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for GIS professionals in the energy sector
- Uses pyproj for accurate geodesic calculations
- Spatial analysis powered by SciPy
- Modern GUI with CustomTkinter
- Cross-platform compatibility via PyInstaller

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the parameter guide for optimization

---

**Version**: 3.0.0  
**Last Updated**: 2024  
**Status**: Production Ready