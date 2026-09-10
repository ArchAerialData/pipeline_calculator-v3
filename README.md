# Pipeline Calculator v4.0 - With Overlap Analysis

A comprehensive GUI application for calculating pipeline lengths and analyzing overlaps from KMZ/KML files. Designed for GIS professionals and aerial survey planning to optimize flight paths by identifying bundled pipeline sections.

## 🚀 Key Features

## Automatic build versions

The app and package filenames share a Git-derived version. No version-bump commit
is created. Baseline `fc4cc05108dda7ae2f61be4a763eb34f5f1ebb0e` represents **4.0**.

- Each main first-parent commit after the baseline adds one: `4.1`, `4.2`, through
  `4.9`, then `4.10`. These are version components, not decimal numbers.
- Normal merge commits and squash merges count once per merged PR. Fast-forward
  and rebase merges may advance several numbers. Prefer normal merge commits or
  squash merges for one increment per PR.
- Pushing alone does not increment anything. Rebuilding a clean main commit gives
  the same version; pushing several main commits at once can skip build numbers.
- Branch/PR builds use `4.N-dev.<12-character-commit-hash>`. Their count follows
  their own first-parent history and does not reserve a future release number.
  Branches can share numeric prefixes; hashes distinguish commits. PR context
  always produces a preview, even for a commit also on main. Branches forked before
  the baseline stay previews; update from main before relying on the count.
- Uncommitted changes, including untracked files, add `.dirty`. Different dirty
  edits can share the same version: commit before sharing reproducible builds.
  CI main/tag builds reject dirty trees.
- Detached builds get release versions only on `origin/main` first-parent history.
  Fetch first so this reference is current; other detached commits are previews.

Windows downloads use `Pipeline_Calculator_v4.N.exe`; macOS downloads use
`Pipeline_Calculator_v4.N.dmg`, including preview suffixes when applicable.
The macOS bundle remains `Pipeline_Calculator.app`, with updated display name and
version metadata. DMG packaging reads the actual app's embedded version, preserving
it even after switching branches. Pass the exact DMG path to notarization.

### Building and publishing safely

1. Run `git fetch origin --prune --tags`. For shallow clones first run
   `git fetch --unshallow origin`. Builds reject shallow history, missing baselines,
   or missing required main references. CI now checks out full history.
2. Inspect the version with `python src/pipeline_calculator/versioning.py`.
   Build using the provided Windows/macOS scripts, which embed that version.
3. Review previews, merge into main, and distribute clean main builds. Protect main
   against force pushes/history rewrites, which can change or reuse version numbers.
   Do not move the baseline to renumber releases. Deleting merged branches does not
   affect numbering because main's history remains intact.
4. Tag the exact clean main commit as `v4.N`, matching the version command's output,
   then push the tag. CI rejects mismatched tags and tags off main's first-parent
   history. Never move/reuse published tags. Existing main/PR/tag/manual triggers
   remain in effect; this does not add CI builds on every feature-branch push.

Major versions remain intentional: to start 5.0, update `MAJOR` and `BASELINE` in
`src/pipeline_calculator/versioning.py`, along with tests and documentation.
Generated metadata stays in ignored `build/`; no tracked version file is rewritten.
Installed apps read bundled metadata and need no Git. Source runs without usable
Git/history show `4.0-dev.unknown`; packaging fails rather than shipping that fallback.
For a given clean commit and build context the version is deterministic.

References: [Git first-parent traversal](https://git-scm.com/docs/git-rev-list)
and [PyInstaller bundled data](https://pyinstaller.org/en/stable/runtime-information.html#using-file).

### New in v4.0
- **Parser hardening**: Supports multipart LineStrings, local KMZ NetworkLinks, and gx:Track/gx:MultiTrack paths
- **Diagnostics**: Reports skipped or unsupported KML/KMZ structures that may affect mileage
- **Centerline-only mileage**: Prevents Polygon/LinearRing outlines from being counted as pipeline centerlines

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
- Cooperative cancellation and named processing stages with work counts and elapsed time

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
- **Windows**: `Pipeline_Calculator_v4.N.exe`
- **macOS**: `Pipeline_Calculator_v4.N.dmg`

### Option 2: Run from Source
Builds and automated validation use Python 3.11. Use the platform setup scripts to prepare that environment.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pipeline-calculator-v4.git
   cd pipeline-calculator-v4
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python src/pipeline_calculator_entry.py
   ```

## 📝 Usage Guide

### Basic Workflow

1. **Launch the application**
   - Double-click the executable, or
   - Run `python src/pipeline_calculator_entry.py`

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
pipeline-calculator-v4/
├── .github/
│   └── workflows/
│       └── build.yaml          # GitHub Actions workflow
├── src/
│   └── pipeline_calculator_entry.py # Main modular application entrypoint
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
   git commit -m "Initial commit - Pipeline Calculator v4.0"
   git branch -M main
   git remote add origin https://github.com/yourusername/pipeline-calculator-v4.git
   git push -u origin main
   ```

3. **Enable GitHub Actions**:
   - Go to Settings → Actions → General
   - Select "Allow all actions and reusable workflows"

4. **Create a release** to trigger builds:
   ```bash
   git tag v4.N  # Replace N with the generated version; see Automatic build versions
   git push origin v4.N
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

- **Coordinate System**: GRS80 geodesic calculations
- **Distance Units**: US Survey Miles (1609.347218694 meters)
- **Performance**: Bounded processing with reproducible synthetic workload measurements; see [performance evidence](docs/validation/comparison.md)

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
   - Files show processing stages and can be cancelled cooperatively
   - Consider splitting very large KMZ files (>100MB)

4. **Incorrect overlap calculations**
   - Verify Angular Tolerance setting
   - Review the Diagnostics tab/export sheet for unsupported or skipped KML/KMZ structures

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

Only continuous, unique segment coverage meeting this minimum on both pipelines
qualifies for bundling **and** mileage savings. Separate coordinate paths do not
combine to meet the minimum. Pairwise bundled rows may describe the same shared
corridor; their sum is not the project's mileage-removed total.

Calculations use sampled segments. Segment spacing can affect overlap endpoints,
and trailing partial segments are retained in original mileage without a savings
discount. Savings now use deterministic, mutually compatible groups: every pair
in a group must satisfy the detection range and qualifying-section rules. A
segment can belong to only one group. For three 300 m lines spaced at 0, 10, and
20 m with a 15 m limit, two lines are bundled and the third remains separate:
approximately 600 m effective mileage from 900 m of pipeline. Group selection is
a conservative heuristic, not a flight-route optimization.

Nearby finite segment tangents are compared so that offset sampling positions do
not hide overlaps. Their endpoints must overlap longitudinally; lines merely
meeting end-to-end are not bundled. Segment length still controls approximation
at endpoints and bends. Corridor centers and polygons use local geodesic
coordinates, including across the dateline.

If a LineString or gx:Track contains an invalid coordinate, that geometry is
rejected rather than connecting across the missing vertex. Other valid geometries
are retained and the result is marked **incomplete**. Calculation failures likewise
show an incomplete notice, with unavailable savings rather than a misleading zero.
Check Diagnostics (also exported to Excel), repair the input, and rerun before
using incomplete results as project totals. Internal KMZ relative links are
normalized within the archive; links escaping its root are not followed.

Missing, malformed, or unsupported linked documents and empty inputs also mark
the analysis incomplete. To bound processing, input limits are 64 MiB decompressed
per KML, 256 MiB total parsed KML, 1,024 linked documents, and 10,000 ZIP entries.
Analysis allows at most 1,000,000 segments and 5,000,000 candidate inspections.
Exceeding a limit stops that computation explicitly; split large projects into
smaller inputs rather than treating failed analysis as zero overlap. Ambiguous
duplicate KML entry names are rejected. Excel exports preserve source names and
diagnostics as literal text rather than executable formulas.

The overlap tab displays 20 rows per page with Previous/Next controls, retaining
access to every section. Its pairwise total is explicitly distinguished from
mileage removed. Both GUI implementations share the summary and overlap tabs,
show corrected/clamped parameter values, and prevent simultaneous analysis jobs.

### Cancellation, progress and corridor recovery

Both GUIs show the current stage, available work counts and elapsed time. **Cancel**
requests a cooperative stop; the app stays busy until the worker acknowledges it.
A single XML, filesystem or numerical-library call may finish before cancellation
is observed. Cancelled work never becomes a result or an analysis-error dialog.
Use **Retry selected file** after cancellation, or browse for another input.
Results are cleared when starting a new analysis.

Exceptionally large or dense jobs can pause for **Continue anyway** or **Cancel**
before expensive overlap comparisons. Initial import and source-distance measurement
run first; density checks also require segmenting/indexing the paths. These checks
run off the UI thread and can be cancelled. The warning suggests splitting geometry
into smaller files or simplifying a copy where distance accuracy is preserved.
Ordinary jobs proceed directly. This is a workload estimate, not a runtime forecast.

Initial advisory thresholds are deliberately high: 750,000 estimated analysis
segments or 10,000,000 estimated neighbor inspections from up to 256 count-only
queries. The latter is twice the existing five-million-inspection safety cap.
Continuing does not override hard limits; source mileage remains available with
an incomplete-analysis notice if overlap exceeds a limit. Sampling may miss a
localized hotspot. No automatic geometry simplification changes source distances.

**View Corridor** prepares KML and requests opening without blocking the main window.
If opening fails, the dialog retains the generated file and offers **Copy Path**,
**Save As** and **Retry**. An accepted opening request does not confirm that Google
Earth rendered the file. Temporary files remain available after closing the dialog;
use Save As for a lasting copy because the operating system may clean temp storage.

Corridors are approximate visualizations of sampled paths, not surveyed boundaries.
KML descriptions identify rectangle fallbacks and invalid preferred geometry.
Non-finite, out-of-range, collapsed and unusable rings are rejected. All ring sizes
receive local-plane topology checks with a bounded edge-inspection budget; a shape
that exceeds that budget falls back to a disclosed simpler outline. Right-angle,
hairpin and loop examples can require broad rectangles enclosing the qualified
samples. End padding helps outlines show the ends of sampled sections. These
visual changes preserve original pipeline distance and sampled overlap/savings rules.

See [automated improvement verification](docs/validation/automated-improvements.md),
[subsequent workload/corridor hardening](docs/validation/workload-corridor-hardening.md),
[implementation status](IMPROVEMENT_ROADMAP.md) and the separate
[owner/platform follow-up runbook](FOLLOWUP_RUNBOOK.md).

See [calculation fix review and verification](CALCULATION_FIX_REVIEW.md) for the
regression cases and remaining validation limits.
See the [follow-up audit](FOLLOWUP_AUDIT.md) for subsequent numerical, input, export,
and GUI fixes.

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

**Version**: Git-derived 4.N (see Automatic build versions)
**Last Updated**: 2026
**Status**: Production Ready
