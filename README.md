# Pipeline Calculator v5.0 - With Overlap Analysis

A comprehensive GUI application for calculating pipeline lengths and analyzing overlaps from KMZ/KML files. Designed for GIS professionals and aerial survey planning to optimize flight paths by identifying bundled pipeline sections.

## 🚀 Key Features

## Automatic build versions

The first commit introducing `MAJOR = 5` on a branch's first-parent history
establishes version **5.0**. A normal merge introducing v5 into main starts main
at 5.0 even if the feature branch had multiple preview commits. Later main
first-parent commits advance to 5.1, 5.2, and so on; rebuilding does not increment.
Use a normal or squash merge for the first 5.0 release. Fast-forward/rebase merges
retain the feature branch's counter and can introduce several increments.

The application title shows the clean numeric version. Run details, Excel
Analysis Details, JSON results, repair provenance and embedded metadata retain
the full build identifier. Branch/PR artifact filenames retain the
`5.N-dev.<commit>` suffix; uncommitted changes add `.dirty`. An uncommitted major
bump starts a 5.0 preview. CI main/tag builds reject dirty trees. Detached releases
must be on origin/main's first-parent history; tags must exactly match the version.
Full Git history is required. Do not rewrite main or reuse published tags.

Windows downloads use `Pipeline_Calculator_v5.N.exe`; macOS downloads use
`Pipeline_Calculator_v5.N_arm64.dmg` (Apple Silicon), including preview suffixes when applicable.
CI builds on macOS 15 and verifies the same archived app on macOS 26 and 27.
Sales uses Apple Silicon; Intel is outside this rollout's verification scope. Local packaging without
`ARTIFACT_ARCH` retains the unsuffixed `.dmg` filename.
The macOS bundle is `Pipeline_Calculator_v5.app`, with a clean display name and
version metadata. Its bundle identifier and user settings locations are unchanged. DMG packaging reads the actual app's embedded version, preserving
it even after switching branches. Pass the exact DMG path to notarization.
The optional app download is a `.app.zip` archive that preserves executable
permissions. CI verifies offline analysis before and after Developer ID signing;
notarization and stapling remain a separate required step before macOS rollout.

Each Mac build inventories its bundled native libraries and records their actual
minimum macOS version in the app's `LSMinimumSystemVersion` and the CI test report.
Setting `MACOSX_DEPLOYMENT_TARGET` alone cannot make newer prebuilt libraries run
on an older OS. The macOS 27 runner currently uses GitHub's `xcode-27` preview
label; verification asserts its actual OS major version and retains its OS build
number. A newer SDK alone does not establish runtime compatibility.

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
4. Tag the exact clean main commit as `v5.N`, matching the version command's output,
   then push the tag. CI rejects mismatched tags and tags off main's first-parent
   history. Never move/reuse published tags. Existing main/PR/tag/manual triggers
   remain in effect; this does not add CI builds on every feature-branch push.

Future major releases change `MAJOR` in `src/pipeline_calculator/versioning.py`.
Update the stable major-specific macOS bundle name and its packaging checks too.
The introducing commit automatically establishes the new baseline, including when
merged into main. Generated metadata stays in ignored `build/`. Installed apps
need no Git. Source runs without metadata/history display 5.0 and retain
`5.0-dev.unknown` in build details; packaging fails rather than shipping that fallback.
For a given clean commit/context, the version is deterministic.

References: [Git first-parent traversal](https://git-scm.com/docs/git-rev-list)
and [PyInstaller bundled data](https://pyinstaller.org/en/stable/runtime-information.html#using-file).

### New in v5.0
- State mileage breakdowns and scoped overlaps.
- Verified geometry-preserving input repair and provenance.
- Improved corridor maps, exports, and responsive result views.
- Verified Windows and Apple Silicon builds, including macOS 27.

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

### State breakdown

Enable **State breakdown** in Analysis Settings before browsing or dropping a
KML/KMZ. The switch starts off and remembers your choice across app sessions.
**Adjust Parameters** also lets you change it before reanalysis.

The app retains its combined analysis, clips pipeline paths at state boundaries,
and independently analyzes each state's interior geometry. **View: Combined**
switches the existing cards, pipeline table, overlaps and corridor previews to an
encountered state. The Combined summary compares all states without adding a card
for each one. Point placemarks remain available in the Combined view.

The View selector sits on the left of the results tab row without increasing
its height. Narrow windows show a section menu beside it; the helper text hides
when space is limited.

Original state mileage reconciles to the original input mileage. Positive-length
lines following a verified shared border are stored once and their mileage is
allocated equally to adjoining states, with no state overlap discount. Outside
coverage and unresolved mileage are shown separately. Short genuine crossings are
preserved. State savings can differ from combined savings because overlap minimum
lengths apply independently inside each state; the boundary split can make a
previously qualifying overlap too short.

Exports create one named package containing `analysis.xlsx`, optional JSON and
optional `Combined/analysis.kmz` plus `States/<State>/analysis.kmz`. Maps are
selected by default. State maps contain exclusive interior geometry, while shared
border geometry appears once in the Combined map. Therefore state map line mileage
matches the workbook's **Interior mileage**, while **Original attributed mileage**
also includes any shared allocation. Export always includes the whole result,
regardless of the selected state view.

The bundled [2025 Census TIGER/Line state data](https://www2.census.gov/geo/tiger/TIGER2025/STATE/)
covers the 50 states and Washington, DC without network access. Boundary provenance,
datum operations and checksums appear in the export. Numerical clipping precision
is separate from the source data's positional accuracy. Thirteen remote Alaska/
Hawaii components outside the published datum-operation areas use explicitly
approximate coordinate equivalence with unknown positional accuracy, recorded
per component; results do not establish surveyed ownership. Boundary updates
require a reviewed resource rebuild using `scripts/data/prepare_state_boundaries.py`
and the original source archive; no runtime downloads occur.

See the [approved specification](STATE_BOUNDARY_ANALYSIS_PLAN.md) and
[implementation validation](docs/validation/state-boundary-analysis.md).

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
- **Windows**: `Pipeline_Calculator_v5.N.exe`
- **Apple Silicon Mac**: `Pipeline_Calculator_v5.N_arm64.dmg`

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
   git tag v5.N  # Replace N with the generated version; see Automatic build versions
   git push origin v5.N
   ```

The GitHub Actions workflow builds Windows and Apple Silicon macOS artifacts on
main pushes, pull requests, and tags. The same Mac app must pass native execution
on Sequoia (15), Tahoe (26), and Golden Gate (27). A tag creates a **draft** release.
Before publishing it, notarize and staple the signed DMG with
`scripts/macos/notarize_dmg.sh`, replace the draft's DMG asset with that verified
file, and check its launch on the target Macs. This keeps pre-notarized builds
from being automatically offered as the latest sales release.

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
at endpoints and bends. Corridor polygons use local geodesic coordinates,
including across the dateline.

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

The Pipelines table and its spreadsheet export show **Placemark ID**, taken directly
from each source KML `<Placemark id="…">` attribute. Missing or blank attributes
display `N/A`; OBJECTID, geometry IDs and generated numbers are not substituted.
These source identifiers stay attached to their pipelines when sorting and do not
replace the internal indices used for overlap calculations.

Both GUIs show the current stage, available work counts and elapsed processing time.
A green percentage bar advances from completed work within each processing stage;
the stages have fixed shares of the overall workflow, so this is not a time-remaining
estimate. The clock starts in the processing worker, excludes time waiting for a
warning decision, and resets with the bar for each new analysis. The yellow **Cancel**
button below the bar
requests a cooperative stop; the app stays busy until the worker acknowledges it.
A single XML, filesystem or numerical-library call may finish before cancellation
is observed. Cancelled work never becomes a result or an analysis-error dialog.
Use **Retry selected file** after cancellation, or browse for another input.
Results are cleared when starting a new analysis.

Jobs can pause for **Continue anyway** or **Cancel** when measured throughput projects
total processing beyond **60 seconds**, or processing has already exceeded that time.
The projection includes elapsed time plus unfinished work in the current stage.
It needs at least three seconds of stage measurements and a sustained estimate
over one second with advancing work counts; segment count alone does not trigger
the warning. The warning can appear during processing, since speeds for later
stages cannot be known in advance. It is an estimate, not a completion-time promise.
Accepting it suppresses further runtime warnings for that run. Warning text scrolls
while Continue/Cancel remain accessible. Continuing does not override hard limits;
source mileage remains available with an incomplete-analysis notice if overlap
exceeds a limit. No automatic geometry simplification changes source distances.

**View Corridor** prepares KML and requests opening without blocking the main window.
If opening fails, the dialog retains the generated file and offers **Copy Path**,
**Save As** and **Retry**. An accepted opening request does not confirm that Google
Earth rendered the file. Temporary files remain available after closing the dialog;
use Save As for a lasting copy because the operating system may clean temp storage.

Corridor maps follow the path portions that qualified for overlap, with **5 m
padding** and rounded ends and bends. They preserve separate pieces and open
centers. Padding is independent of Detection Range and does not change pipeline
mileage, overlap qualification or savings. These are approximate display areas,
not surveyed boundaries or rights-of-way. State maps are clipped to their state.

The completed analysis prepares each map before displaying results. If a map
cannot be constructed and verified within the accuracy and resource limits, its
row shows **Map unavailable** and Diagnostics explains why. Mileage and savings
remain available. New maps never substitute broad rectangles or silently drop
components. Previews and package exports preserve every polygon and hole, without
adding extra centerlines that could be counted as pipeline mileage on reimport.

See [corridor implementation and verification](docs/validation/corridor-buffer-implementation.md)
for the current geometry policy, comparisons, resource limits and sample exports.

See [automated improvement verification](docs/validation/automated-improvements.md),
[subsequent workload/corridor hardening](docs/validation/workload-corridor-hardening.md),
[R4/R5 audit and final verification](docs/validation/r4-r5-audit.md),
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
