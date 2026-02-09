# Pipeline Calculator v3: Feature Brainstorm (High Impact to Low Impact)

This document is a forward-looking backlog of potential features and improvements for Pipeline Calculator v3.

Notes:
- This is intentionally detailed so future implementation work can start with clear direction.
- All paths below assume the refactored modular package layout under `src/pipeline_calculator/`.
- Wherever possible, new work should be implemented in the modular package and only optionally backported to the legacy monolith (`src/pipeline_calculator_v3.py`) for fallback parity.
- Dependency hygiene:
  - If an idea requires a new third-party library, that section should explicitly call it out and we should add it to `requirements.txt` (runtime) or `requirements-dev.txt` (dev/test only) as appropriate.

---

## 1. Batch Processing (Multi-File Sessions + Folder Import + Consolidated Results)

### Why This Matters (User Value)
Right now the app is single-file at a time. In real usage, teams often need to run the same analysis on many KMZ/KMLs (different AOIs, clients, dates, corridors). Batch processing:
- Saves repeated manual steps (import, wait, export, repeat).
- Enables consistent parameter usage across many files.
- Enables a “session” view where multiple runs are comparable and exportable.
- Makes the tool more “production” than “demo”.

### Proposed UX (GUI)
Add a Batch/Session mode that supports:
1. Drag-and-drop multiple KMZ/KMLs at once.
2. Drag-and-drop a folder (optionally recurse) to ingest all `.kmz/.kml`.
3. Browse:
   - “Browse Files…” (multi-select)
   - “Browse Folder…” (pick directory)
4. Queue view:
   - Shows each file with status: `Queued`, `Running`, `Done`, `Failed`, `Canceled`.
   - Shows overall progress: `3/12 complete`, elapsed time, optional ETA.
5. Results navigation:
   - Left sidebar list of processed files (and their status).
   - Selecting an item shows the same results tabs (Summary, Pipelines, Overlap, Placemarks) for that file.
   - A “Batch Summary” view shows totals across all runs.
6. Exports:
   - “Export Current” (existing behavior).
   - “Export All” (writes per-file XLSX/JSON to a chosen output directory).
   - “Export Batch Summary” (single workbook/CSV summarizing all runs).

### Data Model (Needed to Support a Session)
Introduce explicit models so the GUI isn’t juggling raw dicts and file paths:

Proposed new modules:
```text
src/pipeline_calculator/batch/
  __init__.py
  models.py              # dataclasses/enums: BatchSession, AnalysisRun, RunStatus
  discovery.py           # gather files from folder, parse drag/drop payloads
  export.py              # consolidated batch export helpers
```

Example (skeleton) model types:
```python
# src/pipeline_calculator/batch/models.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import datetime as dt

class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class AnalysisRun:
    id: str
    input_path: str
    params: dict[str, float]
    status: RunStatus = RunStatus.QUEUED
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

@dataclass
class BatchSession:
    runs: list[AnalysisRun] = field(default_factory=list)
    selected_run_id: str | None = None
```

### Controller Changes (Queue + Concurrency + Cancellation)
Right now `src/pipeline_calculator/gui/controllers/analysis_controller.py` runs exactly one job on a background thread.

For batch, add a new controller that manages a queue and can optionally use:
- A single worker thread (simplest, safest memory profile).
- Later: a process pool for true parallelism (`concurrent.futures.ProcessPoolExecutor`).

Proposed new controller:
```text
src/pipeline_calculator/gui/controllers/
  batch_controller.py
```

Skeleton:
```python
# src/pipeline_calculator/gui/controllers/batch_controller.py
from dataclasses import dataclass
import queue
import threading
from pipeline_calculator.gui.state import AnalysisParameters

@dataclass
class BatchCallbacks:
    on_run_update: callable  # called on state transitions
    on_session_update: callable

class BatchController:
    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def enqueue(self, paths: list[str]) -> None: ...
    def start(self) -> None: ...
    def cancel(self) -> None: ...
```

### GUI Structure Changes (Reuse Existing Tabs)
The key to not exploding complexity: reuse `gui/tabs/*` for per-run results, and only change the “page” layer.

Proposed page modules:
```text
src/pipeline_calculator/gui/pages/
  file_select_page.py         # extend: multi-file + folder input
  session_page.py             # new: queue view + sidebar selection
  results_page.py             # keep: single-run view (can be embedded inside session page)
```

Core wiring changes:
- `src/pipeline_calculator/gui/main_window.py`
  - Add ability to enter session mode when multiple items are imported.
  - Add a `SessionState` to `AppState` (or a separate `SessionState` dataclass).

### Tests (Minimum Needed)
Add unit tests focused on non-GUI logic:
```text
tests/test_batch_discovery.py      # folder scan, multi-drop parsing, filtering
tests/test_batch_models.py         # run state transitions, serialization (if added)
tests/test_batch_controller.py     # queue progression with a stub analyzer
```

Optional: Add one “smoke-only” GUI import test that only imports session modules (no Tk root creation).

### Risks / Notes
- Drag/drop payload format differs across OSes (Windows quoting vs macOS brace-delimited).
- Folder recursion needs guardrails (avoid scanning huge trees by default).
- Parallel analysis (process pool) can increase memory usage significantly due to numpy/scipy and per-file intermediate state.

---

## 2. Session Persistence (Save/Load Analysis Projects)

### Why This Matters
Batch sessions become much more useful if a user can:
- Save a completed session (inputs, parameters, results, timestamps).
- Re-open later without rerunning analysis.
- Share results internally (without shipping raw KMZ/KML).

### Proposed UX
Add “Project” actions:
- `File -> Save Session…` writes a `.pcv3.json` (or `.pcv3.zip`) file.
- `File -> Open Session…` restores UI state with completed runs.
- “Re-run missing/failed runs” option.

### Implementation Sketch
New module(s):
```text
src/pipeline_calculator/session/
  __init__.py
  io.py            # save/load models, schema versioning
  schema.py        # explicit versioned schema definition
```

Key goals:
- Versioned schema (so future code can migrate older session files).
- Store enough metadata to render results even if inputs move.
- If desired, optionally store a copy of the input KMZ/KML inside a `.zip` session package.

Example session format:
```json
{
  "schema_version": 1,
  "app_version": "3.0.0-fixed",
  "created_at": "2026-02-09T20:00:00Z",
  "runs": [
    {
      "id": "run_001",
      "input_path": "/path/to/file.kmz",
      "params": { "detection_range": 15, "segment_length": 5, "...": 0 },
      "status": "done",
      "result": { "...": "same shape as current_results" }
    }
  ]
}
```

### Tests
```text
tests/test_session_io.py   # round-trip save/load, schema_version gating
```

### Risks / Notes
- Results dicts may contain numpy types; ensure JSON serialization uses plain python types.
- Consider file size: large sessions may be big; provide `.zip` option and compression.

---

## 3. Headless CLI Mode (Batch Automation Without GUI)

### Why This Matters
GUI is great for interactive use, but a CLI unlocks:
- Automated batch processing (CI, nightly builds, internal pipelines).
- Fast validation without building DMGs/EXEs.
- Easier regression testing for the core analyzer.

### Proposed CLI Commands
Examples:
- `pipeline-calculator analyze input.kmz --out outdir/ --xlsx --json`
- `pipeline-calculator analyze ./folder --recursive --out outdir/ --xlsx`
- `pipeline-calculator summary ./outdir --format csv`

### Implementation Sketch
Add a CLI module:
```text
src/pipeline_calculator/cli.py
```

Optionally add a separate entrypoint (so GUI stays default for `python -m pipeline_calculator`):
```text
src/pipeline_calculator/__main__.py     # route based on argv: no args -> GUI, args -> CLI
```

Keep dependencies minimal:
- Use `argparse` (stdlib).
- Use existing `PipelineAnalyzer` and existing export module(s).

### Tests
```text
tests/test_cli.py   # run CLI functions directly (not subprocess) for speed/stability
```

### Risks / Notes
- If we later remove heavy deps (scipy/pandas), CLI becomes even more attractive for automation.

---

## 4. Correctness + Robustness in Spatial Math (Projected Indexing, No False Negatives)

### Why This Matters
Current neighbor search uses a KDTree built on raw `(lon, lat)` degrees with a single `meters -> degrees` conversion (`meters / 111000`).
This can produce **false negatives** at higher latitudes because longitude degrees shrink by `cos(latitude)`.

If we miss candidate neighbors in the spatial index query, we can miss overlaps entirely.

### Proposed Improvements
1. Build spatial indexes in meters, not degrees.
2. Fix segmentation midpoints to be consistently placed along the polyline (avoid accumulating interpolation drift).
3. Make overlap/effective-length computations deterministic and explicitly validated.

### Implementation Sketch
Add a shared “projection and indexing” module:
```text
src/pipeline_calculator/core/
  spatial.py                # lon/lat <-> local xy conversion helpers
  spatial_index.py          # NeighborIndex interface + implementations
```

Recommended approach:
- Compute a local origin `(lon0, lat0)` per file (or per bundled section) using mean/median of all points.
- Convert to local meters using an equirectangular approximation:
  - `x = (lon - lon0) * 111320 * cos(lat0)`
  - `y = (lat - lat0) * 111320`
- Build the KDTree on `(x, y)` in meters.
- Use `detection_range` directly for radius queries.

This eliminates latitude scaling issues without requiring a heavy projection choice.

### SciPy Dependency Strategy (Optional)
We can define an interface:
```python
class NeighborIndex:
    def query_radius(self, x: float, y: float, radius_m: float) -> list[int]: ...
```

Implementations:
- `ScipyKDTreeIndex` (fast, if scipy installed)
- `GridIndex` (numpy-only, spatial hashing, good-enough and lightweight)

This gives us a path to eventually make scipy optional and shrink packaged artifacts.

### Segmentation Accuracy
Current segmentation can be improved by re-walking each polyline edge and recomputing distance for the remaining portion after each “cut point”.

Proposed update:
- Update `src/pipeline_calculator/core/segmentation.py` to compute correct intermediate points even when a single KML vertex-to-vertex segment spans many analysis segments.
- Add tests that validate midpoint monotonicity and approximate spacing.

### Tests
```text
tests/test_spatial_index_projection.py     # high-latitude false-negative guard
tests/test_segmentation_midpoints.py       # validate geometry, not just count
tests/test_overlap_against_bruteforce.py   # small synthetic datasets
```

### Risks / Notes
- Equirectangular projection is an approximation, but for local scales (15m detection range, 5m segments) it is more than sufficient and avoids missing neighbors.
- If future requirements include very large extents, consider UTM zone selection per file.

---

## 5. Dependency Reduction + Smaller Builds (Remove SciPy/Pandas Where Possible)

### Why This Matters
Packaged apps become:
- Smaller
- Faster to build
- Less brittle for code signing and notarization
…when they include fewer compiled dependencies.

Right now:
- `scipy` is included primarily for `KDTree`.
- `pandas` appears to be legacy-only (modular export uses openpyxl).
- `pytest` is in runtime requirements (likely accidental), increasing build surface area.

### Proposed Changes
1. Make `scipy` optional:
   - Use it when available.
   - Fall back to a numpy-only grid index.
2. Remove `pandas` from runtime if modular code no longer needs it.
3. Move `pytest` (and other dev-only deps) to dev requirements only.

### Implementation Sketch
Files likely to change:
```text
requirements.txt               # become runtime-only
requirements-dev.txt           # dev/test-only
src/pipeline_calculator/core/spatial_index.py
src/pipeline_calculator/core/overlap.py
src/pipeline_calculator/core/effective_length.py
```

### Tests
Run test suite in two modes:
1. With scipy installed.
2. Without scipy installed (CI matrix job) to ensure fallback index works.

### Risks / Notes
If we remove scipy, we must ensure performance remains acceptable on large datasets. The grid index approach should be benchmarked.

---

## 6. Cancelable Analysis + Real Progress Reporting (Better UX for Long Runs)

### Why This Matters
With batch processing, long analyses are inevitable. Users need:
- Confidence the app is working (progress).
- The ability to cancel a run or an entire batch.

### Proposed UX
Add:
- Progress bar with percent complete (not only indeterminate).
- “Cancel” button during analysis.
- Batch: “Cancel current”, “Cancel all”.

### Implementation Sketch
Introduce:
- A `CancelToken` (thread-safe event).
- Progress callbacks that report stage + fraction.

Proposed modules:
```text
src/pipeline_calculator/core/progress.py        # ProgressEvent model
src/pipeline_calculator/gui/controllers/analysis_controller.py   # accept cancel + progress
src/pipeline_calculator/core/analyzer.py        # pass progress/cancel through subcalls
```

Define progress events like:
- Parsing (0-10%)
- Segmenting (10-40%)
- Neighbor search (40-70%)
- Corridor stats (70-90%)
- Finalize (90-100%)

### Tests
```text
tests/test_cancel_token.py
tests/test_progress_reporting.py
```

### Risks / Notes
- Cancellation is easiest if loops periodically check the token.
- Some operations (KDTree construction) are not easily cancellable mid-call; cancellation can be “best effort”.

---

## 7. GUI Improvements (Tables, Search/Filter, Overlap UX, Parameter Presets, Help)

### Where the GUI Is Currently Thin
The modular GUI is functional but minimal:
- No file history or session concept in UI yet.
- Pipelines/Placemarks are basic tables without sorting/search.
- Overlap tab shows top 20 only (good default, but users will want to explore more).
- No “explain what this parameter means” help beyond short labels.

### High-Value GUI Enhancements
1. Multi-select browse:
   - Update `filedialog.askopenfilename` -> `askopenfilenames` for batch.
2. Sorting/search:
   - Add a search box on Pipelines/Placemarks tabs.
   - Add column sorting on Treeview headers (common pattern).
3. Overlap tab improvements:
   - Toggle: top 20 vs all sections.
   - Minimum length filter slider.
   - Search pipeline pair by name.
4. Parameter presets:
   - “Default”, “Conservative”, “Aggressive” presets.
   - Add tooltips explaining detection range, segment length, angular tolerance tradeoffs.
5. Better error messages:
   - When parsing fails, show actionable hints (invalid KMZ, missing KML, empty LineStrings).
6. “Open Output Folder” after export.

### Implementation Sketch (Paths)
```text
src/pipeline_calculator/gui/widgets/
  __init__.py
  sortable_treeview.py         # sorting helper wrapper
  search_bar.py
src/pipeline_calculator/gui/tabs/
  pipelines_tab.py             # add search + sorting
  placemarks_tab.py            # add search + sorting
  overlap_tab.py               # add filter/search + all sections toggle
src/pipeline_calculator/gui/pages/file_select_page.py   # multi-drop, folder import, presets UI
```

### Tests
GUI is hard to unit test end-to-end, but we can still test:
- Sorting helper functions (pure logic).
- Filtering logic (pure logic).

---

## 8. Built-In Visualization (Map Preview of Pipelines + Overlap Corridors)

### Why This Matters
Users currently rely on exporting KML and opening Google Earth to “see” the geometry.
An in-app preview would:
- Reduce context switching.
- Make the tool feel more complete.
- Enable quick QA (did we parse the right thing?).

### Options (Tradeoffs)
1. Minimal: plot preview using `matplotlib` in a Tk canvas.
   - Pros: simple, pure-python.
   - Cons: not truly interactive mapping, big dependency.
2. Web-based: generate an HTML map via `folium` and open in browser.
   - Pros: great UX, interactive.
   - Cons: still external, but faster than Google Earth.
3. Embedded map widget: `tkintermapview` (or similar).
   - Pros: interactive inside app.
   - Cons: more dependencies, packaging complexity.

### Dependencies / Requirements (If Implemented)
- Option 1: add `matplotlib` to `requirements.txt` (runtime); expect larger builds.
- Option 2: add `folium` to `requirements.txt` (runtime); consider keeping it optional because it increases build size.
- Option 3: add `tkintermapview` (and its transitive deps) to `requirements.txt` (runtime); packaging complexity likely increases.

### Implementation Sketch
```text
src/pipeline_calculator/gui/tabs/map_tab.py    # new tab
src/pipeline_calculator/geo/geojson.py         # convert results to geojson features
```

Start with the browser-based approach (lowest risk):
- Add “Preview Map” button that generates an HTML file and opens it.

---

## 9. Export Improvements (Batch Reports, GeoJSON, KMZ Outputs, PDF Summary)

### Why This Matters
Excel is good, but many workflows want:
- GeoJSON for GIS tooling.
- A combined KMZ of corridor polygons (one file, not many temp KMLs).
- A PDF summary for sharing.

### Proposed Export Targets
1. GeoJSON:
   - Pipelines as LineString features.
   - Corridors as Polygon features.
   - Placemarks as Point features.
2. KMZ export:
   - One KML containing all corridor sections.
   - Zip to KMZ with sensible naming.
3. Consolidated XLSX:
   - Add a “Batch Summary” sheet.
   - Add a “Parameters” sheet (audit trail).
4. Optional PDF summary:
   - Simple charts: total miles, effective miles, savings.

### Dependencies / Requirements (If Implemented)
- GeoJSON/KMZ: can be implemented with stdlib + existing deps (no new requirements expected).
- PDF:
  - If we adopt `reportlab`, add it to `requirements.txt` (runtime).
  - If we generate charts via `matplotlib`, add `matplotlib` to `requirements.txt` (runtime).

### Implementation Sketch
```text
src/pipeline_calculator/export/
  geojson.py
  corridor_kml.py              # extend: “all corridors” document builder
  report_pdf.py                # optional (if we adopt reportlab/matplotlib)
src/pipeline_calculator/gui/actions/export_actions.py
```

### Tests
```text
tests/test_export_geojson.py
tests/test_export_corridor_kmz.py
```

---

## 10. Parsing Improvements (Multi-KML KMZs, MultiGeometry, Better Metadata)

### Why This Matters
Real KMZ files can contain:
- Multiple KML files inside the archive.
- MultiGeometry placemarks.
- Nested folders and styles.
- Multiple LineStrings per pipeline.

### Proposed Parser Upgrades
1. KMZ: pick the “best” KML:
   - Prefer `doc.kml` if present.
   - Otherwise choose the largest `.kml`.
   - Optionally: allow user to choose.
2. MultiGeometry:
   - Support multiple LineStrings in a single placemark (combine or split with suffixes).
3. Extract more metadata:
   - Capture ExtendedData fields (if useful) for export and display.

### Implementation Sketch
Files:
```text
src/pipeline_calculator/parsers/kml_kmz.py
tests/test_kmz_multi_kml_selection.py
tests/test_kml_multigeometry.py
```

---

## 11. Multi-Way Bundling Model (Beyond Pairwise Sections)

### Why This Matters
The current UI presents bundled sections as pairwise `(pipeline_1, pipeline_2)` sections.
But real corridors can have 3+ pipelines running together.

We already compute “effective total” using a clustering concept, but the UI and exports could be upgraded to represent multi-way bundles directly.

### Proposed Model
Represent a corridor section as:
- `pipelines: list[str]`
- `centerline` geometry
- `width_m`
- `length_m`
- `participation_count` (k)

### Implementation Sketch
```text
src/pipeline_calculator/core/bundles.py         # compute multi-way bundles
src/pipeline_calculator/export/xlsx.py          # new sheet: "Bundles (multi-way)"
src/pipeline_calculator/gui/tabs/bundles_tab.py # new tab
```

### Tests
Synthetic cases:
- 2 pipelines parallel (k=2)
- 3 pipelines parallel (k=3)
- branching sections (k changes along the route)

---

## 12. Data Quality + Diagnostics Tab (Warnings, Outliers, “What Went Wrong”)

### Why This Matters
When results look wrong, users need visibility into:
- Did we parse the expected number of pipelines?
- Were coordinates dropped (out of range)?
- Did overlap analysis fail and fall back to None?

### Proposed UX
Add a “Diagnostics” tab that shows:
- Pipeline count, point count, coordinate ranges.
- Any parser warnings.
- Runtime warnings (overlap failures).
- Environment info (version, platform, build type).
- Buttons:
  - “Copy diagnostics to clipboard”
  - “Export diagnostics bundle” (zip logs + session JSON)

### Implementation Sketch
```text
src/pipeline_calculator/util/logging.py
src/pipeline_calculator/gui/tabs/diagnostics_tab.py
```

---

## 13. Regression Suite + Performance Benchmarks (Protect Against Future Drift)

### Why This Matters
As we improve math and performance, we must ensure:
- No silent correctness regressions.
- Performance doesn’t degrade unexpectedly.

### Proposed Work
1. Golden result tests:
   - Store expected results snapshots for a few representative KMZs (or synthetic fixtures).
2. Benchmarks:
   - Measure analysis runtime on synthetic datasets of size N.
   - Track memory usage if possible (best-effort).

### Implementation Sketch
```text
tests/golden/
  small_case.json
  medium_case.json
tests/test_golden_results.py
tests/test_performance_smoke.py
```

---

## 14. Preferences / Settings (Defaults, Recent Files, Theme, Units)

### Why This Matters
Users shouldn’t have to re-enter parameters every run.

### Proposed UX
Add a Preferences dialog:
- Default parameters
- Units (survey miles vs statute miles)
- Theme (dark/light/system)
- Recent files list (and clear history)

### Implementation Sketch
```text
src/pipeline_calculator/util/config.py          # load/save settings
src/pipeline_calculator/gui/dialogs/prefs_dialog.py
src/pipeline_calculator/gui/state.py           # store config in AppState
```

---

## 15. Distribution Enhancements (Auto-Update, Signed Windows, Universal Builds)

### Why This Matters
Once pilots use the tool, updates become a workflow problem.

### Potential Improvements
1. macOS auto-updater (future):
   - Sparkle is common for mac apps, but Python + PyInstaller integration is non-trivial.
2. Windows code signing:
   - Signed EXEs reduce SmartScreen friction.
3. Dual-arch mac builds:
   - Provide arm64 + x86_64 builds (or universal2) if Intel users exist.
4. Size reduction:
   - Strongly improved if SciPy/Pandas/pytest are reduced or removed.

### Implementation Sketch
This is more “ops and release engineering” than app code:
```text
.github/workflows/build.yaml
scripts/ci/*
scripts/macos/*
scripts/windows/*
```

---

## Appendix: Implementation Principles (So This Stays Maintainable)

1. Keep core logic GUI-free.
   - All math/parsing/export should be importable and unit-tested without Tk.
2. Add structured models (dataclasses) at boundaries.
   - Avoid “big dicts everywhere” once we add sessions/batch.
3. Every new feature should ship with tests for the non-UI logic.
4. Prefer incremental refactors:
   - Keep `PIPELINE_CALCULATOR_BUILD_IMPL=legacy` as a fallback while large changes land.
