# Pipeline Calculator v3 Refactor Plan (Monolith to Package)

This repo currently implements almost all functionality in a single file:
`src/pipeline_calculator_v3.py` (about 2,000 lines).

Refactoring into a package layout similar to Purway Geotagger is feasible and, given the code size, will likely pay off in maintainability and testability. The main risk is churn: updating imports, build entrypoints, and resource paths (icons, PyInstaller) without breaking packaging.

This document describes:
1. The current structure of `src/pipeline_calculator_v3.py`.
2. A proposed package/module structure.
3. A file-by-file mapping of what moves where.
4. A staged migration plan designed to keep CI/build working throughout.

## Working Agreement (How We Execute This Refactor)

We will refactor **one phase at a time**, and we do not start the next phase until the current phase meets its exit criteria.

**Exit criteria for every phase**
- Add or update tests that cover the behavior changed in that phase.
- `pytest` passes locally (and in CI once pushed).
- Update this file:
  - Phase `Status`
  - What tests were added/updated
  - Any issues, follow-ups, or known gaps discovered

**Why this matters**
- Most refactor risk comes from import churn and packaging edge cases. Keeping phases small with tests reduces wasted time rebuilding DMGs/EXEs after accidental breakage.

## Status Tracker

| Phase | Scope | Status | Tests | Notes |
| --- | --- | --- | --- | --- |
| 1 | Package skeleton | DONE | `tests/test_refactor_fallback.py` | `pipeline_calculator.app` resolves to legacy by default |
| 2 | Export modules | DONE | `tests/test_export_workbook.py`, `tests/test_overlap_kml.py` | Implementations moved to `src/pipeline_calculator/export/`; legacy wrappers kept |
| 3 | Parsers | DONE | `tests/test_kml_parsing.py`, `tests/test_kmz_parsing.py` | Parsing moved to `src/pipeline_calculator/parsers/kml_kmz.py`; analyzer delegates |
| 4 | Core overlap + effective length | DONE | Existing overlap/effective tests | Core moved to `src/pipeline_calculator/core/`; legacy delegates |
| 5 | Core analyzer | DONE | `tests/test_analyze_complete.py`, `tests/test_core_analyzer.py` | Added `core/analyzer.py`; legacy delegates `analyze_complete` and length calc |
| 6 | GUI package | DONE | `tests/test_gui_state.py`, `tests/test_analysis_controller.py` | New modular GUI exists behind `PIPELINE_CALCULATOR_IMPL=new`; manual smoke passed (KMZ analyzed successfully) |
| 7 | Compatibility + build entrypoints | DONE | Packaged build smoke checklist | CI/build scripts now default to modular (`PIPELINE_CALCULATOR_BUILD_IMPL=new`); legacy build path retained |

## Current Structure (What’s In The Monolith)

Top-level constants and metadata:
- `__version__`, `__author__`
- Analysis parameters:
  - `DEFAULT_DETECTION_RANGE`
  - `MIN_PARALLEL_LENGTH`
  - `SEGMENT_LENGTH`
  - `ANGULAR_TOLERANCE`
  - `GAP_TOLERANCE`

**Core analysis engine**
- `class PipelineAnalyzer` (starts near line ~45)
- Responsibilities:
  - Parse KMZ/KML into pipeline coordinate arrays and point placemarks
  - Compute per-pipeline lengths (meters, survey miles)
  - Segment pipelines and detect parallel segments
  - Compute bundled corridor polygons and overlap summary statistics
  - Compute effective length via clustering (`compute_effective_length_by_clusters`)
  - Produce a final analysis result dict (`analyze_complete`)

**Export helpers (pure functions)**
- `build_analysis_workbook(current_results)` (starts near line ~903)
  - Builds an `openpyxl.Workbook` with the “Pipeline Length Analysis” and “Pipeline Overlap Analysis” sheets.
- `build_overlap_corridor_kml(section, index)` (starts near line ~1081)
  - Builds a KML document string for the corridor polygon.

**GUI application**
- `class PipelineCalculatorGUI` (starts near line ~1190)
- Responsibilities:
  - Tk/CustomTkinter window creation + layout
  - Drag-and-drop file input (tkinterdnd2)
  - Parameter dialog + re-analysis
  - Results tabs (Summary, Pipelines, Overlap, Placemarks)
  - Export (XLSX/JSON) and “open corridor KML” actions

**Entrypoint**
- `main()` (starts near line ~2043) and `if __name__ == "__main__": main()`

## Refactor Recommendation (Should We Do This?)

Refactor is recommended if we expect ongoing changes, bug fixes, or new features because:
- The monolith currently mixes I/O parsing, geometry/analysis, formatting/export, and GUI concerns.
- Having importable modules makes it easier to:
  - unit test non-GUI logic
  - reuse analysis logic from scripts/CI
  - keep GUI code readable
  - reduce “accidental coupling” between unrelated parts

Refactor is not mandatory if the app is feature-frozen and working, but the file is already large enough that it will keep getting harder to change safely without modularization.

## Proposed Package Layout (Purway-Style)

Create a new Python package under `src/`:

```text
src/
  pipeline_calculator/
    __init__.py
    __main__.py
    app.py
    core/
      __init__.py
      analyzer.py
      constants.py
      effective_length.py
      overlap.py
      segmentation.py
    parsers/
      __init__.py
      kml_kmz.py
    export/
      __init__.py
      xlsx.py
      corridor_kml.py
    gui/
      __init__.py
      main_window.py
      tabs.py
      resources.py
    util/
      __init__.py
      resources.py
      types.py
```

Keep `src/pipeline_calculator_v3.py` as a thin compatibility wrapper for a while:

```py
# src/pipeline_calculator_v3.py
from pipeline_calculator.app import main

if __name__ == "__main__":
    raise SystemExit(main())
```

This avoids breaking:
- existing build scripts (PyInstaller points at `src/pipeline_calculator_v3.py`)
- tests that currently import `pipeline_calculator_v3`
- any user docs that reference the script path

Once stable, CI/build scripts can be updated to target `python -m pipeline_calculator` or `src/pipeline_calculator/app.py` directly, and the wrapper can be removed.

## Module Mapping (What Moves Where)

### `src/pipeline_calculator/core/constants.py`
- `DEFAULT_DETECTION_RANGE`
- `MIN_PARALLEL_LENGTH`
- `SEGMENT_LENGTH`
- `ANGULAR_TOLERANCE`
- `GAP_TOLERANCE`
- Potentially `SURVEY_MILE_METERS = 1609.347218694`

### `src/pipeline_calculator/parsers/kml_kmz.py`
Move parsing logic out of the analyzer so it’s testable and reusable:
- `extract_features_from_file(...)`
- `_extract_objectid(...)`
- `_has_linestring(...)`
- `_has_point(...)`
- `_extract_coordinates(...)`

API suggestion:
- `parse_kml_kmz(path: str) -> tuple[list[Pipeline], list[Placemark]]`

### `src/pipeline_calculator/core/segmentation.py`
- `segment_pipeline(coordinates)`
- Any small math helpers for segmentation

### `src/pipeline_calculator/core/overlap.py`
Move overlap computation and polygon corridor building here:
- `find_parallel_segments(...)`
- `calculate_overlap_results(...)`
- Polygon/corridor helpers currently embedded in `calculate_overlap_results`

### `src/pipeline_calculator/core/effective_length.py`
- `compute_effective_length_by_clusters(...)`

### `src/pipeline_calculator/core/analyzer.py`
Make `PipelineAnalyzer` the orchestrator that composes parsers + core algorithms:
- `calculate_pipeline_lengths(...)`
- `analyze_complete(...)`

### `src/pipeline_calculator/export/xlsx.py`
- `build_analysis_workbook(current_results)`

### `src/pipeline_calculator/export/corridor_kml.py`
- `build_overlap_corridor_kml(section, index)`

### `src/pipeline_calculator/gui/main_window.py`
Move the Tk/CustomTkinter application class:
- `PipelineCalculatorGUI`

Optionally split tab construction to keep the window class readable:
- `src/pipeline_calculator/gui/tabs.py`
  - `create_summary_tab(...)`
  - `create_pipeline_tab(...)`
  - `create_overlap_tab(...)`
  - `create_placemark_tab(...)`

### `src/pipeline_calculator/gui/resources.py` and `src/pipeline_calculator/util/resources.py`
Centralize “where do I find icons/resources”:
- Detect frozen mode and `sys._MEIPASS`
- (Optional) switch to `importlib.resources` so resources can live under the package

### `src/pipeline_calculator/app.py`
Equivalent to Purway’s `purway_geotagger/app.py`:
- CLI entrypoint that instantiates GUI and runs main loop.
- Also the place to set any platform-specific tweaks before window creation.

### `src/pipeline_calculator/__main__.py`
Allow `python -m pipeline_calculator` as a dev entrypoint.

## Refactor Staging Plan (Low-Risk Steps)

### Phase 1: Package skeleton

- Status: DONE
- Scope:
  - Add `src/pipeline_calculator/` package with `app.py`, `__init__.py`, `__main__.py`, and placeholder subpackages.
  - Keep `src/pipeline_calculator_v3.py` as the stable implementation/build target.
- Tests:
  - Add a non-GUI test that verifies the package entrypoint resolves to legacy by default.
- Completion notes:
  - Completed; see `tests/test_refactor_fallback.py`.

### Phase 2: Export modules (XLSX + corridor KML)

- Status: DONE
- Scope:
  - Move `build_analysis_workbook` to `src/pipeline_calculator/export/xlsx.py`
  - Move `build_overlap_corridor_kml` to `src/pipeline_calculator/export/corridor_kml.py`
  - Keep legacy functions as wrappers for backward compatibility until the refactor is complete.
- Tests (must exist and pass before continuing):
  - Workbook structure + style tests
  - Corridor KML generation tests
- Completion notes:
  - Added:
    - `src/pipeline_calculator/export/xlsx.py`
    - `src/pipeline_calculator/export/corridor_kml.py`
  - Legacy wrappers remain in `src/pipeline_calculator_v3.py` to avoid breaking existing imports/entrypoints.
  - Tests now target the package modules directly:
    - `tests/test_export_workbook.py`
    - `tests/test_overlap_kml.py`

### Phase 3: Split parsing from analysis (KMZ/KML)

- Status: DONE
- Scope:
  - Move KMZ/KML parsing helpers to `src/pipeline_calculator/parsers/kml_kmz.py`
  - Update `PipelineAnalyzer` to call the parser module
- Tests (must exist and pass before continuing):
  - Unit tests for KML + KMZ parsing (synthetic fixtures)
  - Error-path tests (invalid KML/KMZ)
 - Completion notes:
   - Added `src/pipeline_calculator/parsers/kml_kmz.py` exporting `parse_kml_kmz(...)`.
   - Updated legacy `PipelineAnalyzer.extract_features_from_file(...)` to delegate to the parser module.
   - Updated tests to target the parser module directly and also validate legacy wrapper behavior:
     - `tests/test_kml_parsing.py`
     - `tests/test_kmz_parsing.py`

### Phase 4: Split overlap + effective-length algorithms

- Status: DONE
- Scope:
  - Move overlap detection and corridor polygon generation into `src/pipeline_calculator/core/overlap.py`
  - Move clustering effective length into `src/pipeline_calculator/core/effective_length.py`
- Tests (must exist and pass before continuing):
  - Parallel detection tests (synthetic)
  - Corridor polygon sanity tests
  - Effective length clustering tests (identical lines, non-overlapping lines)
 - Completion notes:
   - Added core modules:
     - `src/pipeline_calculator/core/segmentation.py`
     - `src/pipeline_calculator/core/overlap.py`
     - `src/pipeline_calculator/core/effective_length.py`
   - Updated legacy `PipelineAnalyzer` methods to delegate to the new core modules:
     - `segment_pipeline`
     - `find_parallel_segments`
     - `calculate_overlap_results`
     - `compute_effective_length_by_clusters`
   - Tests remain green:
     - `tests/test_segmentation.py`
     - `tests/test_parallel_overlap.py`
     - `tests/test_effective_length.py`

### Phase 5: Core analyzer orchestration

- Status: DONE
- Scope:
  - Make `PipelineAnalyzer` mostly an orchestrator in `src/pipeline_calculator/core/analyzer.py`
  - Keep public result schema stable (no behavior changes)
- Tests (must exist and pass before continuing):
  - `analyze_complete` tests (synthetic + real KMZ smoke)
 - Completion notes:
   - Added `src/pipeline_calculator/core/analyzer.py` containing a package `PipelineAnalyzer` that orchestrates:
     - parsing (`pipeline_calculator.parsers.kml_kmz`)
     - overlap (`pipeline_calculator.core.overlap`)
     - effective length (`pipeline_calculator.core.effective_length`)
   - Added `src/pipeline_calculator/core/constants.py` for shared defaults.
   - Updated legacy `PipelineAnalyzer` to delegate:
     - `calculate_pipeline_lengths`
     - `analyze_complete`
   - Tests:
     - Existing `tests/test_analyze_complete.py` remains green.
     - Added `tests/test_core_analyzer.py` to exercise the package analyzer directly.

### Phase 6: GUI package move

- Status: DONE
- Scope:
  - Move `PipelineCalculatorGUI` into `src/pipeline_calculator/gui/main_window.py`
  - Split each results tab into its own module so adding future tabs/windows is low-friction
  - Split non-UI concerns (export, open KML, analysis threading) into testable helper modules
- Proposed folder structure (Tk/CustomTkinter version):

```text
src/pipeline_calculator/gui/
  __init__.py
  main_window.py              # owns root window + page switching + wires controller/state
  state.py                    # dataclasses for app state (no tkinter imports)
  resources.py                # icon/resource path resolution (sys._MEIPASS, etc.)
  controllers/
    __init__.py
    analysis_controller.py    # runs core analyzer on a background thread; publishes progress/events
  pages/
    __init__.py
    file_select_page.py       # drag/drop + browse page (select KMZ/KML)
    results_page.py           # tab container + top summary + navigation
  tabs/
    __init__.py
    summary_tab.py            # Summary tab UI builder + update hooks
    pipelines_tab.py          # Pipelines tab UI builder + update hooks
    overlap_tab.py            # Overlap tab UI builder + update hooks
    placemarks_tab.py         # Placemarks tab UI builder + update hooks
  dialogs/
    __init__.py
    params_dialog.py          # parameter edit dialog (detection range, etc.)
  actions/
    __init__.py
    export_actions.py         # XLSX/JSON export (uses pipeline_calculator.export.*)
    open_kml_action.py        # build temp KML + open in OS (uses export/corridor_kml)
```

- Mapping from legacy monolith methods to new modules:
  - `PipelineCalculatorGUI.show_file_selection()` -> `gui/pages/file_select_page.py`
  - `PipelineCalculatorGUI.show_results()` -> `gui/pages/results_page.py`
  - `PipelineCalculatorGUI.create_summary_tab()` -> `gui/tabs/summary_tab.py`
  - `PipelineCalculatorGUI.create_pipeline_tab()` -> `gui/tabs/pipelines_tab.py`
  - `PipelineCalculatorGUI.create_overlap_tab()` -> `gui/tabs/overlap_tab.py`
  - `PipelineCalculatorGUI.create_placemark_tab()` -> `gui/tabs/placemarks_tab.py`
  - `PipelineCalculatorGUI.export_results()` -> `gui/actions/export_actions.py`
  - `PipelineCalculatorGUI.view_overlap_kml()` -> `gui/actions/open_kml_action.py`
  - parameter dialog + apply/re-analyze -> `gui/dialogs/params_dialog.py` + `gui/controllers/analysis_controller.py`

- Notes:
  - The GUI layer should depend only on:
    - `pipeline_calculator.core.*` (analysis)
    - `pipeline_calculator.export.*` (XLSX + KML)
    - `pipeline_calculator.parsers.*` (if needed)
  - Keep GUI modules thin: UI builds widgets; controller does work; state carries data.
- Tests (must exist and pass before continuing):
  - Unit tests remain green
  - Add/expand tests for non-UI helpers introduced in this phase:
    - export actions (already covered via `pipeline_calculator.export.*`)
    - open KML helpers (already covered via `pipeline_calculator.export.*`)
    - analysis controller logic (no tkinter imports)
  - Manual GUI smoke checklist updated (launch, import file, export xlsx, open corridor KML)
  - Optional: keep a tiny `tests/test_gui_imports.py` that imports non-tkinter GUI modules (`state`, `actions`) only

- Manual GUI smoke checklist (update as part of this phase):
  - Launch app
  - Import a KMZ/KML via Browse and via Drag-Drop
  - Verify Summary tab numbers populate (total miles, savings when overlaps exist)
  - Export XLSX and open it (verify sheets exist)
  - Open overlap corridor KML and verify it loads in Google Earth
  - Change analysis params and re-run (no crash, results change)

- Progress notes (so far):
  - Added new modular GUI package under `src/pipeline_calculator/gui/`:
    - `main_window.py`, `pages/`, `tabs/`, `dialogs/`, `actions/`, `controllers/`
  - Wired package entrypoint switch:
    - `PIPELINE_CALCULATOR_IMPL=new .venv/bin/python -m pipeline_calculator`
  - Added unit tests (non-tkinter):
    - `tests/test_gui_state.py`
    - `tests/test_analysis_controller.py`
  - Manual GUI smoke checklist:
    - Passed on macOS (import KMZ, results populated, export works).
  - CI/build entrypoint flip:
    - Completed in Phase 7 (`PIPELINE_CALCULATOR_BUILD_IMPL=new` in CI).

### Phase 7: Compatibility wrapper + build entrypoints (optional)

- Status: DONE
- Scope:
  - Switch PyInstaller/CI builds to the modular GUI entrypoint while keeping the legacy build path available:
    - New entry script: `src/pipeline_calculator_entry.py` (forces `PIPELINE_CALCULATOR_IMPL=new`)
    - Build-time selector: `PIPELINE_CALCULATOR_BUILD_IMPL=new|legacy`
  - CI defaults to modular builds by setting `PIPELINE_CALCULATOR_BUILD_IMPL=new` in `.github/workflows/build.yaml`.
  - Keep `src/pipeline_calculator_v3.py` (legacy monolith) available as a runtime fallback and for legacy builds.
- Tests (must exist and pass before continuing):
  - Unit tests remain green
  - Packaged build smoke checklist on macOS + Windows
 - Completion notes:
   - Added `src/pipeline_calculator_entry.py` (PyInstaller-friendly entrypoint that defaults to modular).
   - Updated build scripts to select entrypoint via `PIPELINE_CALCULATOR_BUILD_IMPL`:
     - macOS: `scripts/macos/build_app.sh`
     - Windows: `scripts/windows/build_exe.ps1`
   - Updated CI scripts to compile-check the new entrypoint when present:
     - `scripts/ci/macos_build.sh`
     - `scripts/ci/windows_build.ps1`
   - Updated GitHub Actions to build modular by default:
     - `.github/workflows/build.yaml` sets `PIPELINE_CALCULATOR_BUILD_IMPL=new`
   - Local packaged build sanity (macOS):
     - `PIPELINE_CALCULATOR_BUILD_IMPL=new bash scripts/macos/build_app.sh`
     - `bash scripts/macos/package_dmg.sh dist/Pipeline_Calculator.app`

## Files Likely To Change During Refactor

- Build entrypoints:
  - `scripts/macos/build_app.sh`
  - `scripts/windows/build_exe.ps1`
- Tests:
  - `tests/*` imports can gradually switch from `import pipeline_calculator_v3 as pc`
    to `from pipeline_calculator.core.analyzer import PipelineAnalyzer`, etc.
- Resource paths:
  - icon lookup and PyInstaller `--add-data` arguments if icons move under the package.

## Known Risk Areas

- PyInstaller + TkinterDnD:
  - `tkinterdnd2` depends on bundled `tkdnd` assets. The repo already uses a PyInstaller hook:
    `scripts/pyinstaller_hooks/hook-tkinterdnd2.py`.
  - After refactor, keep this hook and validate a packaged build on macOS + Windows.

- Resource resolution:
  - Finder-launched macOS apps and Windows GUI apps can have different working directories.
  - Use `sys._MEIPASS` (PyInstaller) and/or `importlib.resources` to locate icons reliably.

- GUI testing:
  - The unit tests cover core logic and export generation, not GUI interaction.
  - Refactor should avoid behavior changes; keep a small manual GUI smoke checklist.

## Optional Improvements If We Refactor

- Introduce small typed models (dataclasses) in `util/types.py` to reduce “stringly-typed dict” usage.
- Add a `core/results.py` module to standardize the result schema (pipelines, overlaps, savings, parameters).
- Add a CLI mode for headless runs (parse file, output JSON/XLSX) to speed up validation without rebuilding a DMG/EXE.
- Optional (bigger bet): migrate GUI from Tk/CustomTkinter to Qt/PySide6 (Purway-style) for more powerful widgets, theming, and long-term UI extensibility.
  - This should be treated as a separate project after Phase 6, because it’s a rewrite rather than an incremental refactor.
