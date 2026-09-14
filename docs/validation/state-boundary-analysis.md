# State boundary analysis validation

Implementation and local verification: September 14, 2026, Windows 10, Python
3.11.9. This report accompanies the [approved specification](../../STATE_BOUNDARY_ANALYSIS_PLAN.md).
The application code is implemented; a native macOS packaged run remains a release
gate because this workspace runs on Windows.

## Delivered behavior

- Persistent, per-user **State breakdown** toggle in both GUI entrypoints, including
  draft changes in Adjust Parameters and immutable options for running jobs.
- Offline boundary partitioning, per-source mileage reconciliation, independent
  state overlap analysis, equal shared-border allocation and explicit coverage or
  calculation failures. Combined analysis remains available if geography fails.
- Combined/state views with the existing cards, a comparison table, scoped pipeline
  and overlap rows, and clipped corridor previews. Original source identities survive
  clipping; shared geometry is stored once.
- Atomic package export with one workbook, optional JSON, and optional Combined/state
  maps. State maps contain exclusive interior lines and clipped polygon corridors.
  Ordinary-mode export behavior remains covered by the existing workbook tests.

Implementation entrypoints: [analyzer](../../src/pipeline_calculator/core/analyzer.py),
[state orchestration](../../src/pipeline_calculator/core/state_analysis.py),
[partition engine](../../src/pipeline_calculator/core/geography/partition.py),
[preferences](../../src/pipeline_calculator/gui/preferences.py),
[results view](../../src/pipeline_calculator/gui/pages/results_page.py), and
[package export](../../src/pipeline_calculator/export/package.py).

## Evidence

The final full suite passed **352 tests** in 143.92 seconds, including the shared-border
crossing-count correction, preference-notice cleanup and packaged-smoke CI checks.
The preceding focused geometry, state-analysis and UI suite passed **38 tests**
in 18.13 seconds. Test reports are
generated locally under `.validation-output/` and are not distribution assets.
Artifact checksums and compact test evidence are retained in
[state-boundary-evidence.json](state-boundary-evidence.json).

| Area | Verification |
| --- | --- |
| Partition and measurement | Exact splits, sparse geodesic edges, reentry, disconnected paths, duplicate names/source identity, tiny crossings, endpoint touches, shared runs, direction/vertex invariance, islands/holes, antimeridian, outside and unresolved intervals. Both ledger and serialized-coordinate lengths are audited. |
| State overlap | Opposite-state lines excluded; shared geometry cannot contribute to state savings; state minimum-length qualification independent of Combined; failed state overlap keeps original mileage and unavailable adjusted values. |
| Workflow | Parse once, single-state reuse, immutable options, aggregate progress, cooperative cancellation, stale results, retry and parameter reanalysis. |
| Preferences and desktop | Restart persistence, malformed settings, atomic save failure, Apply/Cancel, both GUI entrypoints, keyboard state selection, constrained window layout and elevated widget scaling. Native GUI tests run on an isolated Windows desktop. |
| Exports | Fixed workbook sheets, source-text escaping, shared references, JSON-native data, no empty shared-only state map, line-mileage round trips, polygon holes/multipart containment, package collision handling and failure cleanup. |
| Packaging | Shapely native libraries and boundary resource included by both build scripts; Windows executable built and tested offline with both GUI entrypoints. Native macOS execution pending. |

The final Windows executable passed normal launch with its native window visible
for five seconds, then passed the new CI packaged-smoke runner for both `new` and
`legacy` implementations with `PROJ_NETWORK=OFF`. Each frozen run loaded all 51
jurisdictions, split a Texas/Oklahoma path, reconciled mileage, and reimported the
Combined and state KMZ exports. CI now runs the same
[packaged-smoke gate](../../scripts/validation/check_packaged_smoke.py) after both
platform builds and uploads the reports even when a check fails.

- [Windows preview executable](../../.validation-output/state_boundary_build/dist/Pipeline_Calculator_v4.16-dev.212d0d767c0d.dirty.exe)
- [Packaged-smoke result](../../.validation-output/state-boundary-packaged-final/summary.json)
- [Normal Windows startup result](../../.validation-output/state-boundary-startup.json)

Test sources: [geometry](../../tests/test_state_geometry.py),
[scoped analysis](../../tests/test_state_analysis.py),
[state UI](../../tests/test_state_breakdown_ui.py),
[geography export](../../tests/test_geography_export.py), and
[ordinary workbook contract](../../tests/test_export_workbook.py).

An independent final code review found a display-only issue for a route that entered
a shared border and then emerged in another state. The crossing counter now preserves
the preceding exclusive state through a shared run, resets across disconnected paths
and coverage exceptions, and has forward/reverse regression coverage.

## Reviewed example and interface

The real-boundary Texas/Oklahoma example contains two parallel pipelines. Combined
original mileage is **887.746367450622 m**, with **440 m** of overlap savings.
Oklahoma receives **551.310110178268 m**, with **275 m** savings. Texas receives
**336.43625727235394 m**, with **0 m** savings because its individual fragments fall
below the 200 m qualifying minimum. Original mileage reconciles with **0 m**
difference. State savings correctly differ from Combined savings.

The reviewed package has six workbook sheets, one JSON, one Combined KMZ and two
state KMZs. It has no Shared Borders or Diagnostics sheet because neither applies
to this example. The sheet set expands only when applicable, not for each state.

Local review artifacts:

- [Sample workbook](../../.validation-output/state-boundary-export-review/Texas-Oklahoma_analysis_20260914_153637/analysis.xlsx)
- [Complete sample JSON](../../.validation-output/state-boundary-export-review/Texas-Oklahoma_analysis_20260914_153637/analysis.json)
- [Combined screenshot](../../.validation-output/state_boundary_demo/ui-combined.png)
- [Texas screenshot](../../.validation-output/state_boundary_demo/ui-texas.png)
- [Input screenshot](../../.validation-output/state_boundary_demo/input.png)

The screenshots were visually inspected. The selector sits above the existing tabs;
the summary retains two primary cards with the comparison table below. State details
use the same layout. Boundary and accounting details stay in the disclosure.
The screenshots use a longer Texas/Oklahoma input where both states qualify for
overlap savings; the workbook above uses the shorter threshold example.

## Boundary provenance and numerical limits

The bundled resource contains all 51 supported jurisdictions, with full source
vertices, islands, holes and represented water areas. Each source/transformed polygon
is validated during preparation; runtime loading verifies component checksums and
validates the antimeridian-normalized geometry. Positive-area ambiguity encountered
during partitioning is unresolved rather than silently allocated.

| Resource | SHA-256 |
| --- | --- |
| Original Census archive | `59a220888a8d9be8117c4fcd38f542bd02d81abf0d198c78113595ad540dd957` |
| Bundled `states_2025.zip` | `55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539` |

The [preparation script](../../scripts/data/prepare_state_boundaries.py) requires the
explicit original archive, fixes ZIP metadata, preserves polygon detail, and records
the source CRS and per-component datum operation. Mainland components use EPSG:1188;
main Hawaiian islands use EPSG:1252; applicable Aleutian components use EPSG:1251.
Thirteen remote Alaska/Hawaii components fall outside the published operation areas
and explicitly retain approximate coordinate equivalence with unknown accuracy.
These limitations are recorded in exported provenance; the app does not claim
surveyed ownership accuracy.

Reported mileage uses GRS80 geodesic distances and US survey miles. Local projections
serve clipping only. The engine uses bounded geodesic chunks, adaptive projected
boundary refinement, and crossing refinement on the original path. Its numerical
clipping target is 1 cm relative to the stored boundary model. Conservation is checked
at `max(0.001 m, source_meters * 1e-10)` per source and per analysis. No balancing
adjustment is applied after a failed check. Genuine tiny crossings are retained;
coincidence handling is numerical roundoff, not a user-distance snapping tolerance.

A 10,001-vertex Texas input benchmark completed partitioning in approximately
2.95 seconds, producing one continuous fragment with a passing reconciliation audit.
This is one local measurement, not a guaranteed runtime for all geometries. Work
limits and cancellation checkpoints bound difficult clipping and overlap workloads.

Primary references: [Census 2025 state archive](https://www2.census.gov/geo/tiger/TIGER2025/STATE/),
[TIGER/Line technical documentation](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2025/TGRSHP2025_TechDoc.pdf),
[Shapely geometry model](https://shapely.readthedocs.io/en/stable/manual.html), and
[pyproj geodesic operations](https://pyproj4.github.io/pyproj/stable/api/geod.html).

## Remaining release gate

Run the native macOS CI build, packaged smoke checks and normal launch on macOS
before distributing this feature there. Windows verification does not establish
macOS native-library loading or desktop behavior. A user review of the included
screenshots and sample workbook can still refine appearance without changing the
accounting contract. No publishing, release tagging or remote push was performed.
