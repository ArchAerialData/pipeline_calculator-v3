# State boundary analysis

Status: approved implementation specification. This document records the agreed
behavior; validation results and remaining platform checks belong in the validation
report linked from the README. Boundary data and software precision are distinct.

## Agreed behavior

- Persistent **State breakdown** switch on the import screen, OFF on first use.
- Support the 50 US states and Washington, DC, using bundled offline Census
  TIGER/Line 2025 boundaries. No live boundary service or runtime downloads.
- Parse once, preserve normal combined analysis, partition source geometry, and
  independently analyze each state's exclusive fragments in memory.
- Existing two result cards, Combined/state selector, compact comparison table.
- One workbook per export package, optional combined/state KMZs selected by
  default, optional JSON initially unchecked. No automatic per-pipeline files.
- Equal accounting allocations for positive-length verified shared borders at
  every length; no majority-state shortcut for actual short crossings.
- Shared geometry stored/exported once, with no state overlap discount.

```text
State attributed original = exclusive interior + shared allocation
Sum(state attributed original) + outside coverage + unresolved = combined original
State attributed adjusted = attributed original - exclusive interior savings
```

State savings and adjusted totals need not sum to combined savings/adjusted
mileage. In particular, cutting a qualifying overlap can leave both state portions
below the existing minimum continuous length. Shared-border savings are not
calculated per state; this does not claim that no overlap exists there.

## Geometry and interfaces

Refactor `PipelineAnalyzer` into file parsing plus analysis of normalized features.
Add frozen `AnalysisOptions(state_breakdown=False)`, preserve existing combined
result fields, and add a versioned `geography` result with provenance, state
results, canonical fragments, shared allocations, diagnostics, and reconciliation.
Return JSON-native data; keep engine geometry objects internal.

Use Shapely for planar topology alongside pyproj for GRS80 geodesic length and
existing US survey mile conversion. Densify geodesic curves with bounded error,
handle longitude wrapping and the antimeridian, refine crossing positions on the
source path, and retain original vertices. Never calculate reported mileage using
planar `.length`. Numerical target: 1 cm relative to the selected boundary model;
conservation tolerance per source and project: `max(0.001 m, total_meters * 1e-10)`.
Failures must be disclosed, never hidden by redistributing residual mileage.

Each fragment retains source identity, path index, distance-along-path endpoints,
coordinates, measured length, classification, and adjoining state codes. Classify
every positive-length interval once: state interior, shared border, outside
coverage, or unresolved. Endpoint contacts contribute no mileage. Do not snap a
nearby pipeline onto a border or erase small true crossings.

Group fragments under their original source pipeline, preserving disconnected
paths and repeated state visits without bridges. Names, OBJECTIDs, and XML IDs
are labels, not reliable unique keys. Per-state overlap receives only exclusive
interior fragments; existing short sampled-tail behavior remains unchanged.

Clip corridor polygons after construction, including fallback shapes. Serialize
multipart polygons and holes. Omit invalid state visuals with diagnostics rather
than falling back to an out-of-state rectangle.

Bundle a reproducibly prepared 51-jurisdiction resource with full boundary detail,
source URL, source CRS, transformation metadata, vintage, checksum and attribution.
Boundary updates occur through reviewed releases. Validate source topology and
record transformation accuracy separately from computation tolerance.

## UI and lifecycle

Switch label: **State breakdown**. Helper: **Split mileage and overlap results by
U.S. state.** Persist a versioned preference per OS user using atomic replacement;
default missing/malformed settings to OFF. An unwritable file causes a brief
session-only notice, not analysis failure. Parameter dialogs use a draft choice
committed only on Apply. Snapshot options before starting a worker. Support both
GUI entrypoints and Browse, drag-and-drop, retry, and parameter reanalysis.

Results default to Combined. Encountered states are alphabetical; switching scope
reuses completed results. Scope applies to pipelines, overlaps, and corridor
previews. Counts represent original pipelines rather than clipped fragments.
Placemarks remain Combined-only in v1 with an explanatory state-view message.

The Combined table shows state, attributed original mileage, adjusted mileage,
mileage removed, and status. Clicking a row selects its scope. Existing detail
disclosure contains provenance and accounting details. Shared allocation changes
state wording to **Mileage assigned to Texas**, with **Includes 0.250 mi of
shared-border allocation** and the no-discount explanation. Positive values below
display precision appear as **<0.001 mi**, not zero. Distinguish states represented
from actual crossings. Single-state/no-crossing results are normal.

Preserve single-job control, stale-result rejection, cancellation and runtime
warnings. Aggregate progress across parsing, combined analysis, boundary loading,
partitioning and state runs; only the parent completes the job. Use spatial indexes
and bounded work. Reuse combined analysis only when every interval belongs
exclusively to one state. Cancellation publishes no partial result. Geography
failure keeps combined results available and explicitly marks geography incomplete
or unavailable. State overlap failures retain original mileage and show unavailable
adjusted mileage, not zero.

## Exports

Ordinary mode preserves the XLSX/JSON Save As workflow. Geography mode exports
all scopes regardless of the currently selected result view:

```text
<input>_analysis_<timestamp>/
  analysis.xlsx
  analysis.json                 # optional
  Combined/analysis.kmz          # optional
  States/Oklahoma/analysis.kmz
  States/Texas/analysis.kmz
```

Create only populated/requested folders. Stage the package before publication;
resolve collisions without overwriting earlier exports. Workbook sheets: State
Summary; combined Pipeline Length Analysis and Pipeline Overlap Analysis; State
Pipeline Lengths; State Overlap Analysis; Shared Borders when present; Analysis
Details; Diagnostics when present. State tables separate interior, allocation,
attributed original, interior savings, attributed adjusted and completion status.
Keep ordinary workbook contracts and source-text/formula protection.

Combined KMZ includes source line geometry exactly once across State Interiors,
Shared Borders and coverage-exception folders. State KMZ contains exclusive
interior geometry and clipped polygon corridors. State maps reconcile to interior
mileage; descriptions reference allocated mileage in the workbook and Combined
map. Shared-only states get report rows without empty map files. Do not export
duplicate original lines, corridor centerlines or border-reference LineStrings,
even hidden: the parser counts them. JSON references canonical shared intervals.

## Acceptance and release gates

Delivery order: boundary resource/partition engine; conservation and state
accounting; UI/preferences/progress; workbook/maps/package exports; packaging.

Required scenarios:

- One state, exact two-state cut, multiple crossings, reentry, disconnected paths,
  repeated names and IDs, reversed paths and redundant vertices.
- Long sparse geodesics, Alaska, Hawaii, antimeridian, islands and holes.
- Endpoint touches, shared-border runs, tiny actual crossings, outside-only input,
  unresolved portions and corrupt/missing resources.
- Opposite-side nearby pipelines do not overlap within states; minimum overlap
  thresholds apply independently; shared portions do not assist qualification.
- Per-source and project conservation, exact shared allocation accounting, and
  independent verification of state-line and corridor containment.
- Reimport combined KMZ to recover source mileage and state KMZ to recover
  interior mileage within serialization precision.
- Persistent preference restart, invalid/unwritable settings, cancellation, retry,
  state failures, no premature 100%, small/high-DPI windows and keyboard use.
- Both entrypoints and actual offline Windows/macOS packaged resource/native
  dependency loading. Visual review and a representative exported package.

Release gates: topology/transform validation, numerical precision and containment,
reconciliation, bounded performance, packaged operation and UI/export review.
Do not present numerical precision as proof of surveyed ownership.

## Sources

- [Current analyzer](src/pipeline_calculator/core/analyzer.py)
- [KML/KMZ parser](src/pipeline_calculator/parsers/kml_kmz.py)
- [Census 2025 state archive](https://www2.census.gov/geo/tiger/TIGER2025/STATE/)
- [Census TIGER/Line technical documentation](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2025/TGRSHP2025_TechDoc.pdf)
- [Cartographic boundary simplification](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)
- [Shapely geometry model](https://shapely.readthedocs.io/en/stable/manual.html)
- [pyproj geodesic operations](https://pyproj4.github.io/pyproj/stable/api/geod.html)
