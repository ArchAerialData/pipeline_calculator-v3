# Point-pin corrections and deferred optional geometry counts

Investigation date: 2026-09-18. Source revision: `f679424`.
Status: phase 1 implemented and verified. Toggle controls and polygon-count
reporting are deferred follow-up work.

## Phase 1: point-pin correctness and import reliability

Approved scope: fix existing point counts and the verified import gaps without
adding toggles, new preferences, polygon totals, or line-exclusion rules.

**A pin means an actual KML `Point` geometry with one valid location.** Multiple
Points within one MultiGeometry represent multiple pins. LineString vertices,
polygon corners, outer/hole ring coordinates, and icon style definitions are
never pins. Coincident Points still count individually because they are
separate source objects; no geographic deduplication or visibility filter is
introduced. A malformed Point is reported as an error, not converted into a
pin or silently certified as a complete count.

Implementation scope:

1. Extract every valid Point independently of accompanying line, track, polygon,
   or other Point geometry. Preserve the public `placemarks` list contract with
   one `{Placemark_ID, Name, Count: 1}` row per pin. Names/IDs supplied by clients
   can repeat; repeated names or locations do not collapse separate pins.
2. Preserve pipeline identity, generated names, coordinate paths, and mileage.
   Keep generated-name progression separate from the expanded pin count.
3. Update independent repair inspection and coverage comparison for all pins
   and mixed geometry. Verify SourceSession snapshots, retries, repaired-copy
   save/reopen, linked-document traversal, and the desktop analysis worker.
4. Aggregate routine non-centerline exclusions by document/category into
   informational notices. Preserve actionable malformed-input diagnostics and
   existing bounds, while preventing thousands of valid site polygons from
   exhausting the desktop diagnostic limit.
5. Keep point information available through the existing interface. Clarify
   labels as **Point pins**, make the existing table total agree with its rows,
   and avoid implying that Combined source counts are per-state counts.
6. Export matching point rows/totals to XLSX and retain the corrected list in
   JSON. State the loaded-source scope and avoid presenting partial input as
   an exhaustive inventory. Existing mileage sheets retain their contract.

No new generalized polygon inventory or preference migration is needed for
this phase. The existing authoritative pin list can flow through source
snapshots and analysis results directly. Add the broader inventory only when
the follow-up polygon/toggle feature requires it.

### Line closure and the user's crossing example

A closed LineString ends at its starting coordinate. A line can cross itself
in a 2D view while its endpoints remain different. The user's sketch, interpreted
as one path from its upper-left tip to its bottom tip, is such an open path.
Both self-crossing and genuinely closed LineStrings remain pipeline paths; an
enclosed-looking region is never inferred to be a polygon. The application
measures successive longitude/latitude coordinates and does not use altitude to
infer an overpass, underpass, physical junction, or 3D pipe length.

Regression checks include mixed/nested Point geometry, coincident pins, hidden
features, no counts from line/polygon vertices, malformed points followed by
valid points, stable generated pipeline names, linked files reached repeatedly,
snapshot mutation/source removal, repaired round trips, a 10,001-polygon desktop
run, closed/self-crossing line mileage, and Summary/table/XLSX/JSON consistency.

### Phase 1 verification results

- Broad regression run (`pytest -q -m "not native_gui"`): 890 passed initially;
  six small-window Placemarks layout failures and one legacy startup smoke
  timeout were found and corrected. Fourteen checks were skipped for unavailable
  optional audit/macOS dependencies; 74 explicitly native checks were deselected.
- All layout cases and native point presentation passed on rerun: 34 tests.
  Both desktop startup smoke modes and native point presentation then passed:
  2 tests. No assertions or time limits were weakened.
- Focused parser tests cover 18 new parameterized cases; source-session tests
  cover 10 new cases, including repaired save/reopen and the 10,001-polygon
  desktop-worker regression. The broader source/repair/worker batch passed 209
  tests, and targeted export/state/native UI checks also passed.
- Independent read-only review verified 512 combinations of unnamed lines,
  single/multiple/nested pins, and polygons against repair coverage projection.
- UI totals use compact existing surfaces. Extra wrapped scope paragraphs were
  removed from the point table and Summary during layout/startup verification;
  the XLSX sheet records the complete loaded-source scope explanation.

Regression sources:
[pin geometry](tests/test_point_geometry_parsing.py),
[source and repair round trips](tests/test_point_source_roundtrip.py),
[UI/export agreement](tests/test_point_pin_presentation.py),
[responsive layouts](tests/test_ui_layout.py), and
[desktop startup smoke](tests/test_build_helpers.py).

### Follow-up task: optional reporting

Defer both count toggles, their saved preferences, polygon/standalone-ring
totals, the optional compact counts panel, and category-gated exports. No
automation or separate Codex task is created; this document retains the backlog.
The detailed follow-up proposal below remains subject to later implementation
decisions. Its OFF defaults and visibility changes do not apply to phase 1.

## Historical baseline: verified before phase 1

Mileage is calculated from KML `LineString` and `gx:Track` paths. Actual
`Polygon` and `LinearRing` geometry does not become pipeline mileage.
The app classifies geometry by its encoded type, not by whether it looks like
a pipeline in Google Earth. A boundary exported as a `LineString` is measured,
including a closed `LineString` whose first and last coordinates match.

| Input | Baseline mileage | Baseline point/polygon statistics |
| --- | --- | --- |
| Line plus separate polygon and point pin | Same as line alone; run completes with a polygon warning | One point record; no polygon total |
| Polygon or standalone LinearRing | Excluded | Warning, no retained count |
| Ordinary Point placemark | Zero added mileage | One record with `Count = 1`; already shown without a toggle |
| Line + Point + Polygon within one MultiGeometry | Line measured, polygon excluded | Point omitted because line extraction exits first |
| Multiple Points within one Placemark | Zero added mileage | Only the first Point is inspected; at most one point record |
| Closed LineString | Included, including its closing segment | No polygon classification |
| Well-formed points-only input | Zero miles; result marked complete | Point records available |
| Well-formed polygons-only input | Zero miles; result marked incomplete (`no_supported_features`) | No polygon total |

Current point rows are therefore **not a reliable total of all Point geometries**.
KML `Placemark` is a feature container that can contain lines, polygons, points,
or a `MultiGeometry`; counting Placemark XML elements would count pipelines too.
Icon styles are presentation resources and are not additional pins.

Runtime verification through the desktop worker and parser/analyzer routes:

- The probe's line alone and line + separate polygon + pin both measured
  `853.9385691445951 m` / `0.5306117655813107 US survey mi`.
- A mixed Line/Point/Polygon Placemark produced zero point records.
- Two Points inside one Placemark produced one point record.
- The closed-LineString probe measured `3364.991319718837 m` /
  `2.0909044863852055 US survey mi`.
- A valid line plus 10,001 separate Polygon Placemarks failed the actual desktop
  worker with `diagnostic_limit`. Each polygon emits a warning and the source
  preparation route caps diagnostics at 10,000; the direct analyzer alone does
  not exercise that same cap.
- A line with an invalid polygon coordinate (`999,999`) completed on the
  ordinary route with unchanged mileage. Adding repairable metadata formatting
  triggered the stricter repair verifier, which blocked `invalid_coordinate`.
  Valid point-only and polygon-only files needing repair were also blocked for
  lacking supported centerlines.

Verification: 16 synthetic cases; 86 targeted parser/source/worker regression
tests passed. Five existing non-native preference/options tests also passed
(four native GUI tests were deselected). No GUI visual inspection was performed.
Reproduction artifacts are retained locally in the gitignored
[investigation directory](.validation-output/geometry-count-investigation/README.md),
including the probe script, JSON results, generated fixtures, and pytest log.

These are synthetic fixtures against the source checkout, not a certification
of a separately installed executable or an unprovided client KMZ.

The existing Summary reports point-record count inside **Additional details &
run settings**. The **Placemarks** tab lists ID, Name, Count. It is Combined-only;
state views display an explanatory message. JSON includes the point records.
The XLSX exporter does not currently export point totals or point inventory rows.
There is no polygon count or point/polygon count switch.

Input scope is the selected primary KML and reachable supported local KML
NetworkLinks, each document read once. Unreachable archive KML is reported but
not analyzed; remote links are not fetched. Existing totals do not mean every
shape in every file physically present in a KMZ. Visibility is not a parser
filter, so hidden source features can be included.

Malformed input is a separate issue from valid non-centerline geometry. Ordinary
input has structural validation, and coordinate errors can make results
incomplete. The verified-repair route also independently checks point and ring
coordinates and requires supported centerlines. Count switches must not bypass
these safeguards or modify the source.

## Follow-up proposal: count definitions

1. **Point pins:** one per recognized KML `Point` geometry, including every Point
   inside nested MultiGeometry and Points accompanying line/polygon geometry.
   Two coincident points or identical names/IDs still count twice when they are
   separate source geometries. Do not count styles, icon files, Models,
   GroundOverlays, folders, or all Placemark containers as pins.
2. **Polygons:** one per recognized KML `Polygon`. Its outer ring and any hole
   rings belong to that polygon and never increment the polygon total.
   Multiple Polygon children count individually.
3. **Standalone closed boundaries:** count `LinearRing` geometry outside Polygon
   boundaries separately. If the switch promises "polygons / closed boundaries,"
   report their combined total with a polygon/standalone-ring breakdown. Never
   label that sum solely "Polygons."
4. **Closed LineStrings:** retain their current line semantics. Closure alone
   cannot distinguish a legitimate pipeline loop from a site boundary. An
   advisory count can identify closed paths for review; excluding selected
   layers/features is a separate future feature with its own mileage policy.
5. **Total existing:** inventory recognized source geometry occurrences, without
   spatial deduplication. A detected geometry with invalid coordinates still
   exists; report its validation issue rather than silently reduce the total.
   Counts are not a certification of geometric validity or real-world facility
   identity. Topological repair/area calculation is outside this feature.
6. **Coverage:** call the display "Loaded feature counts" and bind it to the
   same captured source documents as the mileage run. If documents are missing,
   skipped, or unreadable, label available totals partial; never imply an
   exhaustive KMZ total. An overlap-calculation failure alone need not make
   source inventory partial.

For a clean first release, use two independent switches:

- **Count point pins**
- **Count polygons / closed boundaries**

Recommended defaults: OFF for both new optional summary/export features. This
changes visibility of existing point details in the new GUI, so document that
migration: OFF hides that category's summary statistic and detail page; ON uses
the authoritative geometry inventory for both. Keep the legacy parser's public
point-return contract compatible. JSON retains the legacy `placemarks` field
regardless of these switches; only the new `feature_counts` projection is gated.
The switches affect reported statistics, never mileage eligibility. They remain
independent of State breakdown.

## Follow-up backend design

### Inventory and public compatibility

Add a small typed source-geometry inventory to `ParseResult` and `_ParserState`.
Capture it while traversing the validated XML, independently of line-vs-point
extraction precedence. Store geometry kind, source document, feature ordinal,
geometry ordinal, source Placemark ID/OBJECTID, name, and validation information.
A source-document/feature-ordinal/geometry-ordinal identity is stable even when
client IDs are absent or duplicated. Do not retain full polygon coordinate
arrays merely to calculate a count.

Use explicit aggregate fields for point, polygon, and standalone-ring totals;
include coverage status and reasons. Preserve `pipelines`, public tuple-return
wrappers, and legacy `placemarks` records initially. New UI/export readers use
the authoritative inventory, not `len(placemarks)`. This avoids silently changing
existing consumers while fixing the new totals for MultiGeometry.

The cheap source inventory should be captured once even if reporting switches
are OFF. This lets a retained SourceSession supply accurate future runs without
reopening a potentially changed file. Optional switches govern result projection
and presentation. Bound inventory work and rows using existing source limits and
execution-context cancellation checks.

### Immutable sessions and repair verification

Update SourceSession baseline serialization, `fresh_parse()`, repaired-source
coverage comparison, and both saved-copy reimport comparisons. Extend the
independent repair inventory to verify the new geometry identities/totals while
retaining its independence from production extraction. Otherwise the ordinary
parser could report correct counts that disappear on retry or repair/save.
Normalize standalone-KML source paths in new inventory identities during the
existing saved-copy identity comparison, just as pipeline paths are normalized.

Keep existing repair approval, centerline requirements, byte preservation,
resource bounds, and structural rejection behavior. A broader count-only repair
workflow is outside this first change. For polygon-only ordinary input, requested
counts may be shown alongside the existing no-centerline/incomplete notice;
do not present it as a successful pipeline-mileage analysis.

### Options and result contract

Add validated immutable `count_points: bool` and `count_closed_shapes: bool`
fields to `AnalysisOptions`. Record their actual run snapshot in results, and
expose a versioned `feature_counts` object. Suggested shape:

```json
{
  "schema_version": 1,
  "scope": "loaded_source_documents",
  "points": {"status": "complete", "total": 128},
  "closed_shapes": {
    "status": "complete", "total": 14,
    "polygons": 12, "standalone_rings": 2
  }
}
```

Allow `not_requested`, `complete`, `partial`, and `unavailable`; omit or null the
total for not-requested/unavailable data. An actual zero is reserved for a
completed inventory with zero occurrences. Partial totals have explicit reasons
and represent observed counts only. Validation notices describe recognized but
malformed geometry separately from missing-document coverage.

Keep these source totals on the Combined result. Do not insert source polygons
into overlap, segmentation, state clipping, or corridor-generation inputs.
Generated corridor polygons are outputs and do not increment input counts.

Aggregate expected non-centerline exclusions into informational diagnostics
instead of one warning per polygon. Preserve real malformed-geometry and source
coverage warnings/errors with bounded detail. Otherwise routine boundaries make
successful runs appear problematic and can exhaust the 10,000-diagnostic source
limit.

## Follow-up frontend and export design

Use the existing state preference flow: persisted preferences -> shared Tk
bindings -> immutable per-run options -> worker. Generalize the preference store
and reusable toggle control rather than cloning three state-specific classes.
The current writer replaces the whole JSON file with only `state_breakdown`;
adding independent writers would cause settings to erase one another. Migrate
schema v1 explicitly, preserve state preference, validate booleans, and continue
atomic saves with session-only fallback on save failure.

Add both controls alongside State breakdown in the input settings and parameter
dialog. Preserve draft Apply/Cancel behavior, keyboard activation, busy locking,
and accessible ON/OFF labels. Cover the modern GUI and legacy entrypoint, all
file-open routes, and retries against the captured source session.

Keep the two existing mileage cards primary. Add a compact secondary panel below
them when at least one count option was requested:

```text
Loaded feature counts                         Combined
Point pins                        128
Polygons / closed boundaries       14
  12 polygons, 2 standalone boundaries
Excluded from pipeline mileage.
```

Only show enabled rows. Display requested zero totals explicitly. Show a concise
partial-coverage or malformed-geometry notice when applicable. Use responsive
stacking and integer formatting; counts do not need mileage-sized cards.
State views should show "Feature counts are available in Combined" rather than
zero, duplicated global totals, or implied per-state allocation.

For drill-down, reuse the existing table/loading components in one **Features**
page, with Point pins / Closed boundaries filters as needed. Use Name, Type,
source ID, source document, and Count or per-geometry identity. Avoid separate
permanent tabs for every type. The existing point table must use authoritative
inventory whenever point counting is enabled, so its rows reconcile with the
new point total. Polygon drill-down can be deferred in a totals-first release.
Remove the old unconditional point statistic from Additional details and gate
point detail navigation on the captured point-count option.

Add an XLSX **Feature Counts** sheet containing requested totals, scope, status,
breakdown, and coverage notes. Add inventory rows only if the drill-down ships.
Record ON/OFF options in Analysis Details, and preserve current mileage sheets
and formula behavior. Apply the existing spreadsheet literal-text safeguards
to client names/IDs. JSON carries the same versioned counts and option snapshot.
Exports must read completed results, not current switches or selected UI scope.

## Follow-up delivery and acceptance checks

1. Implement inventory and compatibility contracts; verify unchanged mileage.
2. Propagate through source snapshots, repair verification/save, options, and
   analysis results; add focused regression coverage.
3. Migrate shared preferences and wire both GUI entrypoints; add the compact
   stats panel and export totals together.
4. Verify Windows/macOS responsive layouts and existing repair/state workflows.

Essential regression matrix:

- Line-only vs line + points/polygons produces identical source mileage,
  overlap results, state allocations, and pipeline identities for all four
  point/polygon toggle combinations and either State breakdown setting.
- Every Point/Polygon in nested or mixed MultiGeometry is counted; holes and
  outer rings do not inflate polygon totals; standalone rings remain separate.
- Closed LineStrings remain measured and are not silently reclassified.
- Duplicate names/coordinates count as distinct source occurrences; repeated
  NetworkLinks to one document do not multiply its inventory; hidden features
  are consistently included.
- Missing/remote/unreachable documents, invalid coordinates, points-only and
  polygons-only inputs report truthful scope and status without fabricated zero.
- Large polygon inventories use bounded work and aggregated routine notices.
- Fresh parse, retained-session rerun, approved repair, saved-copy reopen, direct
  analyzer, and desktop worker agree on geometry inventory.
- Option defaults/migration, restart persistence, all import routes, busy state,
  Apply/Cancel, and export-after-changing-preferences preserve run snapshots.
- Enabled zero/partial counts and state scope messaging are legible at narrow
  sizes and display scaling; XLSX and JSON match the displayed Combined totals.

## Source map

- [Geometry extraction and current precedence](src/pipeline_calculator/parsers/kml_kmz.py)
- [Source snapshots, document scope, and repair coverage](src/pipeline_calculator/parsers/source.py)
- [Independent geometry inspection](src/pipeline_calculator/parsers/repair.py)
- [Analysis options](src/pipeline_calculator/core/options.py)
- [Analyzer result construction](src/pipeline_calculator/core/analyzer.py)
- [Desktop analysis worker](src/pipeline_calculator/gui/controllers/analysis_controller.py)
- [Shared preference binding and persistence](src/pipeline_calculator/gui/preferences.py)
- [Parameter dialog](src/pipeline_calculator/gui/dialogs/params_dialog.py)
- [Results scope and tabs](src/pipeline_calculator/gui/pages/results_page.py)
- [Summary cards and existing point count](src/pipeline_calculator/gui/tabs/summary_tab.py)
- [Existing point table](src/pipeline_calculator/gui/tabs/placemarks_tab.py)
- [XLSX exporter](src/pipeline_calculator/export/xlsx.py)
- [JSON/export actions](src/pipeline_calculator/gui/actions/export_actions.py)
- [Existing mixed-geometry structure regression](tests/test_kml_structure_safety.py)
- [Existing polygon-only mileage regression](tests/test_kmz_parsing.py)
- [Google KML reference: geometry and containers](https://developers.google.com/kml/documentation/kmlreference)
