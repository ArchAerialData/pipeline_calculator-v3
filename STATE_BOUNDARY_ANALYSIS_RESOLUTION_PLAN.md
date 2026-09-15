# State boundary analysis: audit and resolution plan

Status: **audit complete; fixes deferred**. No application code or automated test
files were changed for this audit. The fixtures and this plan can be reviewed while
the separate **Fix DMG UI contrast and Summary** task finishes its work.

Audited baseline: `a8640fd783e938cf7ea243196dc97bd82f608091` (`State Aware Planning`).
Inspection and probes used an isolated export of that commit, not the GUI files
being edited in the shared working tree. Findings must be checked again against
the other task's final changes before implementation.

## 1. Findings at a glance

| ID | Priority | Confirmed behavior | Impact |
| --- | --- | --- | --- |
| SB-01 | P2 | A precisely shared 364.439 m border path is classified entirely unresolved because projection roundoff exceeds the coincidence test. | Shared allocation is unavailable for valid shared geometry. |
| SB-02 | P2 | A real Texas/New Mexico shared path produces an unresolved floating-point tail in one direction only. | Otherwise correct state analysis is marked incomplete; direction invariance fails. |
| SB-03 | P2 | An ordinary antimeridian input analyzes successfully but package export with maps aborts. | The default export choice prevents delivery of the entire package, including its workbook. |
| SB-04 | P2 | State corridor clipping omits valid fallback shapes; unusable Combined corridors can be omitted without a diagnostic. | Missing visualizations and inconsistent explanations between scopes. |
| SB-05 | P2 | The public float progress callback reaches 100% before geography starts and emits no geography updates. | API callers see premature completion; desktop context-based progress is unaffected. |
| FX-01 | Resolved | The initial archive contains polygons; the subsequently supplied Centerlines KMZ contains valid line geometry. | Both negative and positive regression fixtures are now available. |

Resolve SB-01 through SB-03 before declaring the shared-border and antimeridian
acceptance scenarios complete. Mileage conservation passed
the reproduced cases; conservation alone does not prove correct state ownership,
completion status, or export success.

## 2. Preserved fixture and verified input mismatch

Saved fixture: [adamas_ng_pipeline_row.kmz](tests/fixtures/geography/adamas_ng_pipeline_row.kmz).
Its [expectation manifest](tests/fixtures/geography/adamas_ng_pipeline_row.expected.json)
and [usage notes](tests/fixtures/geography/README.md) belong beside the archive.

- Source found at `C:\Users\rbake\Downloads\Adamas - NG_PIPELINE_ROW.kmz`.
  The backslashes in the supplied underscore-escaped path were not actual folders.
- Byte-for-byte copy; source retained. Size: **3,709,955 bytes**.
- SHA-256: `bd2006ad9c3e673b1a2337409c0b84c3cf375b2f926656de2bee0dcca0aa4892`.
- Direct XML inspection: **281 Placemarks, 281 Polygons, 922 LinearRings,
  zero LineStrings, zero points and zero gx:Track/gx:MultiTrack elements**.
- Independent polygon containment against the bundled state model: **Texas 57,
  Louisiana 197, Wyoming 27**. All 281 polygons were valid and wholly contained
  in one state. This is polygon coverage, not a pipeline mileage result.
- The current parser correctly returns zero pipelines and zero mileage, with 281
  `unsupported_geometry` warnings and one `no_supported_features` error. Combined
  analysis is incomplete; geography is incomplete with no state pipeline scopes.

**Resolution:** retain this exact archive as a negative regression fixture. The user
subsequently supplied the separate centerline archive described below, resolving the
positive-fixture blocker. Do not derive pipeline mileage from polygon perimeters;
that would change the approved geometry contract and would not establish centerline
lengths.

### Centerline follow-up — positive fixture verified

Saved [adamas_ng_pipeline_row_centerlines.kmz](tests/fixtures/geography/adamas_ng_pipeline_row_centerlines.kmz)
from `Adamas - NG_PIPELINE_ROW-Centerlines.kmz`, preserving the original.
Size: **515,885 bytes**; SHA-256:
`c3095b793deaee49de963058a3399b39d79438f7aeb64d6c9056eae9af1127d6`.
The [independent expectation manifest](tests/fixtures/geography/adamas_ng_pipeline_row_centerlines.expected.json)
contains per-source state assignments and original geodesic lengths.

The file contains **281 source features, 4,827 LineString paths and 48,327 vertices**,
including 67 multipart features and 15 closed paths. There are no polygons or invalid
coordinates. All XML IDs repeat `ID_00000`; the application correctly retains 281
distinct source pipelines. Whole-path containment checked independently of the app's
parser and partition engine agrees with all state assignments:

| State | Source pipelines | Paths | Original US survey miles |
| --- | ---: | ---: | ---: |
| Louisiana | 197 | 3,563 | 779.941 |
| Texas | 57 | 452 | 208.211 |
| Wyoming | 27 | 812 | 233.410 |
| Total | 281 | 4,827 | 1,221.562 |

Full default-parameter analysis completed on the fixed baseline in approximately
20.6 seconds. Geography is complete with **zero crossings, shared mileage, outside
mileage or unresolved mileage**. Original state totals reconcile within 1 mm (observed
residual approximately `9.3e-10 m`). The workbook/JSON/Combined map and all three state
maps exported successfully; all map mileage round trips passed, with the largest
observed absolute difference approximately `2.3e-8 m`.

This fixture's state overlap savings were zero at default settings. Treat that value
as characterization rather than an independent overlap oracle. These successful checks
do not resolve SB-01 through SB-05, which concern other inputs or API paths.

Proposed fixture tests, after the concurrent work finishes:

1. Verify the archive checksum and independently count XML geometry types.
2. Assert polygon outlines contribute no pipeline count or mileage with the toggle
   both OFF and ON. Assert incomplete status and a useful unsupported-input diagnostic.
3. Check the user sees a concise explanation rather than a successful zero-mileage
   report. Consolidating repetitive per-polygon warnings in the primary UI is a UX
   review item; retain detailed diagnostics for investigation.
4. Use the new centerline fixture's independently verified source counts, membership
   and geodesic lengths. Assert each original path is assigned once, multiple states are represented,
   `crossing_count == 0`, no artificial bridge is created, state totals reconcile,
   and Combined/state KMZ reimports preserve original/interior mileage.
5. Keep the existing synthetic disconnected-path test alongside the real centerline
   fixture. Test repeated XML IDs, multipart paths and source-level counts explicitly;
   the original polygon archive remains a separate negative test.

## 3. Geometry fixes

### SB-01 — shared-border coincidence depends on projection precision

Source: [partition.py](src/pipeline_calculator/core/geography/partition.py), especially
`_local_projection` near line 34 and the coincidence branch near line 103.

Reproduction uses adjoining synthetic polygons with the canonical shared meridian
`longitude = -118.21696820486636` and a path on that meridian from latitude
`40.595931349466056` to `40.59921322208294`.

- Expected: one **364.4392780854252 m** shared interval; each adjoining state gets
  **182.2196390427126 m**, with no state overlap discount.
- Actual: the complete path is unresolved in both directions. It emits
  `ambiguous_state_coverage`, although the polygons have no positive-area overlap.
- The current projection produces an approximately `3.015e-8 m` transverse residual,
  exceeding the `1e-8 m` coincidence threshold. Changing the CRS text to full-precision
  numbers alone did not repair the probe. A process-local direct AEQD transformer
  construction classified the same path correctly in both directions.

**Resolution design:** make the clipping projection retain its intended center and
ellipsoid precision, with explicit checks at the source origin and known radial
points. Evaluate the direct transformer construction demonstrated by the probe across
the full supported domain before adopting it. Verify shared ownership against common
boundary geometry with an explicit numerical error bound. Numerical uncertainty
should receive its own diagnostic; multiple midpoint memberships do not by themselves
prove positive-area polygon overlap.

Regression coverage must include non-round longitudes/latitudes, both directions,
redundant vertices, state/source ordering, short and long shared intervals, and nearby
parallel lines that genuinely belong to one state. Preserve real short crossings;
increasing a general snapping tolerance is not an acceptable substitute for this fix.

### SB-02 — duplicate representations of an endpoint create unresolved tails

Sources: event/station construction in
[partition.py](src/pipeline_calculator/core/geography/partition.py) near lines 103–107
and 155–174; unresolved status propagation in
[state_analysis.py](src/pipeline_calculator/core/state_analysis.py) near line 237.

Real bundled boundary reproduction:

```text
(-103.064732, 32.744215) -> (-103.064732, 32.75427)
Shared Texas / New Mexico boundary
Original length: 1115.0995456325722 m
```

- Expected: the whole path is shared; state allocations total the original length
  and reversing the path preserves completion status.
- Actual in one direction: shared `1115.099545632572 m` plus unresolved
  `2.2737367544323206e-13 m`. Reversal has no unresolved portion. The tiny residual
  still marks geography and every state incomplete and triggers a misleading
  positive-area-overlap diagnostic.
- A deterministic probe of 500 sampled real shared-boundary directions found 53
  cases with unresolved tails. This is a characterization of that probe set, not a
  failure-rate estimate for arbitrary user inputs.

**Resolution design:** maintain canonical source stations and carry endpoint identity
through intersection/refinement. When two distances represent the same proven source
endpoint, use that endpoint's canonical station. Audit every accepted equivalence
against the path and boundary model. Preserve positive intervals arising from distinct
crossing events, regardless of length. Do not globally discard short fragments,
redistribute discrepancies, or merely suppress the incomplete status.

Add the real TX/NM segment plus the minimal synthetic millimeter shared case to
regression tests. Test reversal, redundant vertices, chunk boundaries, very short
real crossings and per-source ownership/physical-coordinate conservation together.

## 4. Map and package fixes

### SB-03 — antimeridian corridors abort the default package export

Sources: [geography_kmz.py](src/pipeline_calculator/export/geography_kmz.py), polygon
validation near lines 60–66 and `_section_polygons` near lines 75–82;
[package.py](src/pipeline_calculator/export/package.py).

End-to-end reproduction from an ordinary KML, using all default analysis parameters
and State breakdown ON:

```text
Pipeline A: (179.997, 52.0)    -> (-179.997, 52.0)
Pipeline B: (179.997, 52.0001) -> (-179.997, 52.0001)
```

Combined original length is **824.1352745413919 m**, one overlap qualifies, geography
is complete, and reconciliation difference is zero. These particular coordinates
are outside state coverage, which is a supported export scope. With maps enabled,
export raises `ValueError: Clipped corridor polygon is invalid` and publishes no
package. With maps disabled, XLSX/JSON export succeeds. Canonicalizing the same
corridor yields two valid polygon parts; source-line reimport preserves mileage.

**Resolution design:** use one antimeridian-aware polygon normalization path before
planar validity/containment checks and serialization. Preserve outer rings, holes and
multipart geometry; unwrap and split before treating longitude rings as planar
polygons. Apply the same geometry rules to Combined maps, state clipping and previews.
Keep file-system failures fatal and staging atomic. If a visualization is genuinely
unusable, omit that visualization with a diagnostic while retaining the report and
source lines, as specified in the approved plan.

Regression tests must run this actual KML through analysis and default package
export, verify workbook/JSON/maps exist, inspect polygon topology, and reimport the
line mileage. Add an Alaska-scoped case with injected dateline-spanning state geometry,
as well as holes/multipart and a real Alaska fixture when available. Assert no
globe-spanning polygon/line and no duplicate corridor centerline is introduced.

### SB-04 — fallback selection and omission diagnostics differ across scopes

Sources: [corridors.py](src/pipeline_calculator/core/geography/corridors.py) near lines
21–28; [geometry_validation.py](src/pipeline_calculator/export/geometry_validation.py);
[geography_kmz.py](src/pipeline_calculator/export/geography_kmz.py) near lines 81–82
and 168–173.

Confirmed injected-result cases:

- An invalid preferred corridor polygon with a valid oriented rectangle has a
  usable Combined fallback but no state polygon.
- A section with only a valid bounding box produces a Combined rectangle but no
  state polygon.
- A Combined section with no usable geometry can be silently skipped. The exported
  workbook has no Diagnostics sheet and the result has no omission diagnostic.

These establish contract gaps; the fallback/omission probes deliberately construct
malformed or partial visualization records. The antimeridian failure in SB-03 comes
from a normal input and does not depend on injected results.

**Resolution design:** centralize representation selection and validation so both
scopes try the supported representations in the same order. Construct the selected
fallback, normalize it, then clip it to the state. An authoritative empty
`clipped_polygons` result must continue to prohibit an uncut export fallback. Return
structured diagnostics with the visualization decision, and include those diagnostics
consistently in the UI, workbook and optional JSON before publishing the package.
Keep the completed analysis snapshot immutable; use an export snapshot if export-only
validation adds diagnostics. Mileage and state savings must remain unchanged.

Tests should cover invalid preferred/valid oriented, bbox-only, invalid-all, a valid
shape clipped completely outside the state, multipart/holes, and one omitted corridor
among several valid ones. Confirm each state polygon remains contained and that
missing visuals are explained without aborting otherwise usable exports.

## 5. Progress and integration

### SB-05 — callback progress does not cover geography

Sources: [analyzer.py](src/pipeline_calculator/core/analyzer.py) near lines 181–189,
[overlap.py](src/pipeline_calculator/core/overlap.py) near line 537,
and [execution.py](src/pipeline_calculator/core/execution.py).

`analyze_complete(..., progress_callback=callback, options=AnalysisOptions(True))`
emits `[0.5, 0.625, 1.0]` before partitioning starts. The callback receives no further
state-analysis updates. Desktop progress uses `ExecutionContext` and is unaffected.

**Resolution:** adapt the public callback to the same aggregate stage schedule used
by the desktop job. Preserve mode-OFF compatibility. Test callback-only and
callback-plus-context calls, monotonic progress, no completion before the full result
snapshot, unavailable/incomplete geography, and cancellation without completion.

Additional integration coverage to add:

- Promote the audit's successful cancellation/retry probes at boundary loading,
  splitting, each state and finalization into committed regression tests.
- Exercise persisted options through Browse, drag/drop, retry and Apply/Cancel
  reanalysis in both entrypoints. Existing tests cover components, but not every
  complete route.
- Re-run native UI tests after the concurrent UI task settles, including small
  windows, high DPI, keyboard state selection, unsupported polygon-only input,
  shared allocations and unavailable adjusted mileage.

## 6. Execution order and conflict controls

1. **Wait for the other task to finish.** Inspect its status and final changes, then
   record the exact new baseline. Preserve its uncommitted work; a clean tree must
   not be obtained by discarding changes. Do not restore these older snapshot files
   over the current checkout.
2. **Reproduce against the final baseline.** Re-run the minimal cases below and
   mark any findings already resolved by the other task. Compare its final changes
   with the proposed file ownership before edits begin. Use a separate worktree from
   the agreed integrated baseline if parallel work resumes.
3. **Fix shared-border correctness (SB-01/02)** in the geometry core with focused
   tests. Treat these as one owned work unit so projection and station policies remain
   consistent. Require equal allocation and direction invariance, not conservation
   alone.
4. **Unify corridor handling (SB-03/04)** across core/export with a small shared
   geometry utility, avoiding a core-to-export dependency. Coordinate this unit with
   any finished corridor-dialog or export-action edits. Add end-to-end package tests.
5. **Fix callback aggregation (SB-05)** and commit the cancellation/retry tests.
6. **Add fixture regressions.** Both the polygon-only negative regression and the real
   positive disconnected-centerline regression can proceed after the integration gate.
   Use the saved independent expectations, preserving synthetic cases for shared
   borders and qualifying overlaps that the real fixture does not exercise.
7. **Integrate and verify.** Run targeted tests after each unit, then the full suite
   once the integrated changes settle. Inspect a real sample export, shared-border
   exports, antimeridian maps, and the affected UI states. Build and run offline frozen
   smoke checks in both GUI modes on Windows and macOS; native macOS verification
   remains an outstanding release gate from the original delivery.

No automatic implementation, cross-task messages, remote pushes or release actions
were scheduled by this audit. This document defines the deferred work, not approval
to overwrite another task's changes.

## 7. Evidence, passed checks and limits

Audit data is summarized in
[state-boundary-followup-audit.json](docs/validation/state-boundary-followup-audit.json).
Detailed local probes use the immutable snapshot under
`.validation-output/state-boundary-audit-a8640fd/baseline/`.

| Evidence | Location under `.validation-output/state-boundary-audit-a8640fd/` |
| --- | --- |
| Input XML, parser and polygon coverage | `fixture-evidence.json`, `polygon-fixture-evidence.json` |
| Centerline independent expectations and complete application/export checks | `centerline-audit/independent.json`, `centerline-audit/application.json` |
| Projection precision reproduction | `partition-audit/shared_projection_variants.py` and `.json` |
| Real shared-boundary endpoint cases | `partition-audit/real_shared_probe.py` and `.json` |
| Boundary topology check | `partition-audit/topology_probe.json` |
| Minimal executable geometry reproductions | `partition-audit/minimal_findings.py` and `minimal_findings.json` |
| Default-parameter antimeridian input/export | `export-audit/antimeridian_parallel.kml`, `probe_dateline_end_to_end.py`, `antimeridian_end_to_end.json` |
| Fallback and omitted-visual probes | `export-audit/probe_exports.py`, `probe_results.json` |
| Progress and five-stage cancellation/retry | `contract-audit/contract-probe.json` |
| Existing test results | `contract-audit/contract-tests.xml`, `export-audit/existing-export-tests.xml`, `partition-audit/geometry-tests.xml` |

The audit passed 20 existing geometry tests, 38 existing headless contract tests,
and 10 existing export tests. Five native GUI cases were deliberately excluded from
the contract run and one from the export run while the other task edited UI code.
Fresh cancellation/retry probes passed at every tested
geography stage. Checks of the bundled canonical polygons found no positive-area
overlap between states; this does not certify surveyed ownership or eliminate
numerical classification errors.

No duplicate source mileage or conservation loss was demonstrated in these probes.
The confirmed failures instead affect shared ownership/completion, visualization
handling, package delivery and callback progress. Passing the old tests is therefore
insufficient for the additional acceptance scenarios above.

The original polygon archive cannot establish pipeline-mileage correctness. The new
centerline archive verifies positive disconnected-state attribution and mileage, but
does not exercise shared borders or qualifying state overlaps. Its upstream centerline
generation method was not audited. Native macOS operation and the other task's unfinished
GUI changes were not validated by this audit. The existing boundary provenance and
source-accuracy limitations still apply; this plan makes no new surveyed-accuracy claim.
