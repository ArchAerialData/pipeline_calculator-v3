# State boundary analysis: audit and resolution plan

Status, September 15, 2026: **SB-01 through SB-05 and fixture regressions implemented;
437 tests pass. RV-01's deadline mismatch is resolved; its historical native crash,
strict performance evidence and native macOS verification remain open.** Current outcomes are recorded in
[the resolution validation report](docs/validation/state-boundary-resolution.md).
The findings and readiness evidence below describe the pre-fix baseline.

Current implementation baseline: `6bc6e5f7aa083604edbaa25f177f671f58ea15d6`
(`UI Fixes 2`). The worktree was clean at the start of the readiness review and the
other task's changes were committed. Fresh probes of that commit confirmed all five
original findings. At that point the geometry, parser and core export files were
unchanged from audit baseline `a8640fd783e938cf7ea243196dc97bd82f608091`.

The earlier audit used an isolated copy while GUI work was in progress. Its evidence
remains below; the new readiness results and integration constraints supersede its
concurrency hold. No additional product decision or missing fixture blocks starting
the documented fixes. RV-01 and native macOS checks block a clean integrated completion
claim, not independent work on the geometry/export core.

## 1. Findings at a glance

| ID | Priority | Confirmed behavior | Impact |
| --- | --- | --- | --- |
| SB-01 | P2 | A precisely shared 364.439 m border path is classified entirely unresolved because projection roundoff exceeds the coincidence test. | Shared allocation is unavailable for valid shared geometry. |
| SB-02 | P2 | A real Texas/New Mexico shared path produces an unresolved floating-point tail in one direction only. | Otherwise correct state analysis is marked incomplete; direction invariance fails. |
| SB-03 | P2 | An ordinary antimeridian input analyzes successfully but package export with maps aborts. | The default export choice prevents delivery of the entire package, including its workbook. |
| SB-04 | P2 | State corridor clipping omits valid fallback shapes; unusable Combined corridors can be omitted without a diagnostic. | Missing visualizations and inconsistent explanations between scopes. |
| SB-05 | P2 | The public float progress callback reaches 100% before geography starts and emits no geography updates. | API callers see premature completion; desktop context-based progress is unaffected. |
| FX-01 | Resolved | The initial archive contains polygons; the subsequently supplied Centerlines KMZ contains valid line geometry. | Both negative and positive regression fixtures are now available. |
| RV-01 | Validation blocker | The current 100-cycle scope-replacement test crashed on the first run and timed out on an isolated repeat. Root cause is not established. | Diagnose the native failure before claiming integrated UI/release readiness. |

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

Proposed automated fixture tests (the fixtures and expectations are ready):

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
- Re-run native UI tests after RV-01 is resolved, including small
  windows, high DPI, keyboard state selection, unsupported polygon-only input,
  shared allocations and unavailable adjusted mileage.

### RV-01 — current native scope-switching validation fails

Current test: [test_ui_lifecycle.py](tests/test_ui_lifecycle.py),
`test_scope_replacement_releases_views_and_recovers_from_failure` near line 192.
It checks a failed-render recovery followed by 100 Combined/Texas scope replacements,
retired-view cleanup, bindings/styles and event-loop responsiveness.

On the current clean commit:

- A selected UI run had **15 passes and one failure**. This test's child process
  exited with Windows access violation `3221225477` (`0xC0000005`). The diagnostic
  trace was in Tk canvas coordinates / CustomTkinter rendering.
- A repeat of that test alone, with the fixture analysis already finished, also
  failed: its native child exceeded the existing **45-second timeout**. The 20-second
  diagnostic trace was in Tk canvas text creation and CTk frame/scrollbar drawing,
  including `update_idletasks`, while the scope-replacement loop called `root.update()`.
- These are observed failures of the native validation path. They do not establish
  which application, dependency, test-harness or environment behavior caused them.
  The two failure forms should both be retained during investigation.

**Investigation/resolution plan:** minimize the sequence while recording completed
scope cycles, per-cycle duration, Configure/idle callback activity, widget lifetimes,
resource counts and the exact Tk/CustomTkinter environment. Compare the isolated test
with an equivalent bounded normal-event-loop probe. Preserve the new transactional
replacement and callback-ownership contracts. Fix any identified reentrant redraw,
lifetime or harness defect at its source; a timeout increase or skipped test alone
does not explain the observed access violation. Do not infer a specific cause from
the stack trace without a reproduction that distinguishes it.

Acceptance: repeated isolated runs of the unchanged 100-cycle correctness scenario,
the neighboring lifecycle tests, and the full suite must complete with no native crash,
hang, callback error or retained retired view. Then run both packaged GUI modes with
the existing stderr/callback-error checks. Keep strict performance measurements
separate from the correctness run, as specified by the UI task's validation report.

## 6. Execution order and integration controls

1. **Completed — handoff and baseline verification.** The other task is finished,
   its changes are committed at `6bc6e5f`, and the worktree started clean. Do not
   restore older snapshot files over the committed UI improvements. Check for newer
   edits again when implementation starts; a new task could change this condition.
2. **Completed — original failure recheck.** SB-01 through SB-05 still reproduce
   against current imports, and both fixture checks pass. Add RV-01 investigation
   to the work plan now. It can run independently of the core fixes with distinct
   file ownership. Use a separate worktree from the agreed baseline if parallel
   work resumes.
3. **Fix shared-border correctness (SB-01/02)** in the geometry core with focused
   tests. Treat these as one owned work unit so projection and station policies remain
   consistent. Require equal allocation and direction invariance, not conservation
   alone.
4. **Unify corridor handling (SB-03/04)** across core/export with a small shared
   geometry utility, avoiding a core-to-export dependency. Preserve the committed
   corridor-dialog and background-export behavior described below. Add end-to-end package tests.
5. **Fix callback aggregation (SB-05)** and commit the cancellation/retry tests.
6. **Add fixture regressions.** Both the polygon-only negative regression and the real
   positive disconnected-centerline regression can proceed after the integration gate.
   Use the saved independent expectations, preserving synthetic cases for shared
   borders and qualifying overlaps that the real fixture does not exercise.
7. **Integrate and verify, including RV-01.** Run targeted tests after each unit, then the full suite
   once the integrated changes settle. Inspect a real sample export, shared-border
   exports, antimeridian maps, and the affected UI states. Build and run offline frozen
   smoke checks in both GUI modes on Windows and macOS; native macOS verification
   remains an outstanding release gate from the original delivery. Neither the
   earlier passing Windows report nor this recheck establishes macOS behavior.

### Contracts inherited from the completed UI task

These are integration requirements, not requests for another GUI redesign:

| Surface | Preserve during the fixes |
| --- | --- |
| [Results scope selection](src/pipeline_calculator/gui/pages/results_page.py) | Prepare a replacement before removing the previous valid view; keep Retry display and selector/view agreement after failure. Same-scope selection remains a no-op. Keep the complete analysis snapshot available for export. |
| [Background actions](src/pipeline_calculator/gui/background_action.py) and [export actions](src/pipeline_calculator/gui/actions/export_actions.py) | Ordinary export stays off the Tk thread. Package export retains its existing worker, captured options, retry handling and atomic publication. Workers return plain data and never address destroyed widgets. Put visualization preflight in the pure package/geometry layer. |
| [Tables](src/pipeline_calculator/gui/table_loading.py) | Preserve bounded row batches, pause/resume while hidden, callback cancellation on destruction, and full-precision sorting. New state/diagnostic rows must not restore synchronous bulk insertion. UI assertions must wait for loading completion instead of assuming every row exists immediately. |
| [Parameter dialog](src/pipeline_calculator/gui/dialogs/params_dialog.py) | Commit preference drafts only on Apply, preserve Cancel behavior, and keep parameter-transition errors visible. |
| [Bindings](src/pipeline_calculator/gui/bindings.py), [scrolling](src/pipeline_calculator/gui/scrolling.py), [styles](src/pipeline_calculator/gui/styles.py) and [disclosures](src/pipeline_calculator/gui/disclosure.py) | Keep owned subscriptions, deferred/cancelled redraw work, size-keyed shared styles, keyboard behavior and lazy details. Preserve per-popup styling without changing other windows. |
| [Dependencies](requirements.txt) and [packaged checks](scripts/validation/check_packaged_smoke.py) | Keep the tested CustomTkinter 5.2.2 pin unless a separately justified dependency change is needed. Preserve repeated Summary/disclosure smoke cycles and rejection of Tcl callback-error stderr. |

SB-01/02 should primarily own `core/geography/partition.py` and focused geometry tests.
SB-03/04 should primarily own shared polygon normalization, `core/geography/corridors.py`,
the pure export modules and their tests. SB-05 should own analyzer/execution progress
and its tests. Keep RV-01's UI/test-harness investigation separately owned. Changes to
shared files such as `state_analysis.py` or `execution.py` must be integrated serially.

Build a separate export diagnostic snapshot before workbook/JSON/map generation if
preflight adds warnings; do not mutate the displayed analysis from a worker. The two
export worker paths need no consolidation to fix SB-03/04.

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
generation method was not audited. Native macOS operation was not validated. The
other task's now-committed GUI changes received the readiness checks below. The existing boundary provenance and
source-accuracy limitations still apply; this plan makes no new surveyed-accuracy claim.

## 8. September 15 readiness evidence

[Readiness summary](docs/validation/state-boundary-readiness.json) records the exact
commit, fresh outcomes, environment and artifact references. Local detailed evidence
is under `.validation-output/state-boundary-readiness-6bc6e5f/`.

| Check | Current result |
| --- | --- |
| Conflict/integration baseline | Clean initial worktree; finished work committed at `6bc6e5f`; geometry, parser and core export unchanged since the audit. |
| Original findings | All five reproduced against current repository imports. Keep their fixes in the plan. |
| Geometry tests | 20 passed. |
| Headless export tests | 10 passed; one native test intentionally deselected. |
| Scoped UI/state integration | 15 passed, one native stress failure; isolated repeat of the failed test also failed. See RV-01. |
| Polygon fixture | Hash and tracked-file checks passed; zero pipeline mileage and incomplete status verified with State breakdown OFF and ON. |
| Centerline fixture | All 281 per-source expectations / 4,827 paths matched; full analysis complete, zero crossings/unassigned mileage, totals reconciled, all four KMZ mileage round trips passed. |
| Earlier UI validation | Retained report records 366 passes and both Windows packaged modes passing. All 26 recorded source hashes match this commit. These historical passes do not override the fresh RV-01 failures. |
| Native macOS | Still unverified; completion/release gate, not a prerequisite to starting the independent core fixes. |

The real fixtures and expectation manifests are now committed, but no automated
pytest test currently consumes them. Adding those regressions remains implementation
work, not a missing-input blocker. No full-suite or package rebuild was repeated in
this readiness review after the native failures; diagnose them before broad validation.

Readiness verdict: **the concurrency hold is removed and the technical fix plan is
actionable without further product decisions. Add RV-01 to implementation and require
it to pass before an unqualified integrated completion claim.**

## 9. Implementation outcome

SB-01 through SB-05 are implemented, with the real polygon-only and disconnected
centerline fixtures now protected by automated regression tests. The full suite passes
437 tests; both freshly packaged Windows GUI modes pass offline smoke checks. Four
sample export packages were inspected, including nine successful map mileage round
trips. Code and evidence remain uncommitted for review.

RV-01's confirmed whole-test deadline mismatch is resolved in the test harness.
Repeated 100-cycle acceptance runs and the full suite pass with every original
ownership/liveness assertion retained. The historical intermittent Windows access
violation remains unexplained; no production crash fix is claimed. Strict reference
performance and native macOS operation remain outstanding release evidence.

The [validation report](docs/validation/state-boundary-resolution.md) and
[machine-readable evidence](docs/validation/state-boundary-resolution.json) record
the exact results, source/artifact hashes, sample outputs, remaining limits and next
verification steps. The pre-fix findings and readiness sections above are retained
as the audit history.
