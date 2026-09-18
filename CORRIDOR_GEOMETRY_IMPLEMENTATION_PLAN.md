# Corridor geometry implementation runbook

**Status: Implemented, verified and published locally on Windows, September 18, 2026.**
The macOS distribution gate remains open. Evidence and platform limits are tracked in
[the implementation report](docs/validation/corridor-buffer-implementation.md).
Prepared September 18, 2026 against repository commit
`552191dac8e723d32819e38f6c1647d039d6038b`.

## 1. Outcome and scope

Replace the broad rectangles currently used for some bent overlap sections with
polygons that follow the actual qualifying pipeline paths. Apply the same
construction to ordinary analysis, Combined results and individual states.
Keep all original mileage, overlap qualification, savings, state attribution and
reconciliation calculations unchanged.

This runbook covers implementation order, data contracts, geometry construction,
failure handling, UI/export migration, acceptance tests and local distribution.
The subsequent user instruction authorized implementation. No new product
decision is required to start the ordered implementation below; the defaults are
explicit and their visual review is a release gate.

### Decisions for the first release

| Topic | Decision |
| --- | --- |
| Meaning of a corridor | Approximate display area around the original path portions that qualified for overlap. It is not a surveyed right-of-way or an optimized flight route. |
| Padding | **5 meters outward around each qualifying path**, with round joins and round endpoint caps. This matches the feasibility probe and the old nominal 10 m total outer margin. |
| Relationship to detection range | Independent. Detection range decides which lines qualify; padding only controls map appearance. Do not infer one from the other. |
| Width wording | “5 m padding around qualifying paths,” never “10 m total corridor width”: paired paths add their separation to the overall extent. |
| Controls | No additional input-page toggle or numeric field in this release. Use immutable internal display options; disclose the value in existing details and map descriptions. |
| Separate pieces | Preserve them. A qualifying pair more than 10 m apart can have two separate buffered pieces. Do not widen the buffer to join them. |
| Holes | Preserve all holes remaining after the true buffer union. Small voids can naturally close when buffers overlap; do not explicitly fill interiors. |
| Endpoints | Round caps extend 5 m beyond qualified endpoints as display padding. This never adds qualifying mileage or includes trailing remainders as source runs. |
| Across sections | Keep each qualifying pair/section identifiable. Do not dissolve the entire network into one anonymous polygon or change bundled-section counts. |
| Failure | Keep numerical results; mark that section's map unavailable with a diagnostic. No rectangle/hull fallback for newly generated buffered sections. |
| State views | Construct from state-exclusive qualifying paths, then clip the completed polygons to the state. Shared-border allocations cannot generate state corridors. |
| Approximation | Target at most **0.05 m geometric approximation error** relative to the defined 5 m geodesic neighborhood, subject to the verified projection domain. This is separate from state partition precision and real-world boundary accuracy. |

Natural overlap of two buffers may merge their display areas, including across a
small gap. That is allowed by the fixed-distance definition; it does not join
analysis paths. The prohibited behavior is adding connecting linework, filling a
void beyond the buffer distance, or increasing the radius to force connectivity.

## 2. Verified baseline and design sources

The reference launcher is actually `run_jpg_boundary_gui.py` at the root of
`C:/Users/rbake/Desktop/VS Code Shortcuts/Create-Polygon-Border-From-JPGs`.
Its useful primitives are in
[polyline_boundary.py](<C:/Users/rbake/Desktop/VS Code Shortcuts/Create-Polygon-Border-From-JPGs/src/polyline_boundary.py>)
and
[geometry_hardening.py](<C:/Users/rbake/Desktop/VS Code Shortcuts/Create-Polygon-Border-From-JPGs/src/geometry_hardening.py>).
Adapt the fixed-distance buffer, coordinate transformation and coverage-check
ideas into this repository. Do not create a runtime dependency on that checkout,
its launcher, photo-reading dependencies or GUI.

The reference checkout has uncommitted changes. Its HEAD alone is not a complete
version identifier; the retained evidence includes exact hashes of the inspected
source files. Recheck those before adapting code if the reference project changes.

Do not copy the reference application's single-polygon requirement, hole filling,
automatic connection search (up to 2,000 ft), mean-longitude UTM selection, camera
point hulls, or output simplification allowing 5 ft deviation/1% area change.
Its 10 m output densification is not proof that sparse input geodesics are modeled
correctly. Its generic `make_valid`/`buffer(0)` cleanup is not a substitute for
verifying our constructed shape.

Evidence collected before implementation:

- Straight/L/U feasibility probes produced valid reference-buffer polygons
  containing every qualifying source path. L/U cases selected rectangles in the
  current application. These are construction probes, not complete acceptance.
- Reference polyline and geometry-hardening suites: **15 tests passed** during
  the reference review.
- Current repository corridor, state geometry and export baseline:
  **61 passed, 5 native GUI tests deselected**, 4.89 seconds on September 18.
- The diagram and numeric observations are retained in
  [feasibility evidence](docs/validation/corridor-buffer-feasibility/README.md).

Primary library references: Shapely's
[buffer contract](https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html)
defines a radius around the input geometry, with explicit cap/join and arc
resolution settings. Its
[make_valid documentation](https://shapely.readthedocs.io/en/stable/reference/shapely.make_valid.html)
shows that validation repair can produce collections or lower-dimensional output.
PROJ documents ellipsoidal forward/inverse
[azimuthal equidistant projection](https://proj.org/en/stable/operations/projections/aeqd.html).
The proposed numerical limits below are application requirements, not guarantees
provided by those documentation pages.

## 3. Repository findings that dictate task order

| Finding | Consequence and required change |
| --- | --- |
| [overlap.py](src/pipeline_calculator/core/overlap.py) builds display shapes before recording bundled sections and segment coverage; geometry exceptions can propagate into an overlap failure in [analyzer.py](src/pipeline_calculator/core/analyzer.py). | Separate completed statistics from optional visualization first. A polygon problem must not erase savings. |
| [bundling.py](src/pipeline_calculator/core/bundling.py) already provides full segment-ID sets, path indices and representative matches. | Use full qualified segment sets for geometry. Representatives are insufficient evidence of complete path coverage. |
| [corridor_coverage.py](src/pipeline_calculator/core/corridor_coverage.py) retains original vertices and geodesic chainage, but current callers use one min/max span. | Add explicit consecutive-run extraction; do not rely on a hidden contiguity assumption or the clamping in `MeasuredPath.point`. |
| [corridor_kml.py](src/pipeline_calculator/export/corridor_kml.py) uses the single-ring selector in ordinary mode; it ignores canonical polygons and can reconstruct a bbox even when canonical geometry is omitted. | Unify ordinary/Combined/state serialization before activating multipart output. Explicit empty/omitted geometry must remain authoritative. |
| [corridor_geometry.py](src/pipeline_calculator/core/corridor_geometry.py) and [geography_kmz.py](src/pipeline_calculator/export/geography_kmz.py) already support polygon lists, holes and clipped geometry. | Extend this common contract instead of building a second representation for previews. |
| Ordinary preview currently rounds to seven decimals and adds a center Point; its bbox center may fall outside a U shape or hole. | Use full-precision polygon serialization and polygon-only previews for the new representation. Keep old result compatibility explicit. |
| Ordinary analysis does not finalize all visualization decisions at the same stage as state analysis. | Publish ready/omitted decisions and warnings for both modes before the result snapshot reaches the UI. Export must not be the first discovery of a normal construction failure. |
| [xlsx.py](src/pipeline_calculator/export/xlsx.py) exports legacy ring/width columns. | Preserve established numerical columns/formulas while making multipart availability explicit; never silently serialize only the largest polygon. |
| [build_gallery.py](scripts/validation/build_gallery.py) and [audit_dataset.py](scripts/validation/audit_dataset.py) inspect single polygons/rings and legacy labels. | Upgrade verification tools before accepting new geometry; otherwise they can overlook missing components or holes. |

## 4. Contracts and ownership

### Numerical contract

Qualification, continuous minimum length, sampled trailing-remainder behavior,
compatible multi-pipeline grouping and all savings arithmetic remain unchanged.
Do not modify the meaning of `qualified_segment_coverage_v2` or
`state_interior_qualified_segment_coverage_v1` for a display-only change.

First assemble complete numeric section records, per-pipeline coverage and total
savings. Only then attach display results. Preserve source IDs, section count,
section ordering/tie behavior, pair identity, qualifying segment sets, original
meters, effective meters, savings, state ledgers, crossing counts and completion
status. Geometry warnings do not set `analysis_complete=False`.

Catch expected projection, topology, validation and budget failures inside the
visualization boundary, with structured diagnostics. An unexpected visualization
exception must be recorded as a map-generation error with support details and
preserved numeric results, rather than a successful-looking rectangle. Do not
catch `BaseException`; propagate `AnalysisCancelled` explicitly.

Statistical parity applies to identical inputs/settings whose baseline numerical
calculation succeeded. A legacy display failure that incorrectly erased overlap
results is an intentional behavior fix: the new run must recover the independently
expected numbers and report only the map failure. Genuine qualification/accounting
failures must retain their existing unavailable-result behavior. Record these
cases explicitly; do not hide differences under broad comparator exclusions.

### Proposed implementation surface

- New `core/corridor_buffer.py`: `CorridorDisplayOptions`, `CorridorGeometryBudget`,
  bounded projection/densification, buffering and canonical result construction.
- Extend `core/corridor_coverage.py`: immutable `QualifiedPathRun` and
  `qualified_path_runs(...)`, using `MeasuredPath` without changing its existing
  callers' numerical behavior.
- `QualifiedPathRun` contains scope, application-assigned source ID, scope path
  index, consecutive sample range, start/end chainage and immutable coordinates.
  In state inputs, path index refers to that state's fragment path list; do not
  label it as the original document path index without an explicit mapping.
  When a single-state result reuses Combined geometry, the recorded generating
  scope remains `Combined`: zero-length paths can disappear from state inputs
  and shift their indices. The enclosing state metadata identifies the clipping
  destination; it does not relabel the generating run's source indices.
- `build_buffered_corridor(runs, *, geod, options, budget, context)` returns one
  JSON-native visualization decision, separate from numeric section fields.
- One geometry budget is shared across Combined and all states for the job; one
  pathological section also has its own sublimit. Pass dependencies explicitly,
  not through global mutable caches or attributes leaking between jobs.

New sections add the following fields to the existing numeric section record:

```text
visualization_schema_version: 1
visualization_kind: "qualified_path_buffer"
visualization_status: "ready" | "omitted"
visualization_polygons: [{outer: [[lon, lat], ...], holes: [[[lon, lat], ...], ...]}]
visualization_metadata:
  policy: "qualified_path_buffer_v1"
  padding_m: 5.0
  cap_style: "round"
  join_style: "round"
  approximation_target_m: 0.05
  source_runs: [{source_id, scope_path_index, start_m, end_m}, ...]
  part_count, hole_count, vertex_count, chart_count
diagnostics: [...]
```

State sections retain `clipped_polygons` as the authoritative scoped shape for
existing consumers. Metadata also records scope/state and clipping status. A
normalization adapter may expose these as `visualization_polygons`, but must not
rebuild an uncut shape. An explicitly empty canonical list, omitted status, or
invalid canonical representation blocks all legacy fallback selection. Unknown
future schema versions fail visibly rather than being interpreted as legacy.

For new output, legacy `corridor_polygon` may contain the same ring only when the
complete result is exactly one polygon with no holes; otherwise leave it empty.
Do not put a largest-component approximation there. Do not populate
`oriented_polygon` or `oriented_width_m` with invented equivalents. Consumers must
prefer the canonical contract whenever its version/key/status is present.
Legacy dictionaries lacking all canonical fields retain their current validated
candidate-selection behavior, clearly described as legacy approximations.

No Shapely objects, NumPy scalars, infinities or NaNs may enter public results.
Copy changed containers; do not mutate the source coordinates or a previously
published snapshot. Pin internal display options at worker creation for retry,
reanalysis and all states in that job.

## 5. Geometry construction procedure

### A. Extract the exact generating linework

1. Obtain each section's complete `qualified['segment_ids']` and path pair.
   Resolve IDs against the correct pipeline and scope path.
2. Sort and deduplicate `path_segment_index` values independently per path;
   split them into maximal consecutive runs. Reject missing/out-of-path IDs.
3. For run `i..j` and analysis step `L`, use chainage `[i*L, (j+1)*L]`.
   Check finite values, monotonicity and bounds before calling `MeasuredPath.span`.
   Terminal clamping is allowed only within the segmentation implementation's
   existing roundoff allowance, with a numerical regression proving no extra tail.
4. Retain every original vertex inside the interval; calculate endpoints on their
   original geodesic edges. Consecutive duplicate coordinates may be removed from
   a construction copy only. Reject a run that has no positive measured extent.
5. Never concatenate disjoint runs, different coordinate paths, state exit/reentry
   pieces or source pipelines into a new LineString. Preserve identity even when
   names, XML IDs or OBJECTIDs repeat.
6. Buffer both participants' full qualified runs. Keep their possibly unequal
   extents: section reporting uses the existing qualifying length rules and must
   not be redefined from the footprint or its perimeter.

### B. Work in bounded local metric charts

Use GRS80 geodesic forward/inverse operations consistently with current source
measurement. Use a small isolated ellipsoidal AEQD utility, or extract a shared
projection primitive only if all existing state-partition tests remain unchanged.
Do not import private state-partition internals into the new visual module.

- Partition long runs deterministically into chunks of at most **20 km chainage**.
  Choose local origins from each chunk's geodesic midpoint, using longitude
  unwrapping rather than an arithmetic mean across +180/-180.
- Require every construction vertex, padding extent and validation probe to lie
  within **25 km** of the chart origin. Check local distortion; subdivide further
  if the allocated projection error cannot be met. These are ceilings, not a
  blanket claim that every configuration inside them is already accurate.
- Densify the *original geodesic edges* before projection: retain native bends,
  cap subedges at 100 m, and recursively check quarter/mid/three-quarter points
  against the projected chord until source deviation is at most 0.005 m.
  Bound recursion and total generated points. Linear interpolation of latitude
  and longitude is not an acceptable substitute.
- Validate finite forward/inverse transforms and their geodesic roundtrip errors.
  Use explicit axis order and offline resources; do not silently change datum or
  require downloading a projection grid.
- Ordinary Alaska, Hawaii, multizone routes and antimeridian crossings must pass.
  A geometry winding around the globe, exact pole ambiguity or unsupported chart
  domain may omit the map with an explanation; numeric analysis remains available.

### C. Buffer and union

For each qualified run/chunk, construct a metric LineString and use explicit
`buffer(radius, cap_style='round', join_style='round', quad_segs=...)`. Buffer
continuous runs/chunks, not one shape for every sampled analysis segment.

Derive arc resolution from the allocated sagitta error:
`r * (1 - cos(pi / (4*q))) <= 0.005 m`, where `q` is segments per quadrant.
Use keyword arguments supported by the declared Shapely version range. Validate
finite positive radius, and limit arc resolution before allocation.

Union bounded batches of these buffered geometries using deterministic order.
Keep all positive-area components and interiors. Separate sections keep separate
records even when their display polygons overlap. Do not dissolve source lines
before accounting, use a hull, auto-connect paths, fill holes, simplify, snap to a
precision grid, or silently repair invalid output with `make_valid`/`buffer(0)`.
An invalid result is a failed visualization unless an explicitly verified
construction retry at a smaller chart size succeeds within the same budget.

### D. Geographic conversion and chart seams

Inverse-project polygon rings with adaptive output-edge subdivision: verify the
coordinate-linear geographic edge against the intended inverse-projected metric
edge at multiple interior points, with maximum 100 m metric subedges. Preserve
holes and winding consistently. Split polygons at the antimeridian in unwrapped
space before global predicates/serialization; every final longitude is canonical.
Bound all intermediate pieces and never form a Greenwich-spanning union from a
dateline-crossing ring.

Merge chunk footprints in compatible longitude zones, then normalize the polygon
sets. Verify that artificial chart boundaries create no gaps or excess widening
beyond the error target. Round caps at internal cuts are allowed only as part of
the same fixed-radius neighborhood; they must not create an oversized bridge.
Require geometric equivalence under reversed path/source order and alternative
chunk boundaries within the tolerance. Byte-identical ring ordering across GEOS
versions is not required; stable sorting/winding/start-point normalization within
one environment is desirable for deterministic exports.

Those geometry invariance assertions assume equivalent extracted source support.
Whole-path reversal can move an unqualified short tail to the other end; compare
such full-analysis variants against their own expected qualified intervals.

### E. Validate coverage, tightness and state scope

Validity alone is insufficient. For every completed section verify:

- All generating runs, endpoints and original bends are covered by the candidate
  under the independently densified comparison model, not just their midpoints.
- No portion of the candidate lies outside the permitted 5 m neighborhood by more
  than 0.05 m; test boundary edges and interior area against an independent dense
  reference. A filled loop hole can pass an exterior-boundary test, so include
  polygon set-difference/void assertions.
- Require the full width as well as line coverage: for generating support `S`,
  radius `r` and the available tolerance `e`, demonstrate
  `B_(r-e)(S) subset_of candidate subset_of B_(r+e)(S)` with the independent
  reference's own approximation error charged to the budget. A narrow sliver
  enclosing the centerlines must fail. This two-sided check detects missing
  lobes, excessive contraction, chart seam gaps and spurious area.
- Every retained component lies within the permitted neighborhood of generating
  source support. It need not physically intersect a line after state clipping:
  an island or concave state cut may leave a valid disconnected piece of padding.
  No largest-part selection or silent removal of a tiny positive component or
  hole is allowed.
- Before state clipping the full buffer is checked. After clipping compare with
  that same buffer intersected with the chosen state boundary model. Truncation
  at a state edge is intentional and is not a tightness/width failure.
  Apply the inner/outer reference-neighborhood checks after intersecting each
  reference with the same state boundary, in both set-difference directions.
- Serialized state polygons must have **empty difference from the state boundary**
  in the existing coordinate-linear model. Do not borrow the 5 cm display budget
  to allow state leakage, expand the boundary, round clipped coordinates, or
  resurrect an uncut fallback.

Proposed approximation budget: source densification 0.005 m, arc approximation
0.005 m, inverse/output-edge approximation 0.005 m, projection scale effect 0.010 m,
serialization 0.005 m, chart union/seams 0.010 m; reserve 0.010 m. Demonstrate the
**combined 0.05 m bound**, not merely each component independently. An independent
dense reference must use finer spacing, independent distance checks and analytic
cases; calling the production helper twice is not verification. Do not conflate
this display budget with state-source partition location tolerance or source-data
positional accuracy.

## 6. Work limits, cancellation and error presentation

Centralize named limits with the builder's versioned policy. Initial ceilings:

| Limit | Initial ceiling / action |
| --- | --- |
| Source and densified construction points per section | 100,000 aggregate; stop before materializing more |
| Output points per section | Existing 100,000 ceiling, across all outer/inner rings |
| Parts and holes per section | 4,096 parts; 8,192 total rings; reject before unbounded normalization |
| Adaptive subdivision | Depth 20, checked together with point and chart budgets |
| Chunks/charts per section | 1,024 |
| One native buffer/union batch | At most 4,096 input vertices and 32 polygon operands, also checking intermediate output complexity |
| Cumulative job construction work | 2,000,000 processed/generated vertices and 10,000 charts across Combined and all states |
| Cumulative retained output | 1,000,000 polygon vertices across the complete result |
| Immutable boundary verification | Separately cap reference geometry at 2,000,000 vertices and 10,000 full-boundary queries per job. These checks certify containment against the original resource; generated geometry and local overlays retain the 4,096-vertex cap. |

These are conservative starting caps to verify with memory/time evidence, not
promised capacity. Count vertices across phases, retries, holes, intermediate
unions and repeated sections. Count single-state reused objects only once for
retained-output memory, but charge every operation performed. A section that
exceeds its limit is omitted as a whole; later independent sections may continue.
Job-budget exhaustion omits remaining map generation with one scope summary and
bounded per-section detail. Preserve all numerical section rows.

Native union/intersection calls also need input preflight and bounded scheduling;
merging a large accumulated operand is not magically bounded by a small batch of
new operands. Subdivide spatial work or stop when an operand exceeds the native
call cap. Add a bounded spatial-query path for state boundary pieces without
simplifying the full-detail boundary or changing its topology. Measure worst-case
accepted GEOS calls; Python cancellation cannot interrupt one already in progress.

Preflight **output expansion before native allocation**, too. Conservatively
estimate round-cap/join output from vertex count and selected `quad_segs`; estimate
overlay/intersection complexity from operand edges and candidate intersections.
Reserve that work/intermediate memory against the remaining section/job budgets
before the call. Reduce batches/chunks or omit when a defensible bound does not
fit; a size check only after GEOS returns is insufficient. Apply these limits to
validity, coverage/tightness and reference-comparison work as well as buffering.
Use bounded spatial indexes for candidate edges instead of unbounded all-pairs
distance/overlay checks.

Check cancellation before/after every native call and in loops at least every
256 items. Target cancellation response under 1 second on the representative
stress set, with retained timings. If a capped native operation defeats that gate,
reduce the cap/split work; use an isolated worker process only if necessary to
meet the bound. Do not declare cancellation bounded solely because checkpoints
exist. Cancel publishes no partial analysis snapshot.

Implementation review refined the state-clipping procedure: retain original
boundary-edge endpoints in local masks, since shortening edges at a query box
can perturb their represented slope. If GEOS leaves floating-point overlay
residue, permit at most eight further intersections with the same authoritative
mask, charged to the work budget. Each refinement must remain within one
nanodegree of the previous shape, with an area bound as well; the final outside
difference must still be empty. This is not permission to expand a state, snap
source geometry, or accept a nonempty outside sliver. The immutable-boundary
exception above is measured separately: all 51 bundled states were exercised,
and the largest observed native verification call took 5.21 ms in this environment.

Use one existing background analysis job and scoped progress. Add a stage such as
“Building corridor maps” with section progress inside its allocated scope. Worker
completion alone reports 100%; per-state or per-section completion cannot do so.
No geometry operations on the Tk thread or recalculation when switching views.

Update [core/progress.py](src/pipeline_calculator/core/progress.py) with the new
execution order. Currently corridors precede graph/savings and the latter reaches
99%; moving geometry later without reallocating stage shares would park the bar at
99%. Register the actual stage names, reserve a measured share after accounting,
and test meaningful monotonic updates in that share for ordinary and scoped state
jobs, late workload warnings, cancellation and finalization.

Messages:

- Normal details/map description: **“Approximate overlap area with 5 m padding
  around qualifying paths. Padding does not affect mileage.”**
- State addition: **“Clipped to [State].”**
- Failure row/action: **“Map unavailable.”**
- Scope notice: **“Some corridor maps could not be generated. Mileage results are
  available. See Diagnostics.”** Use only when numeric results actually succeeded.

Distinguish `corridor_buffer_limit`, `corridor_projection_unavailable`,
`corridor_geometry_invalid`, `corridor_coverage_failed`,
`corridor_serialization_failed` and `state_corridor_omitted`. Include scope,
pair/section identity, stage, limits and counts in diagnostic context. Deduplicate
on repeated preparation/export. Do not inherit Combined map warnings into an
unrelated state run. Single-state reuse must copy decisions safely and still clip
the state's buffer; it must not mark the reused uncut shape as state-certified.

## 7. UI and export behavior

### UI

Keep existing summary cards, tables, scope selector and corridor launch actions.
Use the common prepared decision in both GUI entrypoints. Ready multipart/holed
sections remain one overlap row with one View Corridor action. Omitted maps keep
their numeric row and disable only the map action. Put padding/approximation
details in the existing disclosure; do not add another full-width status banner
for every successful map or turn this into an input-page redesign.

Verify narrow windows, high DPI, keyboard navigation, long pipeline names, disabled
actions and Combined/state switching. Do not change the pipeline table's sizing
or reintroduce the alignment regressions addressed earlier.

### KML/KMZ

Route ordinary per-section KML previews and geography package maps through one
canonical polygon serializer. Serialize every Polygon outer ring and every
`innerBoundaryIs`, using MultiGeometry when required. Use lossless float roundtrip
formatting for new shapes. Validate the actual serialized/reparsed geometry for
topology, holes, coverage/tightness and state containment.

For the new schema, omit the old center Point marker and keep previews
polygon-only. Do not add corridor centerlines or original-line overlays to user
exports: ordinary LineStrings are counted on reimport. Preserve existing package
layout, collision handling and staged publication. Ordinary analysis still exports
XLSX/JSON through Save As; View Corridor creates a temporary KML. Do not invent an
ordinary whole-analysis KMZ workflow as part of this change.

A map already marked omitted is omitted consistently from all views/packages and
documented in Diagnostics. An unexpected serialization failure while exporting a
ready map aborts the staged package; do not publish a silently incomplete package
or mutate the completed result. Improve preparation/reparse checks to catch the
cause before publication on the next run.

### JSON and workbook compatibility

JSON retains complete canonical geometry, schema/policy, padding, status and
source-run provenance. Update validation so an invalid polygon cannot be hidden by
`default=str`. Results and exports must agree on what is ready/omitted.

Preserve existing workbook sheet names, numerical column order, formulas, formats
and escaping. For a new section, leave unsupported legacy single-ring/width cells
blank rather than writing zero or a misleading partial shape. Append one compact
**Corridor Map** column to overlap sheets (Combined and state): e.g.
`Available (2 parts, 1 hole)` or `Unavailable — see Diagnostics`. A one-part,
hole-free compatibility ring may still populate the old `corridor_polygon` column.
Record padding/policy and the legacy-column explanation in Analysis Details.

Do not put complete multipart JSON into an Excel cell: its size limit can silently
truncate geometry. Maps and optional JSON carry full shapes. Existing ordinary
two-sheet exports can add Analysis Details only when new corridor metadata requires
it; legacy result dictionaries retain their existing workbook contracts. Preserve
source-text/formula escaping and reuse the current Analysis Details sheet when
repair or state metadata already created it.

Apply a preflight character limit to **every** geometry cell, including a single
hole-free compatibility ring and bbox text. Excel's 32,767-character cell ceiling
is a hard constraint: leave an oversized legacy geometry cell blank, and make the
Corridor Map/detail text explicitly say geometry is available in the map/JSON.
Do not allow library truncation, split arbitrary ring fragments across cells, or
make a ready map appear unavailable merely because its coordinates exceed a cell.

## 8. Ordered implementation tasks and exit gates

Use small reviewable changes; do not combine the numeric refactor with the new
algorithm before the first gate is green. Different owners may work in parallel
only on disjoint files after agreeing the shared contract. One integrator owns
`overlap.py`, `analyzer.py`, `corridor_geometry.py` and shared test fixtures.

| Order | Work and main files | Dependencies | Exit gate |
| --- | --- | --- | --- |
| T0 | Record HEAD/dirty files, input hashes, dependency versions and baseline outputs; add a scoped numeric comparator and deterministic corridor fixture generator under `scripts/validation/`. | None | Existing tests pass; before-results include ordinary/state/shared/failed cases and stable section identities. Never regenerate mileage goldens from the candidate code. |
| T1 | Separate completed qualification/accounting from fallible shape construction in `core/overlap.py`; adjust orchestration in `core/analyzer.py` and stage order in `core/progress.py`. Keep the old geometry method temporarily. | T0 | Forced geometry exceptions preserve every numeric result and section; forced cancellation aborts. Existing successful outputs remain numerically identical and progress follows actual work. |
| T2 | Introduce authoritative canonical/versioned decisions in `core/corridor_geometry.py`; migrate `export/corridor_kml.py` to common multipart/holes serialization and update legacy adapters. | T1 | Canonical-only preview works; empty/omitted/invalid canonical geometry never falls back to bbox; legacy-only fixtures still work. |
| T3 | Add explicit qualified-run extraction to `core/corridor_coverage.py` and test provenance, chainage, gaps and tails. | T1 | Independent run coverage equals full qualified sample coverage, with no added connector or excluded partner endpoint. |
| T4 | Implement isolated `core/corridor_buffer.py`, immutable options/budgets and projection/densification/union/validation. Keep production selection on the old builder until its gates pass. | T2, T3 | Analytic, geographic, tightness, holes, multipart, seam, determinism and failure-budget tests pass. Prototype screenshot alone is insufficient. |
| T5 | Attach the buffered builder to completed numerical sections; finalize ordinary/Combined decisions and integrate `core/geography/corridors.py`, `core/state_analysis.py`, scoped `core/progress.py` allocations. | T4 | Complete statistical parity; shared allocation unaffected; single-state reuse safe; all serialized state shapes contained; map work advances progress without premature completion. Expected map failures remain warnings only. |
| T6 | Migrate workbook/JSON metadata, both GUI entrypoints, launch/omission messages, geography package export and source text handling. | T2, T5 | UI/export agree on every section; multipart and holes survive KML/KMZ reparse; no extra countable LineStrings; package transaction failure tests pass. |
| T7 | Upgrade gallery/audit tools and independent fixture validators; run matrix, stress/resource checks and before/after review packet. Tool upgrades may begin after T2 in disjoint files. | T5, T6 | Every polygon/hole checked; no weakened numeric oracle; measured resource/cancellation gates; small-window and high-DPI review accepted. |
| T8 | Extend frozen smoke, build isolated Windows output, run both entrypoints offline, publish verified local dist; execute macOS platform validation on a Mac. | T7 | Artifact/version/hash recorded; Windows local executable verified; macOS distribution withheld until its own gate passes. |

At each task record changed files, tests, evidence location, unresolved failures and
the exact next task in `docs/validation/corridor-buffer-implementation.md` (create
during implementation). Mark a task complete only when its exit gate is satisfied.
No automatic commits, merge, release or remote distribution are required by this
runbook. Record unrelated concurrent edits and coordinate ownership before touching
the same files; never reset another task's work.

## 9. Acceptance matrix

Add meaningful tests across new `tests/test_corridor_buffer.py`,
`tests/test_corridor_buffer_integration.py` and the existing corridor, state,
workbook, GUI and packaged-smoke suites. Build independent expected results from
known geometry, frozen statistics and separate measurement code.

| Area | Required cases and assertions |
| --- | --- |
| Extraction | Bend within one analysis sample, duplicate vertices/zero edges, endpoint exactly on native vertex, sparse 100 km edge, shifted partner endpoint, unequal lengths, nonconsecutive sample IDs, invalid ID/path, duplicate names/IDs, two disconnected paths, state reentry, short unqualified tail. Assert all and only qualified run intervals. |
| Analytic shapes | Isolated straight run: capsule extent and area `2*r*length + pi*r*r` within derived arc error. Coincident/reversed duplicates do not double map area. L/U/S curves, hairpin, repeated/backtracking vertices, tight bend and full loop preserve intended extent/voids. |
| Networks | T/X crossing, branch-then-diverge/rejoin, long parallel pairs, three/four overlapping pipelines, disconnected groups, loop and figure-eight. Numeric grouping remains unchanged; every part has a generating run; no filled hull across empty space. |
| Width boundary | Separations just below/equal/above 10 m with the 5 m radius; distinguish tangent-touch numerical cases from a meaningful gap. Disconnected areas remain valid; no automatic widening. Vary detection range with qualification held fixed and assert unchanged padding. |
| Geographic accuracy | Equatorial/mid/high latitudes, Alaska/Hawaii, UTM-zone crossing, antimeridian in both directions, exact +/-180 endpoint, long sparse edges, chart seams and alternate chunk boundaries, geographic holes. Independent dense reference meets the 0.05 m combined target. Unsupported polar/global winding fails visibly, without math loss. |
| State clipping | Buffer extending over a boundary, exact endpoint touch, islands/holes, reentry, three-state junction, shared-border-only allocation, near-border parallel paths on opposite sides. Full serialized geometry difference from state boundary is empty; no shared allocation becomes an interior corridor. |
| Invariance | For geometrically equivalent extracted runs: reverse run direction, reorder operands, vary chunk boundaries, add redundant vertices, and rigidly rotate/translate metric cases; compare shapes within tolerance with provenance remapped. Full-analysis variants each retain their own frozen numerical expectations. Use full-step, phase-safe fixtures for whole-path reversal; a remainder fixture must honor the different qualified extent when reversal moves the unqualified tail. Do not assume arbitrary reversal, changed segmentation or source-index renumbering is numerically identical. |
| Failure isolation | Inject projection, buffer, union, normalization, state clip, serialization and per-section/job limit failures, including after some maps succeed. All source/adjusted/savings/section values remain available; only affected maps omitted. Cancellation is separately tested to publish no result. |
| Export truth | Reparse every outer/inner ring and component. Empty/invalid/unknown canonical schema cannot revive a rectangle. Preview and package geometries agree. Combined KMZ reimport retains original mileage; state KMZ retains interior mileage; polygons add no line mileage. Source names requiring XML/Excel escaping remain literal. |
| Workbook | Numeric cells/formulas unchanged, optional details sheet reused, legacy-only results unchanged, no false zero width, readable multipart/omission status, no truncated coordinate blob. |
| GUI | Both entrypoints; Combined/state switch; row/action state; no click-time geometry recalculation; long names; 900x650 and supported smaller layouts; 125/150/200% DPI; keyboard; retry, window close and stale-worker protection. Use isolated desktop tests. |
| Resources | Many short sections, maximum complex ring/holes, overlapping high-vertex buffers, a long sparse route, deeply densified curve, dense loop network and huge state boundary piece. Prove cumulative limits and native-call caps are enforced before large allocations; no unbounded all-pairs work. |

For realistic data include the four main independent state KMZ fixtures, their
order/direction/redundant-vertex variants, the branching stress fixture and the
repository's Adamas centerline fixture. Use the repaired August 2023 source only
as an optional local check when available; do not make CI depend on a Downloads
path or its stored `MILES` fields. Those attributes are not geometry ground truth.

Update existing assertions that intentionally describe the old display contract:
`test_parallel_overlap.py`'s detection-range width clamp,
`test_corridor_coverage.py`'s `sampled_curve` text,
`test_calculation_regressions.py`'s single-ring result, ordinary seven-decimal/center
Point expectations, and golden *visual diagnostic* counts. Preserve their numeric
and containment assertions. Never loosen comparisons to “some polygon exists.”

The new before/after comparator must explicitly enumerate visual fields and
known visual diagnostic codes it ignores. Preserve all unrelated diagnostics,
failures, state statuses and numeric fields; do not strip every diagnostic or use
only grand totals. Compare qualified membership/section identity as well as values.
The independent suite's polygon-per-section matching and anti-false-pass checks
must remain active; add radius/tightness and full-run coverage checks instead of
relying solely on its previous broad bbox bounds.

Performance acceptance uses the same pinned fixtures/environment before/after,
three measured runs and recorded peak memory. Target end-to-end median time no
worse than baseline + `max(1 second, 25% of baseline)` on representative normal
inputs, with <=25% peak-memory growth unless a documented reviewed tradeoff is
necessary. Record visual-stage time separately. Stress inputs may omit maps at
the defined limits but must preserve numeric results, remain within the recorded
memory ceiling and meet cancellation checks. No performance claim is made by
this plan itself.

## 10. Execution commands and proof artifacts

Commands are run from the repository root. The implementation now provides the
test/script filenames below; actual run receipts are in the implementation report.

Baseline command already executed:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_corridor_geometry.py tests/test_corridor_coverage.py tests/test_corridor_visibility.py tests/test_corridor_launch.py tests/test_state_geometry.py tests/test_geography_export.py tests/test_export_workbook.py -q -m 'not native_gui'
```

Implementation gates:

```powershell
# New construction/extraction/integration checks plus existing math contracts.
.venv/Scripts/python.exe -m pytest tests/test_corridor_buffer.py tests/test_corridor_buffer_integration.py tests/test_corridor_geometry.py tests/test_corridor_coverage.py tests/test_parallel_overlap.py tests/test_calculation_regressions.py tests/test_grouping_hardening.py tests/test_state_geometry.py tests/test_geography_export.py tests/test_export_workbook.py -q -m 'not native_gui'

# Existing independent state dataset goldens, then broad production regression.
.venv/Scripts/python.exe -m pytest tests/test_state_kmz_regression_suite.py -q
.venv/Scripts/python.exe -m pytest -q -m 'not native_gui' --ignore=tests/fixtures

# GUI hook runs each native case in an isolated desktop/process.
.venv/Scripts/python.exe -m pytest tests/test_corridor_visibility.py tests/test_corridor_launch.py tests/test_state_breakdown_ui.py tests/test_ui_shared_surfaces.py tests/test_measured_progress.py tests/test_execution.py -q

# Upgraded independent geometry reference/contract suite, never regenerate goldens to pass.
.venv/Scripts/python.exe tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py validate
.venv/Scripts/python.exe tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py compare
.venv/Scripts/python.exe tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py audit
.venv/Scripts/python.exe scripts/validation/build_gallery.py --output .validation-output/corridor-buffer/gallery
git diff --check
```

T0 must add a dedicated comparison command that captures before-results and
compares candidate numeric results with the explicit exclusions above; record its
final CLI in the implementation report. `check_hardening.py`'s old exclusions and
historical baseline alone are insufficient for this new schema. Collect missing
independent-oracle dependencies through the fixture suite's documented setup;
do not silently skip these release gates.

Inspect the audit's structured report, not only its exit code: its known-defect
allowances are useful for historical auditing but insufficient for release. Require
`application.fully_passing` to be true, no unexplained known failures, and the
adversarial plus frozen-stress checks to execute without skips. The broad pytest
command intentionally excludes fixture-authoring tools; `audit` supplies those
additional contracts. Update visual expectations narrowly; never regenerate
independent mileage oracles to accommodate this implementation.

Windows build only after source/export/UI gates:

```powershell
$corridorBuildRoot = Join-Path (Get-Location) '.validation-output\corridor-buffer\package'
& '.\scripts\windows\build_exe.ps1' -OutputRoot $corridorBuildRoot
if ($LASTEXITCODE -ne 0) { throw 'Corridor build failed' }
$corridorVersion = (Get-Content -LiteralPath (Join-Path $corridorBuildRoot 'build\version.json') -Raw | ConvertFrom-Json).version
$corridorExe = Join-Path $corridorBuildRoot ('dist\Pipeline_Calculator_v' + $corridorVersion + '.exe')
.venv/Scripts/python.exe scripts/validation/check_packaged_smoke.py $corridorExe --output-directory (Join-Path $corridorBuildRoot 'smoke') --expected-version $corridorVersion
if ($LASTEXITCODE -ne 0) { throw 'Corridor packaged smoke failed' }
Get-FileHash -LiteralPath $corridorExe -Algorithm SHA256
```

Extend `smoke.py`, `check_packaged_smoke.py` and their tests first so “passed”
requires the new policy, curved geometry, multipart/hole serialization, state
containment and numeric parity, in addition to existing repair/resource checks.
Run offline for both application entrypoints. Do not launch test windows on the
user's desktop. Publish only the tested executable to its versioned local `dist`
name and `dist/Pipeline_Calculator_v4.exe`, verify matching hashes and preserve
existing fixture folders. The isolated build path avoids the build script's
cleanup of the user's current distribution folder.

On macOS use the existing `scripts/macos/build_app.sh` and packaged checker in an
isolated checkout/output workspace, preserving local artifacts. Verify Shapely's
native library, PROJ resources, state dataset, both GUI implementations and native
map opening. A Windows success does not clear macOS release gates. Do not initiate
remote release/signing/notarization tasks solely for this local feature.

Retain: source/dependency versions; fixture and boundary hashes; before/after
numeric comparison; analytic/dense-reference errors; full polygon/hole counts;
state containment and reimport results; failure/cancellation/resource timings;
screenshots and L/U/loop/branch/state sample KML/KMZs; workbook/JSON sample; test
logs; package version/hash/smoke. Private customer geometries remain outside any
new public fixture set.

## 11. Readiness and completion checklist

Planning review found no unresolved product choice preventing implementation.
Windows exit gates are now verified by the linked implementation evidence;
macOS distribution still requires its own platform verification:

- [x] T0 numerical and environment baseline frozen.
- [x] T1 geometry failures cannot alter accounting; cancellation still aborts.
- [x] T2 every consumer respects canonical holes/multipart and omission.
- [x] T3 all qualified runs extracted without bridging paths/gaps or adding tails.
- [x] T4 independent geometric accuracy, tightness, seams and resource limits pass.
- [x] T5 state clipping and numerical parity pass on realistic datasets.
- [x] T6 polished UI, workbook/JSON and preview/package contracts pass.
- [x] T7 gallery and sample export reviewed, including holes, disjoint pieces and
      an unavailable-map case; native small-window/high-DPI checks pass.
- [x] T8 tested Windows executable published locally with matching hashes.
- [ ] macOS platform build/test gate completed before macOS distribution.

Stop a task at a failed correctness gate and repair the cause. Do not enlarge
tolerances, discard components, change mileage goldens, fill holes or substitute
rectangles merely to complete the rollout. If the new builder cannot meet its
accuracy/resource contract on supported fixtures, keep the prior released artifact
available while correcting the new code. Unexpected unsupported cases in a valid
run may omit their map; they must never change measured mileage to make a map
look complete.

### Planning review disposition

Independent geometry and integration reviews were completed before finalizing
this runbook. Their required corrections are incorporated: isolate display
failures from accounting; make canonical polygons/omission authoritative; retain
sampling-phase semantics; require both minimum and maximum buffer extent; preflight
native output expansion; bound every Excel geometry cell; migrate progress stages;
and run adversarial/stress audits with no known-failure allowance for release.

The planning baseline and source fingerprints are retained with the feasibility
evidence. The planning task itself did not change production code, tests or the
local distribution. The subsequently authorized implementation updates the
checkboxes above as its gates pass; current receipts live in the implementation
report.
