# External-agent prompt: independently verified pipeline KMZ fixtures

Create a reproducible synthetic KMZ regression suite for our pipeline mileage and
state-boundary analysis application. Deliver actual KMZ files, their generator,
an independent validator, and independently established expected results. Do not
modify application code or replace existing fixtures. Do not merely propose layouts.

We want realistic branching pipeline networks with known answers, including
positive overlap savings and difficult negative controls. A pleasing map alone
is insufficient. No finite fixture suite proves universal correctness; identify
which behaviors each case can and cannot establish.

## 1. Inputs and calculation contract

Use repository baseline `71da499d5756648ae395660f0a241ea00edbea4f`, or record and
explain any supplied baseline change before generating expectations. Read these
files to understand the public behavior; do not use their calculation functions
to generate expected answers:

- `src/pipeline_calculator/core/constants.py`
- `src/pipeline_calculator/core/segmentation.py`
- `src/pipeline_calculator/core/coordinates.py`
- `src/pipeline_calculator/core/angles.py`
- `src/pipeline_calculator/core/overlap.py`
- `src/pipeline_calculator/core/bundling.py`
- `src/pipeline_calculator/core/analyzer.py`
- `src/pipeline_calculator/core/state_analysis.py`
- `src/pipeline_calculator/core/geography/boundaries.py`
- `src/pipeline_calculator/core/geography/partition.py`
- `src/pipeline_calculator/parsers/kml_kmz.py`
- `tests/fixtures/geography/README.md`

Use the bundled full-detail 2025 Census TIGER/Line boundary resource:
`src/pipeline_calculator/data/states_2025.zip`.
Its SHA-256 is
`55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539`.
Read its manifest, retain provenance, and verify the hash. It covers the 50 states
and Washington, DC. Use its existing transformed coordinates and boundary model;
do not substitute simplified boundaries, another vintage, a web map, or a fresh
CRS transformation. Boundary edges are coordinate-linear in longitude/latitude;
pipeline edges are ellipsoidal geodesics. Handle longitude wrapping explicitly.
Numerical agreement with this dataset does not establish surveyed ownership.

If repository access or the resource is missing, request those inputs. You may
prepare the generator structure, but do not invent certified state expectations.

Freeze the default analysis profile:

| Setting | Value |
| --- | --- |
| Ellipsoid | GRS80 |
| Original distance | Sum of geodesic lengths of consecutive vertices, separately for each path |
| Mile conversion | 1 US survey mile = 1609.347218694 meters |
| Analysis segment length | 5 meters |
| Detection range | 15 meters |
| Minimum qualifying parallel length | 200 meters |
| Angular tolerance | 15 degrees, treating opposite digitization directions as parallel |

Original mileage includes short tails. Overlap uses full 5-meter samples on each
independent path; the trailing remainder does not contribute sampled savings.
Sampling continues across vertices within a path and restarts at disconnected
paths and clipped state fragments. For a sample spanning a bend, the application
uses the midpoint and bearing of the geodesic chord between its sample endpoints.

Overlap requires compatible orientation, transverse separation within range,
and positive longitudinal overlap of the finite sampled tangents. It is not
simply midpoint proximity, a buffer intersection, or line intersection. Qualifying
sections are connected matches in the two paths' sample-index grid; missing rows
or columns and separate paths split sections. Section length is the smaller of
the two uniquely covered sample-length totals and must reach the minimum.
Do not infer a geographic gap-bridging rule from the unused `GAP_TOLERANCE` constant.

Savings groups must be disjoint, contain at most one sample per source pipeline,
and be mutually compatible across every pair. The current contract merges nearest
eligible candidates first, with the documented geometry-based tie breaking. It
is a deterministic grouping heuristic, not an optimal route calculation. Each
group saves `sum(member lengths) - max(member lengths)`. Summing pairwise overlap
lengths overcounts three-pipeline bundles. Self-intersection or retracing within
one source pipeline does not create cross-pipeline savings.

State analysis receives exclusive state interiors only. Shared-border geometry
is stored once and its unrounded mileage is allocated equally among its verified
adjoining states. Shared allocation receives no state overlap discount and must
not help an interior run qualify. Combined analysis still analyzes the original
complete paths normally. State adjusted mileage and savings need not add up to
the Combined values.

## 2. Four required KMZs

Choose and verify actual locations against the supplied boundaries. For the main
networks, aim for roughly 20–60 source pipelines and 20–150 km of total linework
per file, adjusting where necessary to keep the cases understandable and practical
for regression runs. These are design targets, not fabricated expected totals.
Use branches, bends, loops, crossings, and runs that join, travel in parallel, then
diverge and rejoin. Mix dense and sparse original vertices. Realistic connecting
lines must be included in the independent accounting and overlap checks.

### 01_three_state_disconnected_networks.kmz

Build separate networks wholly inside Texas, Louisiana, and Wyoming, complementing
our existing real-world centerline fixture. Keep all geometry comfortably inside
its state and verify whole paths, not just endpoints. No path crosses a state
border; expected crossings, shared, outside, and unresolved mileage are all zero.

Each state must have independently verified positive savings and negative controls:

- Long parallel pairs, approximately 6–10 m apart, that qualify comfortably.
- A three-pipeline bundle in which all three pairs qualify. Check group savings
  without triple-counting pairwise corridor lengths.
- A separate A–B–C arrangement where A–B and B–C are within range but A–C is
  outside range. Use unequal adjacent separations to avoid a needless tie. It
  must not be treated as an all-compatible three-line bundle.
- Perpendicular crossings, diverging branches, loops, and isolated lines that
  contribute original mileage without inappropriate savings.
- A multipart source with disconnected sub-threshold parallel runs whose combined
  length exceeds 200 m, proving that qualification does not bridge their gap.

The state selector should list three states despite zero border-crossing events.
For this particular case, Combined savings should equal the sum of state savings
because independent path inputs are unchanged and no cross-state matches exist;
verify that premise and equality instead of assuming it for other files.

### 02_three_state_transverse_crossings.kmz

Choose a real three-state neighborhood with practical, unambiguous boundary
crossings. Use continuous routes visiting A, B, and C through separate crossings,
plus branches that stay in one state and a route that leaves and reenters a state.
Do not rely on hitting the exact three-state junction point.

- Make the intended border intersections approximately perpendicular to the local
  boundary tangent; report measured crossing angles. Exact 90 degrees is unnecessary.
- Include a sparse geodesic edge with a crossing strictly between its endpoints,
  plus a path with several crossings, so endpoint-only assignment cannot pass.
- Include endpoint touches that acquire no positive mileage in the touched state.
- Allow positive overlap controls wholly inside states, but keep crossing routes
  apart from each other near borders to isolate clipping from cross-border overlap.
- Require zero positive shared-border mileage. Produce a source/path crossing
  ledger with crossing chainages and state transitions.

Every genuine short state visit must be preserved; reentry must not introduce a
straight connector between its exit and reentry points.

### 03_parallel_corridors_crossing_borders.kmz

Build another three-state network with distinct pipelines traveling together
across boundaries, then branching and rejoining. These pipelines cross the border;
they do not follow it. Require zero shared-border mileage in this file.

Include separately labeled, independently checked motifs:

1. A long parallel corridor with enough exclusive mileage on each side that
   overlap qualifies independently in both states.
2. A short cross-border corridor with roughly 300 m of parallel travel split into
   roughly 150 m per state. It qualifies Combined but fails the 200 m state minimum
   on both sides. Compute actual sampled results after serialization; these nominal
   construction lengths are not the expected answers.
3. An asymmetric split where one state's run qualifies and the other does not.
4. A three-pipeline bundle crossing a border, with a member joining or leaving
   near the crossing, exposing mistakes in qualification and group savings.
5. Parallel runs that diverge beyond detection range and later rejoin. Keep each
   separated qualifying section distinct; do not bridge an ineligible interval.
6. Two long, close parallel lines lying on opposite sides of a border, each
   exclusively in its own state. Their Combined match may qualify, but neither
   state may import the other state's line or receive savings from that pair.
7. Nonaligned sample starts, non-multiple-of-5-meter tails, opposite digitization
   directions, and sparse-versus-dense vertex patterns away from threshold ambiguity.

Verify all incidental interactions with connectors and nearby motifs. Keep control
areas far enough apart to prevent unintended matches, without omitting the mileage
of lines used to make the overall layout realistic.

### 04_shared_border_and_near_border.kmz

Exercise the different case of a pipeline lying along a verified common border.
Use a real shared meridian segment suitable for exact coincidence under the
supplied boundary model. A known candidate to independently verify is the Texas /
New Mexico segment from `(-103.064732, 32.744215)` to
`(-103.064732, 32.75427)`. Re-read the canonical boundary coordinates and retain
their precision; do not treat these suggested coordinates as proof of ownership.

- Include an interior → shared border → interior route, and a shared-only source.
- Include shared runs both longer and shorter than the overlap minimum; allocation
  applies at every positive length and is independent of the overlap threshold.
- Add a second source close enough to a shared run to create a Combined overlap
  control. Its interior portion must not use shared geometry to qualify state savings.
- Add exclusive parallel lines just inside either state, including a clearly
  resolved centimeter-scale offset. Near a border is not shared ownership.
- Include a tiny but real exclusive crossing and a zero-length endpoint touch.
- Generate a reversed-path variant and an extra-collinear-vertex variant. Shared
  ownership and allocated original lengths must remain unchanged.

Do not represent a winding river boundary or a latitude parallel as an exactly
shared geodesic just because its endpoints lie on the boundary. The whole
positive-length interval must coincide with the chosen model. If a proposed
location cannot meet that condition, select a valid one and document the change.

## 3. Independent expected results

Establish two clearly named forms of evidence:

**Geometric reference:** Parse the final saved KMZ directly. Independently sum
GRS80 edge lengths and solve path/boundary intersections into disjoint intervals
along each source path. Do not call the application's parser, partitioner, or
analyzer for this reference. Shapely and geodesic libraries are permitted, but
plain intersection of sparse longitude/latitude endpoint chords is insufficient.
Use an independently implemented, convergence-checked approach with bounded
crossing error. Cross-check selected distances using a second implementation
configured for GRS80. Measure the serialized coordinates, not pre-export designs.

**Sampled-contract reference:** Independently calculate the frozen 5 m matching,
qualification, and savings rules. Favor a clear, slower brute-force implementation
on isolated motifs over reproducing the production spatial-index algorithm. Use
explicit sample-coverage sets and independently checked grouping. Validate it
against hand-worked two-line, three-line, and non-clique controls. Do not copy
production functions or bless application outputs as golden values. The existing
`tests/reference/intervals.py` models continuous planar intervals and is not an
exact substitute for this sampled contract on arbitrary networks.

For aligned controls with `Q` meters of fully qualifying sampled coverage per
source, a pair saves `Q`, and an all-compatible trio saves `2Q`, not `3Q`. A chain
where the outside pair is incompatible cannot collapse all three lines into one
pass. Prove the alignment and eligible samples for these controls before using
these formulas; nominal geometric run length is not necessarily `Q`.

For each fixture and analysis profile, provide:

- Exact archive checksum, XML source order, feature/path/vertex counts, stable
  fixture source keys, expected source-pipeline counts, and represented states.
- Per-source and per-path original meters and US survey miles, with per-state
  interior, shared allocation, and attributed original lengths.
- A canonical interval ledger: source key, path index, chainage start/end, kind
  (`interior`, `shared`, `outside`, or `unresolved`), state codes, and length.
- Each crossing's source/path, coordinate, chainage, and state transition. Distinguish
  crossing events from the number of represented states and from endpoint touches.
- Combined and per-state original length, qualifying overlap sections, savings,
  and adjusted length. Keep pairwise qualifying coverage distinct from total savings.
- Qualifying section source/path pairs and covered sample ranges; for each source,
  uniquely covered sample totals. Do not invent a per-pipeline allocation of total
  savings where the application has no such allocation contract.
- Shared intervals stored once, with explicit allocation references for each state.
- Expected statuses and diagnostics, plus a coverage matrix linking each intended
  failure mode to a named source or motif and a numerical assertion.

Keep machine-readable values in meters without presentation rounding; include
survey-mile conversions for human review. Full-precision numbers still have a
numerical error bound: report the method and achieved uncertainty.

Required accounting:

```text
state attributed original = state interior + shared allocation
state adjusted = state attributed original - qualified interior savings
sum(state attributed original) + outside + unresolved = Combined original
```

Validate conservation both per source and for the entire fixture within
`max(0.001 m, original_meters * 1e-10)`. Separately certify each cut location to
within 0.01 m along the source geodesic relative to the chosen boundary model.
Derive state-total comparison bounds from the certified endpoint uncertainties;
do not confuse the clipping-location target with the stricter conservation check.
Never rebalance state lengths to hide discrepancies or discard tiny intervals.

For well-separated controls, require exact qualifying sample counts and savings
in sample-length units, allowing only floating-point arithmetic tolerance.
Do not hide a one-sample mistake behind a general ±5 m allowance. If a case is
threshold-sensitive, document it, determine a defensible expected decision from
the serialized input, or redesign the control with a margin. Never fabricate an
exact overlap expectation that the independent reference has not established.

The four valid main fixtures should have no unresolved mileage. Outside coverage
must be zero unless explicitly included and independently measured in an edge case.
Unexpected incomplete analysis is a failed check, not a zero-savings success.

## 4. Serialization, variations, and export checks

Use one `doc.kml` per KMZ. Only actual source pipelines may appear as LineStrings;
use MultiGeometry for disconnected paths belonging to one source. Preserve those
gaps. Separate pipelines intended to overlap must be separate Placemarks: several
parallel LineStrings inside one source do not test cross-pipeline savings. Give
each source a stable fixture key and record its XML order. Include a
few deliberately repeated display names and OBJECTID values without merging source
identities. Keep XML IDs valid and unique in the main files; the existing real
fixture already exercises repeated XML IDs.

Use sufficient coordinate precision for the stated error bounds. Include no
reference centerlines, state-outline LineStrings, corridor centerlines, hidden
duplicate overlays, or NetworkLinks in the input. Hidden lines still count as
input geometry. Separate preview maps and annotations from the actual test KMZs.
Altitude must not contribute to the 2D geodesic mileage reference.

The generator should support deterministic source-order permutations, path
reversal, and redundant vertices on the same original geodesic. Compare by stable
fixture identity, not parser-assigned integer IDs. Original mileage and geographic
allocation should remain invariant. Reversal can move the sampled tail and change
sampling phase; do not blindly require identical savings for every variation.
Calculate variant expectations independently and identify phase-safe cases where
unchanged savings are a valid assertion. Reordering source records should preserve
savings away from deliberately ambiguous ties. Renaming is not a universal overlap
invariant because the current grouping tie breaker includes source names.

Provide assertions usable for later application export tests:

- Combined KMZ reimport recovers original source mileage without added overlays.
- State KMZ reimport recovers exclusive interior mileage, excluding shared allocation.
- Every state line and corridor polygon remains within its state, including multipart
  shapes and holes. Polygons must not become extra pipeline mileage.
- Shared source intervals occur once in the Combined map, not duplicated into state
  maps. A state represented only by shared allocation needs no empty state map.

A separate optional small edge-case file may cover an Alaska dateline crossing,
Hawaii/island coverage, outside-only geometry, and boundary holes selected from
the real resource. Verify actual ownership rather than assuming every island or
dateline path belongs to a state. Threshold micro-cases may exercise detection
distance, angle, and qualifying sample counts around their cutoffs; use explicitly
separate parameter profiles when necessary to make those cases geometrically possible.

After the four core fixtures are verified, offer a deterministic larger generator
profile with more branching and overlapping groups. Aim below the application's
documented segment and neighbor-work limits for a successful stress run; record
actual counts, runtime, hardware, and memory. More bytes or more isolated lines
alone do not establish meaningful overlap stress. A resource-limit rejection is
a separately labeled test, not evidence of successful full analysis.

## 5. Delivery and readiness

Deliver a single organized folder or ZIP:

```text
pipeline_kmz_regression_suite/
  README.md
  requirements.txt
  generator/
  reference/
  fixtures/
    01_three_state_disconnected_networks.kmz
    02_three_state_transverse_crossings.kmz
    03_parallel_corridors_crossing_borders.kmz
    04_shared_border_and_near_border.kmz
  expected/
    <fixture>.expected.json
    <fixture>.intervals.csv
    <fixture>.overlaps.csv
  previews/
    <fixture>.png
  validation/
    reference_report.json
    coverage_matrix.md
```

Document one command to regenerate and one to independently validate all files.
Pin dependency versions, random seeds, generator version, baseline commit, and
boundary checksum. Make KMZ creation deterministic, including ZIP entry metadata.
Once dependencies and the boundary resource are supplied, generation and validation
must operate offline. Define a versioned expectation schema and explain each field.

README must contain a compact Combined/per-state results table for every fixture,
the expected positive and negative controls, numerical tolerances, and known limits.
Preview maps must make branches, borders, and named control areas easy to inspect.

If you can run the application, compare it with the independent references only
after fixing those references. Record mismatches separately with a minimized
reproduction. Do not adjust expected answers until they agree with the application.
If the application is unavailable, explicitly mark application comparison pending.
Mark each fixture reference-verified or incomplete; incomplete or unexplained
expectations are not ready to adopt as regression goldens.

## Repository references for this prompt

- [Defaults and distance units](../../src/pipeline_calculator/core/constants.py)
- [Sample construction](../../src/pipeline_calculator/core/segmentation.py) and
  [disconnected-path handling](../../src/pipeline_calculator/core/coordinates.py)
- [Overlap eligibility](../../src/pipeline_calculator/core/overlap.py) and
  [section qualification and group savings](../../src/pipeline_calculator/core/bundling.py)
- [State accounting and independent scoped analysis](../../src/pipeline_calculator/core/state_analysis.py)
- [Boundary interpretation](../../src/pipeline_calculator/core/geography/boundaries.py)
  and [partitioning and conservation](../../src/pipeline_calculator/core/geography/partition.py)
- [Existing real-world fixture evidence](../../tests/fixtures/geography/README.md)
- [Existing continuous planar reference and its distinct model](../../tests/reference/intervals.py)
