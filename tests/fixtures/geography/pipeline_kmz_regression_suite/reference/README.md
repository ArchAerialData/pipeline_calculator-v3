# Independent references

The references measure **saved KMZ coordinates**, importing no application
calculation functions. Expectations are fixed before the separate
[application comparison](../validation/compare_application.py).
The governing specification is the [fixture prompt](../../../../../docs/validation/kmz-fixture-agent-prompt.md)
at repository baseline `71da499d5756648ae395660f0a241ea00edbea4f`.

## Frozen calculation contract

| Quantity | Contract |
| --- | --- |
| Ellipsoid | GRS80: equatorial radius 6,378,137 m; inverse flattening 298.257222101 |
| Original length | Sum consecutive geodesic edges separately for every path; altitude ignored |
| Human distance unit | 1 US survey mile = 1609.347218694 m |
| Samples | Full 5 m path intervals; retain original mileage of unsampled tails |
| Detection / orientation | 15 m transverse distance; 15° orientation difference modulo 180° |
| Qualification | At least 200 m unique sample coverage on **each** side of a connected match section |
| Savings | Disjoint mutually compatible groups, at most one sample per source; each group saves `(member_count − 1) × 5 m` |

Sampling continues through vertices but restarts at disconnected paths and state
fragments. A bent sample uses its endpoint **geodesic chord** midpoint and bearing,
while its counted coverage remains 5 m. Shared geometry contributes equally
allocated original length to its adjoining states and contributes no state
overlap samples. Combined analysis uses complete original paths.
[Contract implementation references](../../../../../src/pipeline_calculator/core/constants.py):
[sampling](../../../../../src/pipeline_calculator/core/segmentation.py),
[eligibility](../../../../../src/pipeline_calculator/core/overlap.py),
[grouping](../../../../../src/pipeline_calculator/core/bundling.py).

## Geometric reference: `geometry.py`

[The XML reader](geometry.py) requires one `doc.kml`, unique fixture keys/XML IDs,
and pipeline LineStrings. It preserves multipart gaps and rejects overlays.
The boundary archive and every state WKB are
SHA-256 checked against their manifest; all 50 states and DC must be present.
Existing transformed coordinates, polygon components and holes remain intact.

GeographicLib traces each source edge on GRS80. For a native boundary segment,
the reference solves its coordinate-linear line equation along that geodesic.
Longitude/latitude second derivative bounds limit interpolation
error by `C × station_span² / 8`. These envelopes select candidate boundary edges
and exclude impossible roots. Monotonicity bounds permit sign-bracketed
bisection; inconclusive near tangencies fail rather than becoming expectations.
This differs from the application's local projection approach and avoids
intersecting sparse source endpoint chords with a planar state polygon.

Only exact meridian/equator coincidence is accepted as a positive shared
interval. Exact endpoint determinants use 80-digit Decimal arithmetic over the
serialized binary coordinates. A rounded near-boundary endpoint is not promoted
to an exact touch. Roots are sorted by geodesic chainage, intervals classified,
and adjacent intervals of identical ownership combined while retaining original
vertices. Each shared interval is stored once with explicit allocation references.
Ring unwrapping and queries at periodic longitude copies handle wrapping without
another CRS transformation. Already canonical native longitudes remain bit-for-bit
unchanged. Overlapping numerical brackets alone cannot merge distinct boundary
events: cuts whose order cannot be resolved fail certification instead of erasing
the intervening interval. Exact collinear endpoint touches contribute no positive
shared length. The core fixtures do not test Alaska/dateline cases.
[Implementation and limits](geometry.py).

## Sampled-contract reference: `sampled.py`

[Sample locations](sampled.py) come from whole-path cumulative chainages and edge
lookup, independently of the application's edge-carry loop. Every cross-source
sample pair is visited in bounded NumPy arrays. A unit-sphere chord multiplied
by 6,330,000 m provides a conservative lower distance bound: this radius is below
GRS80's minimum meridional curvature radius. The screen can retain extra pairs;
it cannot reject a pair satisfying the finite-tangent distance bound. No spatial
neighbor tree is used.

For retained orientation-compatible pairs, let `d` be midpoint geodesic distance
and `δ₁, δ₂` the directions to the other midpoint relative to each sample's
bearing. Exact eligibility requires:

```text
max(|d sin δ₁|, |d sin δ₂|) ≤ 15 m
max(|d cos δ₁|, |d cos δ₂|) < 5 m − 1e−8 m
```

The second midpoint uses its back azimuth, accounting for meridian convergence.
Eligibility is a finite-tangent test; intersecting buffers or nearby midpoints
alone are insufficient. Matches form an explicit eight-neighbor graph in each
path pair's sample-index grid. Missing rows/columns and distinct paths split
sections. Flood filling and unique coverage sets determine qualification.

Qualified edges are processed by increasing midpoint distance. Ties use sorted
node keys `(longitude, latitude, bearing % 180, source name, path index,
source-wide sample index)`. Groups are explicit sets: a proposed union is accepted
only if sources remain distinct and every cross-pair is a qualified edge.
Membership is rewritten directly without a union/find tree. Assertions verify
disjointness and pairwise compatibility. This is the specified greedy heuristic, not an optimal routing
calculation. Renaming is not a universal invariant because names enter the tie.
[Reference implementation](sampled.py).

## Reading the sampled evidence

`sections` includes qualified and rejected components. Coverage ranges are
zero-based, path-local, **half-open** sample indices: `[0, 60]` means samples
0–59, covering 300 m. State path indices refer to newly sampled fragments; their
original interval identities are retained in `fragment_references`.

`source_coverage` is the union across qualified sections, so a trio's source
coverage is counted once. `qualifying_pairwise_meters` deliberately counts pairs
and can exceed total savings. `group_size_counts` includes unpaired qualified
nodes as size-one groups with zero savings. `savings_by_source_set` and
`motif_savings` describe groups; they do not allocate savings to individual
pipelines. `nontrivial_groups_sha256` hashes sorted groups of sorted
`(source key, path index, sample index)` members using compact JSON. It detects
membership changes; the hash alone does not prove correct grouping.
[Coverage assertions](../validation/coverage_checks.py).

## Nine hand-worked checks

Run `python reference/sampled.py` from the suite folder.

| Control | Independent assertion |
| --- | --- |
| Aligned pair | 60 diagonal eligible matches; Q = 300 m; savings 300 m |
| Compatible trio | Three diagonal pair graphs; 900 m pairwise coverage; savings 600 m |
| Unequal non-clique chain | Two diagonal pair graphs; no triple; savings 300 m |
| Disconnected short runs | Separate 120 m + 120 m components; savings zero |
| One-source retrace | Original/sample length retained; cross-source savings zero |
| Starts shifted 1.7 m | 119 eligible finite-tangent pairs; 60 disjoint groups; savings 300 m |
| Opposite direction | The tested 302.3 m pair retains 300 m savings despite changed tail phase |
| Redundant vertex | Vertex at chainage 82.1 m preserves 120 total samples and 300 m savings |
| Bend-spanning chord | A 5 m sample spanning 2 m east + 3 m north has an approximately √13 m chord; GeographicLib checks its midpoint/bearing |

All source permutations are tested for the first three controls. Reversal savings
invariance is asserted only for the specified phase-safe case. Fixture 01's
trio/chain controls have full qualifying coverage but tiny phase offsets; their
graphs also contain off-diagonal matches. [Check results](../validation/reference_report.json).

## Numerical uncertainty and limits

Roots are refined to a station bracket width of at most 0.25 µm. For refined
roots, the reported station bound is:

```text
half bracket width + max(2 µm, E / Dmin)
```

`E` is the boundary-line evaluation allowance derived from **2e−12 degrees per
coordinate**; `Dmin` bounds the absolute change in that line equation per meter
along the source. The local secant, curvature bound, and evaluation allowance
establish `Dmin`. Shallow crossings amplify coordinate uncertainty, so an
unprovable derivative or a resulting cut bound above **0.01 m** rejects the
certificate. A fixed 2 µm allowance alone is insufficient for those cases.
The coordinate allowance remains an engineering assumption, not a formal
interval-arithmetic enclosure of library rounding. See the
[implementation](geometry.py) and [reproduced audit findings](../validation/audit/geometry_audit.json).

All seven existing fixture geometries remain unchanged by this correction. Their
artifacts report maximum cut uncertainty below 2.117 µm and refinement change
below 1.403 µm. Independent pyproj edge-distance comparisons differ by less than
2.474 nm; GeographicLib/Python and PROJ/C implement related Karney algorithms,
so this is not independence of mathematical models.
[Per-fixture evidence](../expected/).

Conservation separately requires `max(0.001 m, original × 1e−10)` per source and
fixture. State comparison bounds sum relevant endpoint uncertainties, divided
among owners for shared allocations. No length is rebalanced, and positive tiny
intervals are not removed by a mileage threshold. Counts and sampled savings
are exact multiples of 5 m, without a ±5 m comparison allowance.

The geometry reference rejects nonpolar-domain violations (85° envelope), edges
over 200 km, ambiguous tangencies, and unproved shared coincidence. These fixtures
do not establish every boundary topology, threshold decision, stress workload,
or surveyed ownership. Reproduction and observed application mismatches remain
separate evidence. [Suite documentation](../README.md).

## Adversarial audit

From the repository root, run:

```powershell
python -m pytest tests/fixtures/geography/pipeline_kmz_regression_suite/validation -q
```

The checks use analytic answers and deliberately incorrect observations to test
whether the references and comparison harness can reject plausible false passes:

| Checks | Evidence |
| --- | --- |
| [Geographic oracle](../validation/test_geometry_reference.py) | Sparse geodesic visits missed by endpoint chords; exact versus centimeter-offset sharing; 10 cm crossings; literal endpoint touches; shallow or indistinguishable roots; reversal, redundant vertices, altitude and domain limits. |
| [Sampled contract](../validation/test_sampled_contract.py) | Hand-counted pairs/trios/non-cliques, 39 versus 40 samples, disconnected runs, finite tangents, angle margins, opposite directions, bends and duplicate source membership; reference and production core are checked separately against the controls. |
| [Application checker](../validation/test_application_contract.py) | Correct totals with wrong ownership, geometry, interval identity or corridor exports; holes, foreign polygons, and narrowly bounded floating-point boundary residue. |
| [Validation harness](../validation/test_suite_validation.py) | Missing or extra files, empty selections, NaN/type confusion, stale success reports, changed contract files, failed publication and disabled assertions. |
| [Stress harness](../validation/test_stress_contract.py) | Frozen-pair validation without rewriting, staged publication, contract guards, workload counters and retained failure details. |

The geographic tests include **synthetic** dateline and polygon-hole models.
They test numerical handling without certifying actual Alaska, Hawaii, or native
boundary-hole ownership. These focused tests complement full archive validation
and application/export comparison; they do not establish universal correctness.
The complete `suite.py audit` command also checks collection and rejects skipped
tests. Ordinary repository discovery may skip these optional audit modules when
their separately pinned dependencies are absent.
