# State-aware fixture review and corrections

The delivered KMZ suite exposed two application precision defects and two gaps in
its comparison helpers. This follow-up fixes the application defects, strengthens
the independent checks, and brings the seven saved KMZs into ordinary application
test discovery. Fixture geometry and independently calculated expected values are
preserved.

## Confirmed application defects

1. **An exact endpoint touch became a border crossing.** Converting already
   canonical boundary longitudes to radians and back moved a native NM/TX vertex
   by one floating-point unit. The resulting invented Texas tail was about
   `1.27e-9 m`, adding a crossing and a source to Texas. Boundary unwrapping now
   operates directly in degrees, retaining unchanged native coordinates.
2. **Endpoint arithmetic made a valid result incomplete.** A four-meter Texas
   approach ending exactly at a shared meridian produced an unresolved
   `8.881784197001252e-16 m` fragment in one direction. The partitioner now proves
   source-endpoint membership on the native edge with exact binary arithmetic
   and retains that endpoint's canonical distance along the path during root
   refinement. It does not remove short intervals or snap nearby geometry.

The baseline comparison reproduced seven failed assertions in fixture 02 and ten
each in fixture 04 and its redundant-vertex variation. Their original standalone
[reproductions](../../tests/fixtures/geography/pipeline_kmz_regression_suite/validation/reproductions/README.md)
remain historical evidence; they are not updated to disguise the baseline failures.

Changes: [boundary normalization](../../src/pipeline_calculator/core/geography/boundaries.py),
[partitioning](../../src/pipeline_calculator/core/geography/partition.py), and
[20 focused regressions](../../tests/test_state_precision_regressions.py).
The regressions include reversed paths, inserted vertices, oblique exact endpoints,
nearby non-coincident endpoints, dateline holes, and real positive 10-micrometer
crossings. Numerical precision describes the boundary model, not surveyed accuracy.

## Independent-check improvements

The original corridor comparison accepted a polygon replacing a local corridor
with the entire Texas state polygon: it still covered the expected samples and
remained in Texas. Checks now require independently bounded section extents and
one-to-one assignment between exported corridor placemarks and expected sections.
This rejects oversized maps, remote extra polygon components, and duplicated
geometry borrowed from a different section of the same source pair.

A separate mutation showed that dropping a canonical fragment's coordinates
skipped its geometric checks while retaining valid accounting fields. Fragment
geometry must now be present and usable before its interval can pass.

These are demonstrated validation weaknesses, not observations that the application
actually exported either corrupted result. The strengthened
[helpers and adversarial tests](../../tests/fixtures/geography/pipeline_kmz_regression_suite/validation/test_application_contract.py)
keep the frozen mileage and overlap goldens unchanged. Corridor extent bounds are
conservative enough for supported curved, rectangular, and bounding-box fallbacks;
they are not exact polygon-shape goldens.

Comparison and audit receipts now include application source-file fingerprints,
so an uncommitted repair is distinguishable from its Git base. Generated suite
documentation derives the latest comparison status from matching fingerprints
and preserves the initial defects as history.

## Additional regression coverage

- [Seven saved KMZs in both analysis modes](../../tests/test_state_kmz_regression_suite.py):
  archive/resource integrity, source/path counts, source-level ownership, crossings,
  original lengths, exact sampled savings, section counts, shared allocations,
  completion status, and conservation. These run without optional reference packages.
- [Four additional integration cases](../../tests/test_state_additional_edges.py):
  real shared-only TX/NM export with no empty state folders; Alaska/Hawaii multipart
  geometry with an Aleutian dateline exit; isolated Combined overlap failure; and
  isolated state overlap failure. They check maps, workbook, and JSON behavior.

The Alaska/Hawaii test measures the saved coordinates and independently brackets
the chosen coverage exit against native WKB polygons. It covers those chosen paths,
not every island or arbitrarily short possible coastal reentry.

## Verification

Completed September 15, 2026 against fixture commit `26d585c`, with the application
repairs uncommitted. [Machine-readable results](state-fixture-review.json) include
application source fingerprints and evidence hashes.

| Verification | Result |
| --- | --- |
| Full application regression suite, Python 3.11 | 468 passed in 260.15 seconds |
| Independent/adversarial audit tests, pinned Python 3.13 | 104 passed; no skips |
| Independent reconstruction | All seven core archives reverified |
| Application and KMZ export comparison | 10,175 checks; zero mismatches across all seven files |
| Larger overlap stress profile | 2,050 checks passed; 96 sources and 27,120 samples |
| Stress application time | 7.18 seconds on this host, including profiling overhead |
| Frozen archives and expectation/ledger assets | All 37 unchanged |

The ordinary application run skipped five optional reference test modules because
its Python 3.11 environment lacks the separate audit dependencies. All 104 tests
from those modules ran successfully in the dedicated Python 3.13 audit. Every
core application comparison passes normally; no known-defect exception was needed.

An additional bounded native-endpoint investigation preserved all 15,282 unique
Alaska and 3,502 Hawaii resource vertices exactly. Of 100 selected endpoint cases,
98 were independently certified and passed, including four long dateline traversals;
zero mismatches were found. Two near-tangent Hawaii cases could not be certified
by the independent reference and are explicitly excluded from passing evidence.
The largest scope residual in certified cases was approximately `1.06e-9 m`.
The bundled AK/HI polygon components did not require longitude unwrapping, so this
does not establish arbitrary synthetic wrapped-ring behavior.

The raw before/after commands and receipts are retained under
`.validation-output/state-fixture-review/`. The dedicated independent audit uses
the suite's pinned Python 3.13 environment; ordinary application tests also run in
the repository's Python 3.11 environment. Optional reference dependencies are not
required for ordinary application regression tests.

This is source-level Windows validation. It does not establish native macOS or
fresh packaged-build behavior, arbitrary tangencies, every boundary topology, or
all workload sizes. Earlier release gates are not cleared by these fixture checks.
