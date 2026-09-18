# Fixture creation audit

The audit strengthens the independent references and the production comparison.
The four main archives, three variants and their numerical expected results stay
unchanged. Production code is outside this fixture-creation task's scope.
[Regeneration evidence](regeneration_verification.json) verifies this bytewise;
seven JSON files changed only from Windows CRLF to platform-independent LF.

**September 15 application follow-up:** both baseline production defects below
are now corrected. The fresh fingerprinted receipt reports **104 adversarial tests,
10,175 application/export checks and 2,050 stress checks passing**, with zero
remaining mismatches in the seven core fixtures. The follow-up also closes two
comparison false-passes: missing interval coordinates and oversized or duplicated
corridor maps. See the [application review](../../../../../../docs/validation/state-fixture-review.md)
for fixes, ordinary CI coverage, real Alaska/Hawaii checks and explicit limits.

## Run and interpret

From the repository root, after installing the suite's pinned requirements:

```powershell
python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py audit
```

[The audit receipt](audit_report.json) records the actual test counts, fresh
reference and production results, file fingerprints, and remaining known defects.
It passes only when the adversarial tests and independent reconstruction pass,
every required production/export check runs, and no unexpected discrepancy exists.
Known production defects remain explicit; an audit pass does **not** certify that
production is defect-free. `suite.py compare` still exits unsuccessfully on them.

## Corrections made

| Area | Gap corrected | Executable evidence |
| --- | --- | --- |
| Geographic oracle | Wrapping already canonical longitudes could alter native coordinates. | [Geometry tests](../test_geometry_reference.py), [geometry audit](geometry_audit.json) |
| Very short visits | Overlapping root uncertainty brackets could collapse distinct boundaries and erase a genuine visit. Unresolved ordering now fails closed. | Geometry tests for distinct roots and the real 10 cm crossing |
| Exact endpoint touches | Collinear zero-length touches could be misread as unproved positive coincidence. | Geometry endpoint/coincidence controls |
| Shallow crossings | A universal 2 µm uncertainty floor could understate error. Bounds now account for local crossing slope and reject cuts exceeding the 1 cm target. | Analytic shallow-crossing test and [method](../../reference/README.md) |
| Sampled savings | Matching a total alone could conceal incorrect sources, paths, ranges, or public section attribution. | [26 sampled controls](../test_sampled_contract.py), [production comparison helpers](../application_contract.py) |
| Export acceptance | Strict native-border reclassification and zero-area floating comparison falsely rejected ordinary export rounding. | [Adversarial export tests](../test_application_contract.py) |
| Export completeness | Missing, duplicated, shifted or unrelated line/corridor geometry could escape aggregate mileage checks. | Independent interval spans, original-geodesic membership, source-pair identities and qualifying sample coverage |
| Follow-up export geometry | Whole-state corridor replacements, remote islands, borrowed section coverage and missing coordinates could pass earlier helper checks. | Independent section extent bounds, one-to-one matching and mandatory usable coordinates in [adversarial tests](../test_application_contract.py) |
| Validation workflow | Empty selections, incomplete inventories, stale green receipts, nonfinite numbers, skipped assertions, namespace collisions, partial publication and truncated previews could mislead. | [Harness tests](../test_suite_validation.py) |
| Stress profile | Aggregate totals and absent profiler counters were insufficient evidence. | [Stress runner](../run_stress.py), exact sampled-contract checks and counter validation |

Reference calculations remain independent of application functions. A mechanical
import audit supports that separation but does not prove algorithmic correctness.
Tests include hand-worked answers, analytic geometry and deliberately corrupted
observations that retain correct aggregate totals. The references and application
must independently reach the intended source-level result.

## Numerical acceptance

- Sample counts, source/path identities and qualifying ranges are exact. Savings
  allow only `1e-7 m` floating arithmetic tolerance; a 5 m sample error fails.
- Conservation uses `max(0.001 m, original × 1e-10)` independently of the 1 cm
  cut-location criterion. No positive interval is discarded by a length filter.
- Export vertices and segment midpoints must follow the original source geodesic
  within 10 µm. Independent interval identities establish ownership.
- Polygon rounding may occupy only a strip 64 local floating-coordinate units
  wide along the canonical boundary, bounded in area by perimeter and strip width.
  Holes stay excluded. A tiny polygon away from the state still fails.
- Numerical certificates depend on documented coordinate-evaluation assumptions;
  they are not formal interval-arithmetic or surveyed-ownership proofs.

## Historical baseline production defects — resolved in follow-up

1. **02:** longitude normalization moves an exact native border vertex by one
   floating-point unit, producing a false Texas tail and 10 crossings instead of 9.
2. **04 and its redundant-vertex variant:** endpoint arithmetic creates an
   `8.881784197001252e-16 m` unresolved fragment and incorrectly marks analysis
   incomplete. The reversed case passes.

[Minimal reproductions](../reproductions/README.md) retain evidence of both original defects.
[Classification rules](../known_application_failures.py) require exact archive
fingerprints and narrow measured signatures. They never waive unrelated checks;
new corruption on an affected source is still an unexpected regression.

The old [polygon residue](../reproductions/polygon_residual.json) and
[endpoint certificate diagnostic](../reproductions/export_precision.json) are
retained as observations, not current production-failure classifications.

## Limits

The audit covers the delivered inputs and named adversarial cases. Synthetic
dateline/hole controls do not establish real Alaska, Hawaii or every native
boundary topology. Arbitrary tangencies, winding shared rivers, allocation-only
state maps and all workloads remain outside the demonstrated coverage. Timing
and memory measurements describe the recorded host, not a performance guarantee.
