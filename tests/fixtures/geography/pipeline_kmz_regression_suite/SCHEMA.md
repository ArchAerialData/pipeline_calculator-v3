# Expectation schema 1.0.0

The canonical files are `expected/<archive stem>.expected.json`. JSON values use
meters without presentation rounding. Survey miles divide by **1609.347218694**.
Schema versions change when field meaning changes. Additional descriptive fields
may be added without changing an existing field's meaning. The executable schema
checks live in [suite.py](suite.py), [geometric accounting](reference/geometry.py),
and [intent assertions](validation/coverage_checks.py).

## Top-level document

| Field | Meaning |
| --- | --- |
| `schema_version` | `1.0.0` for the four main archives and three variations. |
| `fixture` | Path relative to `fixtures/`, including `variants/` when applicable. |
| `baseline_commit`, `boundary_sha256` | Frozen public-behavior baseline and exact boundary archive fingerprint. |
| `analysis_profile` | Named GRS80/default profile, conversion factor, 5 m samples, 15 m range, 200 m minimum, 15 degree orientation tolerance. |
| `archive` | Exact SHA-256/bytes; XML element counts; ordered source keys and unique XML IDs; source/path/vertex counts; repeated names and OBJECTIDs; deterministic ZIP metadata. Parallel XML-order and XML-ID arrays map each source identity without using parser integer IDs. |
| `geometry` | Independently measured original paths, state intervals and allocations, crossing and touch evidence, certification, and exclusive inputs to state overlap analysis. |
| `analyses` | One `Combined` sampled result and one result per represented state code. |
| `expected_application_status` | Complete status, zero outside/unresolved mileage, exact expected informational diagnostics, and failure policy for unexpected warnings/errors. These are requirements, not observations from the application. |
| `reference_status` | `reference-verified` is emitted only after accounting and intent assertions pass. An exception prevents writing a new golden. |
| `coverage_checks` | Named assertions with motif, stable source keys, numerical assertion, measured evidence and Boolean result. |

## Geometric reference

`original_meters` and `original_survey_miles` describe the entire archive;
`combined_original_*` are explicit aliases. `sources` preserves XML order and
records `key`, `name`, `motif`, `xml_order`, original lengths, and `paths`.
Each path has its original path index, vertex count, original length, per-state
accounting, and interval IDs. Each source also records conservation differences,
the independently remeasured fragment-geometry difference, tolerance and result.

`intervals` is the authoritative ordered ledger. Each row contains:

- `id`: stable source key plus original path index and interval index.
- `key` / `source_key`: equivalent stable fixture identity.
- `path_index`: original zero-based path, never a fabricated connection.
- `start_m`, `end_m`: increasing chainages along that original path.
- `kind`: `interior`, `shared`, `outside`, or `unresolved`.
- `state_codes`: one state for an interior; all independently verified adjoining
  states for a shared interval; otherwise empty unless ownership is unresolved.
- `length_meters`: source-chainage interval length. No length is rebalanced to
  make a state total match.
- `coordinates`: geodesic fragment endpoints and all intervening original
  vertices, in source order. Altitudes are excluded from measurement.
- `start_error_bound_meters`, `end_error_bound_meters`: separate endpoint
  uncertainty bounds. `cut_error_bound_meters` is their maximum.

`states` and each source/path's `states` map state codes to name, exclusive
`interior_meters`, `shared_allocation_meters`, `attributed_original_meters`,
survey-mile conversion, `length_error_bound_meters`, and explicit
`shared_interval_references`. A state's bound sums its incident endpoint bounds,
dividing by the number of adjoining states for each shared allocation.

`shared_meters` counts shared geometry once. `shared_allocations` references each
canonical interval once per adjoining state, with `allocated_meters` equal to
the interval's unrounded length divided equally. `outside_meters` and
`unresolved_meters` are physical interval totals. `represented_states` includes
positive allocations as well as interiors.

`crossings` records source/path, chainage, coordinate, state transition,
`angle_degrees` relative to the local native boundary tangent, uncertainty and
`via_shared_interval`. A transition through shared geometry is recorded at entry
into the following exclusive interior; it is not a pointwise transverse crossing.
`crossing_count` counts these transitions. `endpoint_touches` instead records
zero-positive-length contact with another state, with start/end endpoint and
verified interior/touched state codes.

`state_inputs` contains exclusive fragment paths grouped by stable source.
`interior_fragments` is the same geometry indexed by state then source key, with
original path/interval/chainage references. These redundant views are convenience
inputs; accounting comes from `intervals`. Neither view includes allocated shared
geometry. `certification` documents the method/domain, endpoint uncertainty,
root refinement convergence, conservation and GRS80 distance crosschecks.

`root_bracket_width_target_meters` is the root refinement target, not the total
cut uncertainty. For refined roots the endpoint bound adds the larger of the
2 µm station floor and the coordinate evaluation allowance divided by a
certified derivative lower bound. The historical
`coordinate_roundoff_allowance_meters` field records that **floor**, while
`maximum_cut_error_bound_meters` records the achieved full bound. Shallow roots
that cannot meet `cut_target_meters` (0.01 m), and distinct cuts whose order cannot
be certified, fail instead of producing complete expectations. These remain
conditional engineering bounds; see [method and assumptions](reference/README.md#numerical-uncertainty-and-limits).

## Sampled-contract reference

Every entry in `analyses` contains:

| Field | Meaning |
| --- | --- |
| `reference_version`, `profile`, `status`, `diagnostics` | Versioned independent method, frozen parameters and completion state. |
| `source_count` | Distinct source inputs in that scope; a source with only shared allocation is absent from state overlap inputs. |
| `original_meters`, `original_survey_miles` | Physical input length. **For a state this is exclusive interior length**, before adding shared allocation. |
| `sample_count`, `sampled_meters` | Full 5 m samples, restarting per independent path/fragment. |
| `sections` | Every connected positive matching component, including rejected sub-threshold sections. |
| `qualifying_section_count`, `rejected_section_count` | Counts by the section's Boolean `qualified` decision. |
| `qualifying_pairwise_meters` | Sum of qualified section lengths. This is **not savings** and may count a three-line corridor three times. |
| `source_coverage` | Per-source original/path lengths, full sample counts, unsampled tails, uniquely covered qualifying sample totals, and coverage ranges by path. |
| `savings_meters`, `savings_survey_miles` | Disjoint compatible group savings, never allocated to individual member pipelines. |
| `adjusted_meters`, `adjusted_survey_miles` | Combined original less savings; state attributed original less exclusive interior savings. |
| `motif_savings` | Sum of actual group savings by construction label, not an allocation to individual pipelines. Mixed-motif groups are explicitly labeled. |
| `grouping` | Group-size histogram, merge/rejection counts, verified disjointness/clique status, canonical group hash and savings by complete source-member set. |
| `workload` | Actual exhaustive sample pair visits, conservative screen survivors, geodesic tests and accepted pair count. These are reference workloads, distinct from application spatial-index counters. |

Sections record two source keys, two path indices, qualification, the smaller
covered length, both coverage sample counts/meters, half-open coverage ranges,
eligible cell count and measured transverse/longitudinal extrema. For example,
`[3, 8]` means sample indices 3 through 7, covering 25 m. State section path
indices refer to the state's fragment input list; use `fragment_references` to
recover the original source path and interval. Full exact sets are recoverable
from the ranges.

State results additionally contain `interior_meters`,
`shared_allocation_meters`, `attributed_original_meters`, its survey conversion,
and `fragment_references`. The accounting identities are:

```text
attributed original = interior + shared allocation
state adjusted = attributed original - qualified interior savings
sum(state attributed original) + outside + unresolved = Combined original
```

## Companion files and reports

- `.intervals.csv`: flat canonical interval ledger, including endpoint errors.
- `.overlaps.csv`: one row per scope/section, including qualification, both source
  identities/path indices, exact counts and JSON-encoded half-open ranges.
- `.crossings.csv`: flat event ledger; zero-length touches remain in JSON.
- `validation/reference_report.json`: dependency/hardware evidence, source-file
  fingerprints, no-application-import audit, hand controls, per-fixture status,
  exact regeneration checks, variant comparisons and the saved stress reference.
  `status`, `passed`, and `selection` distinguish completed full validation from
  running, failed or focused work. `artifact_sha256` fingerprints the required
  archives, expectations, CSV ledgers, design manifest, previews and available
  stress artifacts; `independent_module_audit` fingerprints reference, generator
  and intent-check code. Runtime is observational.
- `validation/application_comparison.json`: later application observations,
  diagnostics, source/path and interval-ledger checks, exact qualifying sample
  coverage, numeric differences and export checks. It cannot alter an expectation.
  Exported lines are checked against independent source spans and canonical
  intervals. Polygon containment retains holes and allows only a locally bounded
  floating-point residue; the recorded width/area bounds are not a general
  geographic buffer. See [comparison checks](validation/application_contract.py).
- `validation/audit/geometry_audit.json`: adversarial geographic tests, corrected
  defect reproductions, code hashes and the comparison of all seven geometries
  with unchanged goldens. This audit observation does not replace current full
  archive validation or application comparison.
- `validation/design_manifest.json`: generator version/seed, input identities,
  intended motifs, native boundary evidence and variant archive fingerprints.
- `expected/stress/*.expected.json` uses the separate `stress/1.0.0` schema:
  `profile`, `groups`, `archive`, `geometry` and `sampled` have the same meanings
  above. Its measured application work/runtime/memory is in `stress_report.json`.

A stored `passed: true` is insufficient for readiness. The
[freshness check](suite.py) requires a complete passing reference report whose
artifact and independent-code hashes still match. Validation writes a running
receipt before work and a failed receipt on errors; focused results have separate
filenames. The fixed archive/expectation inventory and required ledgers prevent
an empty glob or missing fixture from producing a zero-test success. Numeric
comparisons reject nonfinite numbers and Boolean/count confusion; sample-count
or 5 m savings mistakes cannot use a general geometry tolerance.

[Adversarial tests](reference/README.md#adversarial-audit) exercise these failure
paths. Known application defects remain failed observations in the raw comparison;
their [separate classification](validation/known_application_failures.py) uses
archive hashes and narrow failure signatures without changing the goldens or
excusing unrelated assertions.

Full-precision values are numerical estimates with explicit uncertainty, not
surveyed ownership or infinite-precision geometric facts.

`validation/audit/audit_report.json` distinguishes an audit pass (no unexpected
regressions) from `application.fully_passing`. It records collected test counts,
fresh core and stress checks, narrow known-defect classifications and code/evidence
fingerprints. Known failures remain failed checks in the ordinary application
report. Audit never changes the frozen expectations to match production.
