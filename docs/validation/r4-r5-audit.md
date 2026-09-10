# R4/R5 implementation audit

September 10, 2026. Audited base `9583285a7d368690558372ec27fefe3e98587883`
(Bug Fixes 3). Status: **complete: all reproduced findings fixed, source checks and
both refreshed Windows artifact checks passed**. Changes remain local and uncommitted.

Verified Python source fingerprint:
`9cc51970dc1422e9d7ab43109206aef1902ffb061bac2dec3fd7606c82e73b6a`.
This report supersedes the active source/package status in
[the initial R4/R5 hardening report](workload-corridor-hardening.md).

## Scope and findings

Reviewed workload estimates, warning publication/Continue/Cancel/close, GUI layout,
segment allocation limits, grouping continuity and savings, serialized geometry,
fallback selection and bounded topology work. Regressions were reproduced before
fixing them; no changes to grouping policy, distance units or user defaults were made.

| Finding | Verified failure | Resolution |
| --- | --- | --- |
| R4 density sampling blind spot | A 16,321-point repeated layout put every evenly spaced sample on an isolated point despite over 98% of the points being coincident; no warning appeared | Repeatable seeded sampling within 256 strata, weighted by stratum size; avoids periodic alignment while keeping the same bounded count-only query cost |
| R4 aggregate allocation | A reduced four-segment project cap was checked after six segments across two parts were already allocated | Pass the remaining project budget through each pipeline/path; reject before exceeding it, including direct core-helper callers |
| R4 trailing remainder | With a one-segment cap, a 5.5 m path at 5 m spacing was rejected despite needing only one full segment | Compare the next full-segment boundary with the budget; ignored remainders do not consume another full segment |
| R4 estimate overflow | A finite `1e-320` step overflowed the workload estimate, preventing valid source mileage from returning | Saturate estimates above the supported budget before division; preserve source totals and return an explicit unavailable-overlap diagnostic; zero-length paths remain valid |
| R4 small-window layout | Long warning/name content requested 561 physical pixels in the initial layout and could overflow a smaller resized window | Scroll text in a responsive body; reserve Continue/Cancel and progress in a separate footer; adapt wrapping to available width |
| R5 unbounded preprocessing | The edge-inspection budget applied only after reading, projecting and sorting the entire polygon input | Enforce 100,000 raw input points before expensive preprocessing, including duplicate streams; use disclosed fallback if exceeded |
| R5 repeated fallback validation | A shared invalid preferred/oriented polygon object was validated twice | Attempt each distinct polygon object once; preserve fallback ordering and error details |

Implementation: [workload](../../src/pipeline_calculator/core/workload.py),
[source measurement](../../src/pipeline_calculator/core/analyzer.py),
[segmentation](../../src/pipeline_calculator/core/segmentation.py),
[multipart propagation](../../src/pipeline_calculator/core/coordinates.py),
[overlap](../../src/pipeline_calculator/core/overlap.py),
[shared GUI](../../src/pipeline_calculator/gui/controllers/analysis_session.py),
[geometry guard](../../src/pipeline_calculator/export/geometry_validation.py).

Additional review retained the existing cancellation design: deterministic tests
cover cancellation before a warning, while it is being published, while waiting,
and immediately after acceptance. No wait hangs or successful cancelled results
were observed. Polling now reads a single warning snapshot rather than reading the
mutable value twice. Native widget checks cover long filenames and 800x400/400x400
logical windows, accounting for Windows DPI scaling; visible user acceptance remains
deferred. The controls use the same session in both application entrypoints.

## Validation

- **210 tests passed**, no skips, Python 3.11.9, 14.81 seconds. Added 17 audit test
  cases, including resource boundaries, sampling determinism/cost, cancellation
  races, tiny-step and zero-length inputs, fallback budgets, and small-window layout.
- **180 grouping-order evaluations**: 30 seeded four-pipeline layouts, six input
  permutations each, including duplicate names, shifted endpoints, overlapping and
  non-transitive proximity. Savings stayed invariant. Existing continuity,
  multipart, reversed-digitization and gallery order checks still pass.
- **60 independent polygon comparisons**: shuffled crossing outlines and simple
  radial shapes agree with the separate local-plane intersection checker. Existing
  large-circle, crossing, dateline/polar, hairpin/loop, serialized rounding and
  fallback extent cases continue to pass.
- [Six saved benchmark comparisons](r4-r5-audit-comparison.json) use identical input
  hashes and preserve all numerical/group results under the documented visual-field
  exclusions. Five successful inputs remain quiet; the previously over-limit linked
  case still warns and retains the same incomplete outcome after continuation.
- [Thirteen refreshed gallery sections](r4-r5-audit-gallery.md) pass independent
  serialized geometry checks; [structured evidence](r4-r5-audit-gallery.json) records
  the exact source fingerprint. No viewer result is inferred from these checks.
- Source compilation, whitespace checks, documentation links and consistent source
  fingerprints across comparison/gallery/package records pass.
- Both modular and legacy Windows previews rebuilt successfully and passed frozen
  smoke (version, Tk/widgets, DnD library, source-distance/XLSX and bundled resources).
  The synthetic manual density fixture still reaches its warning and cancels before
  matching. Exact package and fixture hashes are in the handoff record below.

Regression source: [audit tests](../../tests/test_r4_r5_audit.py).
The six-case comparison reuses the retained `.validation-output/final-context`
baseline; its source identity is recorded rather than inferred from a version label.

## Handoff and remaining limits

The advisory thresholds remain deliberately high: **750,000 estimated segments**
and **10,000,000 estimated neighbor inspections**. Hard caps remain 1,000,000
segments and 5,000,000 candidate inspections. Exceeding the source-measured segment
cap now says overlap exceeds supported size; Continue retains source mileage with
an incomplete-overlap notice. Neither threshold changes source geometry or mileage.

Geometry validation now has both a 100,000 raw-point cap and a 250,000 active-edge
inspection cap. Exceeding either chooses an explicitly described simpler outline.
These guard preprocessing cost as well as intersection work; they do not claim
that every global-scale geometry can be certified in a local projection.

Windows preview destination:
`%TEMP%/pipeline-calculator-r4-r5-audit-9583285/{new,legacy}/dist`.
Version: `4.4-dev.9583285a7d36.dirty`. Earlier previews are preserved. Use
[artifact hashes and smoke evidence](r4-r5-audit-packages.json) to identify the exact
content; the version label alone is not a content fingerprint.

The [consolidated manual session](../../FOLLOWUP_RUNBOOK.md#consolidated-deferred-manual-session)
remains deferred: R1 real project files, independent project-distance expectations,
Google Earth rendering, visible packaged workflows and the macOS GitHub Actions
follow-up. No user files, remote workflows, signing or release publication were
requested during this audit. Sampling can still miss a localized hotspot, and
cooperative cancellation still waits for a current native library call to return.

No confirmed defect from this audit remains unresolved. This is bounded automated
evidence, not a claim that untested real projects or external viewers are bug-free.

Reproduce source checks from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/validation/check_hardening.py --baseline .validation-output/final-context --output docs/validation/r4-r5-audit-comparison.json
.venv/Scripts/python.exe scripts/validation/build_gallery.py --output .validation-output/r4-r5-audit-gallery
```
