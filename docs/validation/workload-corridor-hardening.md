# Workload warning and corridor hardening

Completed September 10, 2026 on base `b058ac08ed08ed9e0069d5942328252c69cefb1a`.
Changes remain local; no release, remote workflow or manual acceptance session was
started. R1 real project inputs and macOS GitHub Actions follow-up remain deferred.
All remaining manual checks belong in one future session in the
[runbook](../../FOLLOWUP_RUNBOOK.md#consolidated-deferred-manual-session).

Verified source fingerprint:
`bc6670f7618ad10dd31f5d5c02584b413fb51fa174530349aecfd1ca3f2a3a21`.
This supersedes the source/preview identity in the
[initial completion report](automated-improvements.md).

## Extreme-workload warning

Normal user flow remains choose file, Analyze, review/export. Exceptionally large
or dense input pauses with **Continue anyway** and **Cancel**, recommending split
files or simplification of a copy only where required distance accuracy is retained.
No automatic simplification changes source geometry.

- Source lengths are measured before estimating full analysis segments per path.
  The segment advisory starts at 750,000, near the existing 1,000,000 hard cap.
- Density is checked after segmentation/indexing, before expensive overlap matching.
  Up to 256 deterministic count-only neighbor queries estimate work; no neighbor
  lists are materialized by this check. Warn at 10,000,000 estimated inspections,
  twice the existing 5,000,000 hard cap. The initially considered four-million
  advisory was raised to keep slow-but-viable workloads quiet.
- Import, measurement, indexing and sampling run on the worker. They precede the
  warning; this is not a prediction from compressed file size before reading it.
  Native library calls still have the existing cooperative cancellation limits.
- The worker waits without performing analysis. The shared Tk session handles
  Continue/Cancel; close also cancels the wait. One acceptance suffices per job.
  Synchronous/noninteractive calls never wait for a GUI response.
- Continue does not disable processing caps. A known over-segment-limit job returns
  measured source mileage with an explicit unavailable-overlap diagnostic without
  allocating those analysis segments. Estimated density is not a runtime promise;
  small localized hotspots can escape sampling.

Sources: [thresholds/estimator](../../src/pipeline_calculator/core/workload.py),
[execution control](../../src/pipeline_calculator/core/execution.py),
[shared GUI](../../src/pipeline_calculator/gui/controllers/analysis_session.py),
[analyzer](../../src/pipeline_calculator/core/analyzer.py).

## Reproduced geometry findings and fixes

Before implementation, the new targeted tests demonstrated:

1. **Bent fallback rectangles clipped the group.** A right-angle case had qualified
   sample points outside its fallback. The old rectangle used a narrow strip around
   the average lateral position. It now bounds all qualified samples on both axes,
   with end/margin padding, including partner samples omitted from the chosen
   representative centerline. Broad rectangles are disclosed as approximations.
2. **Straight outlines stopped at sample centers.** The 300 m straight case's source
   endpoints lay outside the displayed polygon. Extending the first/last outline
   ends by the existing segment-sized padding makes the visual extent more useful.
   This does not add qualified overlap or savings.
3. **An unused malformed bounding box blocked a valid preferred outline.** Backup
   box errors now matter only if no earlier geometry candidate can be used.
4. **Large self-crossing outlines bypassed topology checks.** A 600-point crossing
   ring with nonzero area previously passed unchecked. All sizes now use a sweep of
   edge bounds, limited to 250,000 active-edge inspections. Crossings or excessive
   check work select a simpler valid fallback with an explanation. A simple
   600-point circle still validates, preserving curves where practical.

KML descriptions now call the result an approximate bundled pipeline area. The
nominal strip width is not displayed as the width of a potentially broader fallback
rectangle. Local-plane geometry checks do not certify geographic topology in every
possible global-scale configuration or replace Google Earth review.

Sources: [shape construction](../../src/pipeline_calculator/core/overlap.py),
[serialized validation](../../src/pipeline_calculator/export/geometry_validation.py),
[KML export](../../src/pipeline_calculator/export/corridor_kml.py),
[reproductions](../../tests/test_grouping_hardening.py).

## Grouping review and numerical preservation

Reviewed continuity/path boundaries, representative sample selection, and disjoint
all-pairs savings allocation. Existing rules remain: continuous coverage on both
paths, separate multipart paths, no duplicate contribution by one pipeline within
a savings group, and every pair within that group must qualify. No new numerical
grouping defect was established by this pass; no grouping-rule changes were made.

All ten gallery scenarios preserve source totals, savings and section-length sets
when pipeline input order is reversed. Existing tests additionally cover mutually
compatible triples/quads, non-transitive chains, minimum thresholds, offset samples,
opposite digitization, invalid geometry and multipart continuity.

The [six-case comparison](hardening-comparison.json) reused exact input hashes from
the previous benchmark run. All numerical/group fields match exactly, including
source totals, savings, diagnostics and section lengths. Explicit exclusions are
only the listed visual fields; input directory names are normalized. Five previously
successful jobs produce no warning. The linked fixture already exceeded the hard
candidate limit before this change; it now warns before matching and retains the
same incomplete numerical outcome when the test context continues it. That case
is not claimed to have become a successful analysis.

## Validation and handoff

- **193 tests passed**, Python 3.11.9, no skips (full suite, 13.66 seconds).
- Source compilation, whitespace checks, planning/report links and consistent
  source fingerprints across comparison/gallery/package records passed.
- Eleven warning/control tests cover threshold behavior, quiet sparse jobs, dense
  input, sample cancellation, real worker pause/continue/cancel, source preservation
  at the hard limit, and both Continue and Cancel/close through native withdrawn Tk
  controls. Both entrypoints share this session; visible appearance remains deferred.
- Six numerical benchmark comparisons match, with warning outcomes as above. The
  recorded one-run times are sanity checks, not new performance guarantees.
- Thirteen gallery exports pass independent serialized geometry checks. In selected
  fixtures, source endpoint distance outside the shape changed as follows:

| Fixture | Before, m | After, m |
| --- | ---: | ---: |
| Straight | 5.003 | 0.000 |
| Right angle | 48.347 | 0.445 |
| Hairpin, section 1 | 79.838 | 0.000 |
| Loop | 56.417 | 0.000 |

These are visual observations, not mileage errors or operational accuracy tolerances.
Some paths extend beyond their qualified sections; remaining endpoint shortfalls
are not automatically defects. See [gallery table](hardening-gallery.md) and
[structured evidence](hardening-gallery.json). Before/after overlays remain in
`.validation-output/hardening-before-gallery/` and `.validation-output/hardening-gallery/`.

Both Windows previews rebuilt successfully and passed frozen smoke checks (embedded
version, withdrawn Tk, drag-and-drop library loading, source mileage, XLSX and
resources). [Artifact hashes and smoke results](hardening-packages.json) identify
the exact files under `%TEMP%/pipeline-calculator-hardening-20260910/{new,legacy}/dist`.
Both report `4.3-dev.b058ac08ed08.dirty`; use hashes to distinguish earlier previews.
Old previews and repository build/dist outputs were preserved.

A synthetic manual warning fixture is prepared at
`.validation-output/manual-packet/extreme-density.kmz` (200 coincident 300 m paths).
It was verified to reach the density warning and cancel before overlap matching;
its hash is in the package record. Use it for the future Continue/Cancel check,
with a normal gallery file for successful review/export. No private file is needed
to reproduce the warning. Continuing this intentionally excessive case can reach
the existing limit and show incomplete overlap results.

Reproduction commands from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/validation/check_hardening.py --baseline .validation-output/final-context --output docs/validation/hardening-comparison.json
.venv/Scripts/python.exe scripts/validation/build_gallery.py --output .validation-output/hardening-gallery
```

The comparison requires the retained local baseline files. Real project ground
truth, Google Earth rendering, full visible packaged interaction and macOS CI/native
acceptance remain deferred together; this report does not mark them passed.
