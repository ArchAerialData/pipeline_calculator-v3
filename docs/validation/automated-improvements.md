# Automated improvement verification

Historical evidence for the initial automation batch. The later
[workload/corridor hardening report](workload-corridor-hardening.md) supersedes its
source/artifact identity, geometry-validation limits and current handoff status.
Keep the measurements below as baseline evidence; the earlier preview hashes do
not identify the latest source.

Implemented September 10, 2026 on base commit `b058ac08ed08ed9e0069d5942328252c69cefb1a`.
Changes remain local and uncommitted. No release/tag was published.
Packaged source SHA-256: `084561ee8c9c06e0e61d9eb2f88a60bd30a8d1c4d5764fbd8e742e6140303423` (ordered relative Python paths
and file bytes, as calculated by `scripts/validation/common.py`).

## Delivered behavior

- Both GUIs use one analysis session/controller with atomic terminal publication,
  cooperative cancellation, safe polling/close cleanup, job identity checks and
  Retry selected file. Workers never update Tk widgets.
- Stage/count/elapsed snapshots use bounded storage and throttling. Existing numeric
  callbacks and synchronous analysis still work. No fabricated overall percentage.
- Corridor launch happens off the Tk thread. OS errors/exit codes/timeouts are
  surfaced with the saved path and Copy Path, Save As, Retry and Close. The string
  compatibility wrapper raises an error retaining the path when launch fails.
- KML checks finite/range-valid coordinates, serialized precision, distinct vertices,
  area and bounded small-ring topology. Fallback rectangles are described explicitly.
  Numerical mileage rules and processing limits are unchanged.
- Corpus, benchmark, cancellation, independent reference and geometry gallery tools
  produce reproducible local evidence. Build/test helpers propagate failure status;
  Windows builds support isolated output directories.

Implementation sources: [execution context](../../src/pipeline_calculator/core/execution.py),
[job controller](../../src/pipeline_calculator/gui/controllers/analysis_controller.py),
[shared session](../../src/pipeline_calculator/gui/controllers/analysis_session.py),
[launch recovery](../../src/pipeline_calculator/gui/dialogs/corridor_dialog.py),
[geometry guard](../../src/pipeline_calculator/export/geometry_validation.py).

## Validation results

| Check | Outcome |
| --- | --- |
| Original suite | 106 passed on system Python 3.13 and project Python 3.11 |
| Final integrated suite | **166 passed** on Python 3.11.9; no skips |
| Dependency consistency | `python -m pip check`: no broken requirements |
| Corpus | Eight authored synthetic inputs pass; tracked KMZ explicitly unreviewed; JSON/XLSX round-trips and manifest/hash failures tested |
| Independent reference | 280 comparisons; all projection checks pass; operational error tolerance is not assumed |
| Geometry gallery | 13 exported sections pass independent local/serialized checks; Google Earth pending |
| Performance/output preservation | Six full benchmark outputs match after excluding three additive geometry fields and normalizing input directory names |
| Dense context overhead | 2.30% vs final disabled mode; five measured runs after warm-up |
| Cancellation | Ten requested stages cancelled with no partial result; max observed acknowledgement 93.44 ms (sorting candidates) |
| Windows packaging | Modular and legacy previews built; both frozen smoke checks passed with matching embedded versions and synthetic results |
| macOS | Shell-helper failure behavior tested with Git Bash on Windows; native build/visual execution not performed |

Runtime: Python 3.11.9, NumPy 1.24.3, SciPy 1.15.3, pyproj 3.7.2,
CustomTkinter 5.2.2, PyInstaller 6.16.0, pytest 9.1.1. The only installed addition
was development-test dependencies in the existing venv; runtime requirements were
not changed. A later system-runtime test run is supplementary, not packaged-runtime
acceptance.

## Performance decisions

[Comparison table](comparison.md) and [machine-readable comparison](comparison.json)
record times and peak RSS. Dense execution changed from 1.960 to 1.990 s; the final
context-disabled mode measured 1.945 s. Linked-document execution measured 13.873 s
versus 14.256 s baseline, with about 534 MiB peak worker RSS. These are controlled
synthetic workloads on this machine, not guarantees for operational projects.

Initial per-item checkpoint calls exceeded the progress target. Profiling found
about 1.84 million checkpoint calls in the dense case. Indexed checks every 256
items and removing redundant checks in bounded nine-neighbor loops reduced that
overhead while retaining cancellation coverage. No numerical algorithm or accuracy
tradeoff was introduced. Broader data-structure changes were not justified by the
available evidence and were not pursued. Subsecond percentage changes are noisy.

[Cancellation measurements](cancellation.json) identify requested/observed stage and
acknowledgement delay. A stage may advance before the requesting thread is scheduled;
these single-run observations do not bound arbitrary filesystem/library call latency.

## Numerical and geometry limits

[Reference model](reference-model.md), [error table](reference-results.md) and
[full comparisons](reference-results.json) keep model distortion, section sums,
unique interval coverage and savings comparisons distinct. Common-axis groups up
to six pipelines have an exhaustive reference. Arbitrary branched/curved group
optima are outside this declared model. Savings differences combine sampling and
grouping; they are not falsely labeled pure heuristic loss.

The report includes large discontinuities at exact range/minimum thresholds and
coarse hairpin cases. A tiny geometric difference can change whether a long section
qualifies. These observations are not silently repaired by changing business rules
or broadening acceptance tolerances. Runbook R2 receives the actual cases and must
decide operational tolerances or a separately scoped semantics change.

[Gallery](gallery.md) and [numerical gallery evidence](gallery.json) show confirmed
self-crossing preferred shapes for right-angle, hairpin and loop examples. Valid
existing rectangle fallbacks are labeled approximate. Source-path endpoint
shortfalls are reported separately because a section can cover only part of a
source path. A rectangle can omit bends and include unrelated area; it is not
advertised as a precise survey boundary.

Production topology checks are capped at 512 ring vertices to avoid unbounded
quadratic export work. Larger rings explicitly disclose that detailed topology
has not been validated. Gallery checks are independently bounded at 4,096 vertices.
Neither these checks nor frozen smoke substitute for Google Earth/native visual
review. The app's sampled computation method remains `qualified_segment_coverage_v2`.

## Reproduction

Run from the repository root with the existing venv:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe scripts/validation/run_corpus.py --suite local --output .validation-output/corpus-final
.\.venv\Scripts\python.exe scripts/validation/compare_reference.py --output .validation-output/reference
.\.venv\Scripts\python.exe scripts/validation/build_gallery.py --output .validation-output/gallery
.\.venv\Scripts\python.exe scripts/benchmarks/cancellation.py --output .validation-output/cancellation
.\.venv\Scripts\python.exe scripts/benchmarks/run.py --suite medium --repeat 5 --context --output .validation-output/final-context
```

The original baseline is retained in `.validation-output/baseline-medium`. Raw
profiles and generated files stay ignored/local. Existing project build/dist
folders were not cleaned: preview build output uses the dedicated
`%TEMP%/pipeline-calculator-validation-b058ac0/new` and `legacy` directories.
[Artifact/source hashes and smoke outcomes](packages.json) record both completed
checks. Full machine-specific paths are retained in `.validation-output/package-evidence.json`. Preview version: `4.3-dev.b058ac08ed08.dirty`; a dirty version alone does not
identify a unique patch, so retain the accompanying source/artifact hashes.

A6 used isolated **output directories against the exact working source**, supported
by the new `-OutputRoot`, instead of duplicating the whole checkout. This preserves
user artifacts and avoids testing an older HEAD. Moving the spec directory initially
exposed relative resource paths; those build arguments now use absolute source paths.
Both corrected previews built and ran successfully. Compilation and `git diff --check` also pass.

## Follow-up

[FOLLOWUP_RUNBOOK.md](../../FOLLOWUP_RUNBOOK.md) contains the remaining owner/platform
work: approved real projects and expected results (R1), operational/model decisions
(R2), native visuals/macOS (R3), production budgets (R4), Google Earth (R5), and optional
access/distribution (R6). The local automated scope does not need those decisions
reopened to complete. No sensitive geometry was copied into these checked-in reports.
