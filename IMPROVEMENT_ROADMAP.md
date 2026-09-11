# Verified improvement implementation plan

Latest follow-up: **dark-mode/DPI visual polish implemented**, September 10, 2026.
See [visual polish validation](docs/validation/visual-polish.md) for the layout matrix,
238 passing tests and corrected Windows startup verification. This follow-up changes presentation,
not pipeline distance or corridor geometry calculations.


Reviewed September 10, 2026 at commit `b058ac08ed08ed9e0069d5942328252c69cefb1a`.
Status: **automated implementation, requested hardening and available preview
verification complete**, September 10, 2026. Latest source checks: 210 tests, six unchanged
numerical comparisons, thirteen gallery sections and both refreshed Windows smokes.

R4/R5 audit on `9583285`: seven reproduced gaps fixed, including periodic density
blind spots, budget enforcement/overflow and polygon preprocessing limits. Full
source checks and both refreshed artifact checks pass. See the
[audit report](docs/validation/r4-r5-audit.md) for current evidence and completion.

[Initial verification report](docs/validation/automated-improvements.md): 166 passing tests,
verified Windows modular/legacy previews, six equivalent-output benchmarks,
280 reference comparisons and 13 validated approximate gallery exports.
External/owner acceptance remains in [FOLLOWUP_RUNBOOK.md](FOLLOWUP_RUNBOOK.md).

Owner clarification, September 10, 2026: **total source-polyline distance is the
primary result**. Grouping/overlap are supplemental; corridor shapes are rough
visual guides to identified bundled segments. Their purpose is settled, with no
surveyed-boundary or flight/sensor requirements. Favor reliable operation while
preserving source-distance accuracy and visible incomplete/error reporting; no
arbitrary numerical error allowance has been approved.

Additional testing and owner acceptance are **deferred to future work**, including
macOS follow-up through the existing GitHub Actions runner and Google Earth review.
The completed evidence above remains valid within its stated limits. R1 needs real
examples only when that review resumes; R4 is optional performance investigation,
with no extra everyday workflow or immediate hardware/runtime decision required.
See the runbook for these decisions and the verified recent corridor-change history.

Subsequent owner direction: R1 real project inputs are explicitly deferred. Ordinary
performance is satisfactory; R4 now includes an automated warning only for extreme
workloads. R5 includes further autonomous investigation/fixes before manual review.
Complete source checks and refresh previews first, then combine outstanding manual
acceptance into one future session. See
[hardening evidence](docs/validation/workload-corridor-hardening.md).

## Implementation ledger

| Task | Status | Evidence |
| --- | --- | --- |
| A0 | Complete | 106 passed on both Python 3.13 and project Python 3.11; pytest installed only in venv |
| A5 / A3.1 | Complete | Corpus tools + 4 tests; all synthetic expectations pass; six medium benchmark baselines saved under `.validation-output/baseline-medium` |
| A1 / A2 / A8 | Complete | Shared lifecycle/retry/progress and asynchronous launch recovery; focused and full-suite checks pass; 10 measured cancellations <=94 ms |
| A4 / A7 / A3.2 | Complete with stated model limits | 280 reference comparisons; 13 valid approximate gallery exports; all six benchmark outputs equal; dense context overhead 2.30% |
| A6 / A9 | Complete for available environment | Both Windows previews and frozen smoke pass; 166 tests; compile/diff checks pass; macOS native/interactive checks deferred to R3; report and runbook populated |
| A10 (R4) | Implemented and source-verified | 750,000 segment / 10,000,000 estimated neighbor warning; shared Continue/Cancel; early cap preserves source mileage; successful benchmarks stay quiet |
| A11 (R5) | Implemented and source-verified | Clipped rectangle/endpoint fixes; resilient fallback selection; bounded large-ring checks; grouping order invariance; 193 tests pass and six numerical comparisons unchanged |
| A12 (R4/R5 audit) | Complete | Seven reproduced gaps fixed; 17 audit cases added; 210 full-suite tests; 180 order evaluations and 60 independent topology comparisons; both Windows previews rebuilt/smoked; manual review consolidated/deferred |

### A10/A11 implementation and acceptance

1. Estimate full analysis segments per path during source-distance measurement;
   warn at 750,000. After indexing but before overlap matching, use up to 256
   count-only neighbor queries; warn at 10,000,000 estimated inspections. Preserve
   the existing 1,000,000 segment and 5,000,000 inspection hard caps.
2. Share one pending warning through the execution context. Both GUIs show Continue
   anyway and Cancel on the Tk thread. Waiting does no analysis; cancel/close wakes
   the worker. One acceptance suffices for that job. Non-GUI callers never wait for
   UI input. Do not automatically simplify geometry or alter source distances.
3. Preserve source and grouping rules. Enclose every qualified midpoint in rectangle
   fallbacks, pad curved outline ends, tolerate unused malformed fallback boxes,
   and validate serialized rings within 250,000 active-edge inspections. Use a
   disclosed simpler fallback when detailed validation would exceed that budget.
4. Verify successful benchmark cases remain quiet and the known over-limit linked
   case warns. Compare numerical/group fields with explicit visual exclusions.
   Exercise Continue/Cancel/close and geometry regressions, and check input-order
   invariance across straight, curved, multipart, reversed, dateline/polar examples.
5. Refresh both Windows previews and frozen smoke with exact source/file hashes.
   Consolidate real-data, viewer and native interaction checks into one deferred
   packet. macOS GitHub Actions remains a future task; no piecemeal manual testing
   is requested before handoff preparation is complete.

This replaces the suggestion list with a verified, actionable backlog. The separate
[FOLLOWUP_RUNBOOK.md](FOLLOWUP_RUNBOOK.md) contains tasks requiring owner judgment,
additional data, another platform, or interactive external applications. Those are
follow-ups after the automated work, not prerequisites to starting it.

## Implementation operating decisions

Implementation is now authorized. Use these decisions to proceed without routine
approval requests; the verification table below records the earlier planning baseline:

- Preserve GRS80 source lengths, US Survey Miles, defaults/clamps, minimum continuous
  coverage on both paths, multipart isolation, and groups where every cross-pair
  qualifies and each pipeline contributes at most one segment. Retain
  `qualified_segment_coverage_v2`; [README](README.md#minimum-parallel-length) and
  [bundling](src/pipeline_calculator/core/bundling.py) establish the current contract.
- Keep clearing results when starting an analysis. After cancellation, return to
  file selection with the path/parameters available to retry. A previous-results
  retention feature is unnecessary for these improvements.
- Share lifecycle/presentation helpers across modular and legacy GUIs; preserve
  synchronous APIs and wrappers using optional arguments. Avoid a general rewrite.
- Show stages, counts where known, and elapsed time. Do not add overall percentage,
  time-remaining prediction, or change the default segment length.
- Synthetic fixtures and the existing tracked KMZ suffice to implement validation
  tooling. Treat that KMZ as an unreviewed regression input, not ground truth.
  Do not redistribute its geometry in new reports or require private files to start.
- Fix reproduced regressions against established behavior autonomously: capture a
  minimal reproduction, add a focused test, fix, and verify. For an unresolved
  business/model rule, preserve existing semantics, add the evidence to runbook R2,
  and continue independent tasks. Routine technical choices do not need owner input.
- Profile before optimizing. A documented decision to keep the current code
  completes profiling if no useful result-preserving optimization is demonstrated.
- Use isolated build outputs: current platform scripts delete `build/` and `dist/`.
  Verify resolved paths before cleanup; preserve the user's current artifacts.
  Do not change desktop settings, associations or signing credentials, or launch
  external GUI applications during unattended work.
- Local changes, tests, documentation and preview builds belong to future automated
  implementation. Publication, release tags, distribution and credentials are R6.
  Do not trigger the tag-based release workflow just to obtain test artifacts.
- If an external dependency blocks one check, prepare its evidence and record the
  specific blocker once in the runbook; finish independent work. Do not transfer
  ordinary failing tests or fixable regressions to the user to claim completion.

## Implementation decisions and bounded deviations

- [Final verification](docs/validation/automated-improvements.md) is the completion
  record; the original acceptance steps below remain as traceable design criteria.
- A6 builds the exact working source with isolated `-OutputRoot` artifact directories
  instead of copying the checkout. Both previews retain source and artifact hashes;
  the repository's existing build/dist contents were not cleaned.
- A3 optimized checkpoint placement after measured overhead, retaining all numerical
  outputs. Broader algorithm/data-structure optimizations were not justified.
- A4 reports section sums, unique covered intervals, projection distortion and
  combined savings gaps separately. Arbitrary branched group optima are not claimed;
  the combined savings gap cannot be described as isolated heuristic loss.
- Initial A7 checked topology only up to 512 vertices. A11 supersedes that limit
  with a bounded sweep for all ring sizes and disclosed fallback on excessive work.
  Invalid preferred shapes in the tested sharp
  bend/loop cases use explicitly approximate existing rectangle fallbacks. No new
  sensor swath or operational accuracy rule was introduced.
- Optional stress/near-limit workloads were not used to set production budgets;
  resource-limit regression tests pass, and the measured medium workloads are in
  [the comparison report](docs/validation/comparison.md). R4 owns operational budgets.

## Direct verification performed

These are observations from this review, not claims that the planned changes exist.
During the planning review, no source code, requirements, installed dependencies or
build artifacts were changed. The implementation ledger now records subsequent work.
Launcher probes used mocks and did not start external applications.

| ID | Evidence | Conclusion and limit |
| --- | --- | --- |
| V1 | `python -m pytest -q`: **106 passed in 7.31 s**, Windows, system Python 3.13.3, including withdrawn Windows pagination widgets. | Baseline passes; no packaged visual or operational accuracy certification. |
| V2 | `.venv/Scripts/python.exe -m pytest -q`: **No module named pytest**. Venv: Python 3.11.9, NumPy 1.24.3, SciPy 1.15.3, pyproj 3.7.2. System: NumPy 2.2.6, SciPy 1.16.0, pyproj 3.7.1, pytest 8.4.1. | Passing system tests do not verify the repo's build runtime. Establish its baseline first. |
| V3 | Read `AnalysisJob`, both `process_file` flows and downstream loops/exception boundaries. | No cancellation context; legacy duplicates threading. Broad catches would convert a normal cancellation exception into an error unless explicitly bypassed. |
| V4 | Parser callback on tracked KMZ emitted no events. Full analysis emitted `[0.5, 0.5833333333333334, 0.6666666666666666, 1.0]`. Controller does not pass a callback. | Parser accepts an unused callback; search/group work is not reported; progress never reaches the GUI. |
| V5 | `git ls-files .danny`: tracked `.danny/Centerlines.kmz`, 8,886 bytes, SHA-256 `2f240c0b680c0afa211e6f0d6d37c1a8ce3bd7386dc00ecbf0166f68a02e7f4f`. Parsed 3 pipelines, 4 paths, 642 vertices, 0 placemarks, 1 document, only `selected_primary_kml` diagnostic. | Corrects the earlier no-project-files assumption. Provenance, representativeness and independent expectations remain unknown. |
| V6 | Default analysis of that file: complete, 100,035.7094129826 original meters, 0 savings, 0 sections. One `cProfile` run took about 0.367 s including analyzer construction and recorded 20,005 neighbor queries. | Available smoke input; no dense-bundle/corridor coverage. Values are app output, not ground truth; one profiled timing is not a performance budget. |
| V7 | Mocked `open_path` raising `OSError`: wrapper still returns saved path. Mocked macOS launcher exit 7: `open_path` returns `None` with `check=False`. | Launch failure is demonstrably swallowed. |
| V8 | Passed a polygon with `float('nan')` to KML exporter: output contained `nan`. Source selects rectangle fallbacks without reporting geometry kind. | Confirmed validation/disclosure gaps. Does not prove normal analysis generates NaN or all curved shapes fail topology checks. |
| V9 | Read platform scripts/workflow. Windows build deletes `build/` and `dist/`; CI uses Python 3.11 and publishes on tags. Windows test helper can print `OK` after failed native commands; macOS helper checks bare `pytest`. | Isolate builds and make test exit status truthful. No artifact was built during this review. |

Sources:
[controller](src/pipeline_calculator/gui/controllers/analysis_controller.py),
[modular GUI](src/pipeline_calculator/gui/main_window.py),
[legacy GUI](src/pipeline_calculator_v3.py),
[parser](src/pipeline_calculator/parsers/kml_kmz.py),
[analyzer](src/pipeline_calculator/core/analyzer.py),
[segmentation](src/pipeline_calculator/core/segmentation.py),
[overlap](src/pipeline_calculator/core/overlap.py),
[bundling](src/pipeline_calculator/core/bundling.py),
[KML exporter](src/pipeline_calculator/export/corridor_kml.py),
[launcher](src/pipeline_calculator/gui/actions/open_kml_action.py),
[tracked KMZ](.danny/Centerlines.kmz),
[regressions](tests/test_calculation_regressions.py),
[follow-up tests](tests/test_followup_audit.py),
[Windows tests](scripts/windows/run_tests.ps1),
[macOS tests](scripts/macos/run_tests.sh),
[Windows CI](scripts/ci/windows_build.ps1),
[workflow](.github/workflows/build.yaml).

The earlier [FOLLOWUP_AUDIT.md](FOLLOWUP_AUDIT.md) remains a historical record.
Its no-data assumption is superseded by V5. Existing numerical tests establish
individual cases, not a universal sampling-error bound or a general geometry oracle.

## Disposition of every original suggestion

A1-A8 preserve original task numbers; A0/A9 provide setup and handoff. Current task
status is in the implementation ledger above. A verified validation *gap* justifies building tools, not predetermining
what their results will show.

| Original task | Autonomous implementation | Follow-up |
| --- | --- | --- |
| 1. Cancellation | A1 shared lifecycle, checks, cleanup, race tests | R3 packaged interactive behavior |
| 2. Progress | A2 stages/counts, bounded transport, overhead checks | R4 operational responsiveness targets |
| 3. Performance | A3 baseline, instrumentation, gated optimization | R1/R4 representative projects and budgets |
| 4. Interval reference | A4 analytic oracle, bounded model, error study | R2 tolerance and any semantics changes |
| 5. Project corpus | A5 manifest/runner, synthetic cases, tracked sample | R1 provenance, approved files, reviewed expectations |
| 6. Packaged GUI | A6 test scripts, isolated Windows builds, automated checks | R3 native visuals/deferred macOS GitHub Actions; R6 distribution |
| 7. Corridors | A7 gallery, checks, export guards/disclosure | R5 viewer checks deferred; rough visualization purpose confirmed in R2 |
| 8. Launch errors | A8 explicit outcome and recoverable saved file | R3/R5 real associations and external rendering |

## Order, dependencies and artifacts

1. **A0 -> A5 -> A3.1:** intended runtime, reusable fixture/manifest infrastructure,
   then baseline before production execution changes. A5 does not wait for R1.
2. **A1 -> A2:** shared cancellation lifecycle, then progress presentation.
3. **A8:** independent launch recovery improvement.
4. **A4 -> A7:** reference evidence and gallery. Share fixture specifications, never
   production qualification/geometry routines for generating expected answers.
5. **A3.2:** measure the changed app, instrumentation overhead and any justified
   optimization; keep each optimization independently reviewable/revertible.
6. **A6 -> A9:** package stable behavior, verify available checks, prepare one handoff.

A1/A4/A8 do not depend on additional project data or business decisions. Source-level
A6 checks can proceed on Windows even though macOS execution is unavailable here.
Implementation boundaries above are suggestions, not instructions to create commits,
PRs or releases without a corresponding future request.

Implementation locations (see the ledger for verification status):

| Location | Purpose |
| --- | --- |
| `src/pipeline_calculator/core/execution.py` | Tk-independent execution context |
| `src/pipeline_calculator/gui/controllers/` | Shared ownership/polling/presentation |
| `scripts/validation/` | Fixture generator, corpus runner, gallery and error reports |
| `scripts/benchmarks/` | Isolated workload runner and profiling |
| `tests/reference/` | Independent interval/group reference and tests |
| `tests/fixtures/` | Small synthetic manifests and analytical specifications |
| `docs/validation/` | Shareable schemas, methodology, reports and completion ledger |
| Ignored local evidence directory | Raw profiles and generated/private KML/JSON/XLSX |

## A0 â€” Establish the intended runtime and baseline

**Basis:** V1-V2/V9. **Dependency:** none.

1. Recheck HEAD, local changes and repository instructions before implementation.
   Preserve unrelated edits and existing build outputs. Record Python, OS, Tcl/Tk,
   dependencies, revision and dirty status.
2. Run the existing system-runtime suite to compare with V1. Use the healthy existing
   Python 3.11 venv, installing `requirements-dev.txt` there, or create a separate
   3.11 environment if needed. Resolve against existing requirements/NumPy 1.24.3;
   do not silently change the dependency contract to match system Python 3.13.
3. Run the full suite using the explicit 3.11 executable. Investigate setup failures
   before production edits. Installing test dependencies is routine autonomous
   setup in implementation; credential/network restrictions, if any, become R6.
4. Establish output directories and a ledger: task, status, files, commands, result,
   evidence path, deviation and runbook reference. Preserve pre-change results.

**Done when:** intended-runtime baseline is recorded and user work/artifacts are
preserved. A diagnosed unavailable runtime is recorded as a real limitation, never
replaced by a misleading claim that system tests verified the packaged dependency set.

## A1 â€” Cooperative cancellation and reliable job cleanup

**Basis:** V3. **Dependencies:** A0 and baseline A3.1.

Touchpoints: controller, both GUIs, analyzer, parser, segmentation, overlap, bundling,
[coordinate paths](src/pipeline_calculator/core/coordinates.py), and the
[effective-length compatibility helper](src/pipeline_calculator/core/effective_length.py).

1. Add an optional execution context with cancellation event/progress sink. Define
   `AnalysisCancelled` and explicitly re-raise it before broad exception wrapping,
   parser diagnostics and `overlap_analysis_failed`. A cancelled job must never
   become incomplete-success, failed analysis, or successful zero savings.
2. Give jobs immutable IDs and file/parameter snapshots. Define
   `pending -> running -> completed | failed` and
   `pending/running -> cancellation_requested -> cancelled`. Publish terminal
   state/result/error consistently under a lock before setting `done`. Reject
   starting twice; make cancel idempotent; terminal states cannot change.
3. Resolve races deterministically: cancellation recorded before success publication
   wins and discards the result; a request after terminal publication is a no-op.
   Already-published failures remain failures. Use barriers to test this contract.
4. Check cancellation before/after document reads/XML parsing; within document,
   feature/coordinate and source-length loops; during segmentation/spatial arrays,
   neighbor/candidate loops, section components, group merges and corridor loops.
   Check inner loops so a single dense document/pipeline remains cancellable.
5. Start with checks every 256 Python work items and around expensive calls, then
   measure/tune. KDTree construction/query, large sorts, XML/geodesic/NumPy/SciPy and
   filesystem calls can only acknowledge after the individual call returns. State
   that limit; do not forcibly terminate a Python thread or promise instant stopping.
6. Replace the legacy GUI's duplicate worker/event machinery with the shared
   controller/helper. Preserve synchronous analyzer/wrapper behavior via optional
   context arguments; avoid a mutable analyzer shared across jobs.
7. Add Cancel and `Cancelling...`; disable duplicate requests and keep busy until
   acknowledgement. Then clear the overlay and return to selection without result
   or error dialogs. Keep the selected file/parameters available for retry.
8. Track active job, scheduled poll ID and closing state. Centralize success,
   failure, cancellation, startup-error and close cleanup. Route window close/Exit
   through it: request cancel, detach/cancel callbacks and destroy safely, without
   an unbounded Tk-thread join. Preserve daemon-worker process shutdown behavior.
9. Check job identity/open-window state before consuming events. Normal operation
   still allows only one active analysis; synthetic late callbacks from detached
   jobs must not overwrite results or busy state.

**Tests:** before-start, repeated and per-stage cancellation; a single long feature;
near-completion/error races; startup failure; close with pending poll; old callback;
new successful job after cancel. Use events/barriers rather than sleep-based races.
Cover both GUIs and no-context APIs; assert no partial result, false diagnostic,
worker widget call or stuck thread in controlled tests.

**Done when:** lifecycle invariants pass and numerical outputs remain unchanged.
Measure request-to-acknowledgement latency by stage. Initial engineering target:
<250 ms in benchmarked Python loops, excluding blocking calls. Report actual
outliers and investigate; this is a tuning target, not an operational guarantee.

## A2 â€” Useful, bounded progress reporting

**Basis:** V4. **Dependencies:** A1; overhead validation in A3.2.

1. Publish immutable snapshots: job ID, sequence, stage, completed count, optional
   total and monotonic elapsed time. Use a locked latest snapshot instead of an
   unbounded queue; terminal job state remains separately authoritative.
2. Map stages to actual call order: reading documents, source lengths, segmentation,
   spatial index, neighbor search, qualification, corridor construction, savings,
   finalization. Count documents, paths/edges, searched segments, sections or merge
   candidates in their own units. Unknown linked-document totals remain unknown.
3. Show stage/elapsed time during indivisible library calls. Do not fabricate counts
   or combine stage fractions into a global percent. Finalization is complete only
   when the result is published, including valid incomplete-analysis outcomes.
4. Publish stage changes immediately; coalesce within-stage progress to <=10 updates
   per second; keep cancellation checks independent. Poll on Tk's existing roughly
   100 ms cadence through the shared helper. Cancel/terminal status overrides text;
   reject wrong-job and older-sequence snapshots.
5. Preserve existing float callback APIs through a compatibility adapter. The GUI
   consumes structured progress; do not send objects to numeric-only callbacks.

**Tests:** linked parsing emits work; search advances within one dense pipeline;
changing/unknown totals, empty/point-only input, incomplete results, failure,
cancellation and stale messages. A fake clock verifies throttling; a fast producer
and slow consumer verify constant transport storage. No worker touches widgets.

**Done when:** honest stages/counts advance, no false global 100%, and numerical
outputs match. A3 compares progress enabled/disabled; target median overhead <=5%
on medium workloads lasting >=1 s. Enlarge too-fast workloads to avoid timer noise,
tune if exceeded, and report actual overhead rather than conceal it.

## A3 â€” Reproducible performance evidence and gated optimization

**Basis:** V5-V6 and inspected processing limits. No general bottleneck is proven.
**Dependencies:** A5 fixtures; A3.1 before production edits, A3.2 after improvements.

### A3.1 â€” Capture the baseline

1. Implement `scripts/benchmarks/run.py` with explicit suite/input, seed, parameters,
   repeat count, timeout and output directory. Run workloads in subprocesses for
   memory isolation and bounded termination. Stopping a benchmark subprocess is
   separate from cancelling a GUI worker thread.
2. Provide smoke, medium and opt-in stress suites. Start near 1,000/20,000/100,000
   segments, adjusting fixture sizes to existing limits. Include sparse long lines,
   dense compatible bundles, incompatible chains, many short paths, multipart
   curves and linked documents. Record actual counts. The tracked KMZ is a separate
   unreviewed case; it cannot substitute for dense/corridor workloads.
3. Exercise over-limit behavior with reduced budgets in unit tests; actual near-limit
   work belongs in the bounded stress runner. Retain document/byte/archive/segment/
   candidate limits. Stop scaling at resource/time limits and record the outcome.
4. Record JSON and Markdown: revision/dirty patch identity, input hash/seed,
   parameters, CPU/OS, Python/packages, completion/diagnostics, original/adjusted
   totals, sections, segments, candidate inspections and unique accepted matches.
5. Separate timing from profiling: one warm-up plus five unprofiled runs, median
   and range, then a separate `cProfile` run. Record stage durations where observable;
   label combined baseline stages rather than change production first. Benchmark
   instrumentation for unavailable counters must be separate from timing runs.
6. Measure peak resident memory with an OS-supported collector; label collector,
   units and sampled-vs-peak semantics. Python allocation measurements can supplement
   RSS but cannot stand in for native NumPy/SciPy memory. Unsupported fields are
   unavailable, never zero. Keep instrumentation out of ordinary product output.

### A3.2 â€” Compare and optimize only when justified

1. Repeat identical fixtures/runtime/parameters after A1/A2/A7. Measure context and
   progress enabled/disabled, plus per-stage cancellation latency. Compare totals,
   diagnostics, section ordering and geometry; isolate intentional export metadata
   differences from numerical equivalence.
2. Rank measured costs before selecting work. Investigate query overhead, repeated
   geodesic/conversion calls, graph retention and corridor construction only when
   the profiles support it. V6's single sparse run cannot establish dense behavior.
3. Prototype one improvement at a time. Keep only reproducible improvement exceeding
   run variability, with reference/regression equivalence, deterministic ordering,
   unchanged limits, maintained cancellation responsiveness and no unexplained
   regression on another workload. Prefer simpler code for indistinguishable speed.
4. Revert unsuccessful prototypes without disturbing user changes. Record
   `complete - no optimization justified` when appropriate. Do not need an owner
   decision about data structures or a production target to close this task.

**Done when:** documented commands reproduce baseline/comparisons, every kept
optimization has before/after timing and memory plus output checks, and oversized
work fails clearly. No normal unit test asserts wall-clock speed. Production
budgets and representative-project performance acceptance remain R1/R4.

## A4 â€” Independent covered-interval reference and error study

**Basis:** full-segment sampling and conservative grouping verified in source/tests.
**Dependency:** A5 fixture specifications; comparison after execution changes stabilize.

1. Write the reference specification first. Use local Cartesian polylines, cumulative
   distance along each original part, and separately expose source lengths, pairwise
   coverage, continuous-minimum decisions and group savings. Start with common-axis
   lines where interval endpoints/lengths have hand-derived answers.
2. Implement intersection/union/subtraction and unique covered length independently
   in `tests/reference/`. No imports of production segmentation, qualification,
   savings or corridor code in the oracle. A separate adapter may call production.
   Author expected intervals from fixture definitions, never current app outputs.
3. Specify finite-edge matching for the bounded planar model: undirected tangent
   angle, positive longitudinal overlap, clipping to permitted transverse distance
   for both edges, endpoint inclusion and projection conventions. Retain edge-pair
   correspondence when unioning by path; enforce a continuous minimum on both
   participating paths. Do not merge disjoint parts or count repeated matches twice.
4. Begin with analytical cases: two identical 300 m lines -> 300 m savings; two
   300 m lines shifted 50 m -> 250 m common coverage at minimum 200 m; the same
   at minimum 251 m -> zero; end-to-end contact -> zero; three compatible 300 m
   lines -> 600 m savings; offsets 0/10/20 m with range 15 m -> 300 m savings;
   separate 150 m parts cannot meet minimum 200 m. Add unequal lengths, partial
   endpoints, duplicates and reversal.
5. For common-axis multi-line cases (at most six pipelines), split at interval
   boundaries and exhaustively enumerate disjoint compatible groups, including
   singleton passes. Require every cross-pair to qualify. Report the optimal
   reference and production heuristic gap separately from geometric sampling error;
   a better optimum is not evidence that production breaks its conservative contract.
6. Extend pairwise coverage to bounded piecewise-linear curves/multipart paths in
   that declared model. Keep exhaustive multi-line optimization limited to supported
   common-axis cases. Ambiguous branch/turn correspondence becomes a documented model
   limitation for R2 rather than an invented operational rule.
7. Map small synthetic fixtures geographically using a documented local projection
   and existing pyproj. Record origin, ellipsoid, transformation and extent; initially
   restrict fixtures to 10 km from the origin. Compare planar edge/pair distances
   with GRS80 geodesic distances and report maximum distortion; shrink/reject a case
   exceeding `0.001 m + 1e-5 * distance`. This is a reference-model engineering
   check, not a production accuracy tolerance. Dateline/polar cases use their own
   local origin; never calculate topology in raw longitude/latitude degrees.
8. Sweep segment lengths 1/2/5/10/25 m, reversal/bearings, endpoint offsets, range/
   angle/minimum boundaries, curves, multipart, dateline and near-pole fixtures.
   Separate projection distortion, pair sampling error and group heuristic gap.
   Report signed/absolute meters; relative error is N/A when the reference is zero.
9. Validate the oracle against hand-derived cases and test-local mutants that drop
   partial intervals, double-count or join below-minimum parts. Demonstrate each
   mutant is rejected without editing production files. Add a dependency/import
   check to prevent accidental reuse of production qualification in the oracle.

**Done when:** analytic checks pass with documented numerical epsilon (start at
1e-7 m for modest planar cases and investigate adjustments), independence is checked,
and every requested category has either comparison evidence or a specific model
limit. Deliver `docs/validation/reference-model.md` and an error report. This task
measures production results, not forcibly makes them equal to a different model.
No operational endorsement or production semantics/default change is needed to
complete it; hand those evidence-based decisions to R2.

## A5 â€” Corpus infrastructure and the available tracked input

**Basis:** V5 corrects the original no-data assumption. **Dependency:** A0.

1. Define a versioned manifest: fixture ID/purpose, relative/local path or private-root
   key, SHA-256, parameters, geometry model, expected source/adjusted values and
   diagnostics, tolerance per asserted field, expectation basis, provenance/reviewer
   if known, and status `analytic`, `unreviewed` or `reviewed`. Unknowns stay unknown.
2. Generate deterministic KML/KMZ fixtures for A3/A4/A7 and parser coverage: linked
   documents, multipart, tracks, malformed/empty and unsupported features. Extend
   the pattern of [parser tests](tests/test_kmz_parsing.py); the
   [existing generator](tests/tools/generate_test_kml.py) only creates two simple lines.
3. Characterize `.danny/Centerlines.kmz` by hash/parameters, labeling V5/V6 as observed
   baseline rather than correct mileage. A hash change invalidates the baseline.
   Zero sections cannot validate savings/corridor behavior. Leave the file and its
   Git history intact; provenance/distribution decisions are R1.
4. Implement `scripts/validation/run_corpus.py`: validate manifests/hashes, analyze,
   compare only stated expectations, output JSON/Markdown. Distinguish pass/fail,
   unreviewed, missing private input and not-run. Default suite uses synthetic plus
   available local inputs; strict reviewed mode fails for missing required fixtures.
5. Support a private data root outside Git without requiring one now. Keep sensitive
   geometry, names and absolute personal paths out of public reports; use fixture
   IDs/hashes and ignored raw output. Prepare a manifest template for R1.
6. Export small synthetic results through actual JSON/XLSX code and reopen outputs
   to check units, status, diagnostics and values. Preserve intentional formulas
   and literal source text in [existing export tests](tests/test_export_workbook.py).

**Done when:** deterministic fixtures and hash checking work; bad manifests fail
clearly; optional absent private input neither silently passes nor blocks synthetic
work. Every assertion identifies its expectation basis. R1 receives a functioning
harness/template instead of being asked to design it before automation can proceed.

## A6 â€” Automated packaging checks and GUI regression preparation

**Basis:** V1-V2/V9. **Dependencies:** A0 and finalized A1/A2/A7/A8 behavior.

1. Make Windows test/build helpers check and propagate every native Python,
   pytest/PyInstaller and final child-build exit code. `try/catch` and
   `$ErrorActionPreference='Stop'` alone do not enforce this. On macOS use the venv's
   `python -m pytest`; fail explicitly if absent instead of a PATH-selected runner
   or a final `OK` after skipped tests. Preserve platform/version defaults.
2. Add focused subprocess checks with fake failing/successful tools for exit behavior.
   Do not install global packages or create a release to test these wrappers.
3. Extend shared/headless GUI coverage: import, corrected parameters, reanalysis,
   cancel/retry, startup failure, close, incomplete/empty, export and launch recovery.
   Extend withdrawn Windows widget tests for controls/progress/pagination. Full
   display/focus review belongs in R3.
4. Prepare an isolated source snapshot/build directory with exact intended changes
   and required Git/version context. Include uncommitted changes if applicable;
   do not build an older HEAD and claim it tests the edits. Record revision, patch
   digest, package inventory and version. Verify resolved cleanup targets first.
5. Build modular and legacy Windows previews with
   [build_exe.ps1](scripts/windows/build_exe.ps1). Preserve each artifact outside the
   next build's cleaned directories. Verify process status, files, embedded metadata
   where inspectable, resources and [version tests](tests/test_versioning_packaging.py).
   Label dirty artifacts and attach their patch identity.
6. If needed, add a bounded packaged smoke mode for frozen imports, embedded
   metadata/resources, a withdrawn widget tree and a tiny synthetic analysis. Keep
   it outside ordinary UI flows. Compare with source results; impose a timeout on
   that test process. Open no dialogs/external apps and test both entrypoint choices.
7. Review macOS integration locally; that is not native execution evidence. With the
   current Windows-only environment, prepare exact commands/checklist for R3. If
   an authorized macOS environment becomes available during implementation, run its
   automated checks there and remove completed items from follow-up.

**Done when:** deliberate tool failures return nonzero, Windows previews/smoke
are tied to changed source, available source/widget checks pass, and unavailable
macOS/interactive checks are individually not-run. Existing artifacts stay intact;
no tag or release is created. References:
[Windows build](scripts/windows/build_exe.ps1),
[macOS build](scripts/macos/build_app.sh),
[CI workflow](.github/workflows/build.yaml).

## A7 â€” Corridor gallery, export guards and approximation disclosure

**Basis:** V8 and inspected geometry/test coverage. **Dependency:** A5; A4's local
model may support validation, but expected geometry cannot come from production.

1. Generate a gallery: straight, gradual curve, right-angle turn, hairpin, loop,
   short qualifying section, disconnected paths, reversal, dateline and near-pole.
   Save source paths, qualified-section identity, polygon, center, KML, fixture ID
   and parameters. Keep private-project examples for R1/R5.
2. Check both in-memory geometry and coordinates re-read from KML (currently seven
   decimal places): finite/range-valid coordinates, distinct vertices, closure,
   nonzero area and center locality. Test NaN/infinity, malformed preferred polygons,
   degenerate rings and bad fallbacks directly at the exporter boundary.
3. Transform bounded gallery cases to the declared metric model. Independently check
   nonadjacent self-intersections, point inclusion and distances to qualified source
   paths/endpoints. Validate the checker with rectangles, bow-ties, collinear and
   repeated-vertex fixtures. Bound quadratic intersection checks to small test
   fixtures; do not add an unbounded quadratic production topology pass.
4. Distinguish sampled represented extent, full original path and reference-qualified
   intervals. Report endpoint shortfalls/excess area; do not require the whole
   pipeline inside a partial-section polygon. Preserve separate path identities.
5. Implement the confirmed export guard: validate chosen polygon, fallback and center;
   reject unusable geometry with an actionable error. A malformed preferred shape
   may use a separately valid existing fallback only with an explicit approximation
   reason. Never serialize NaN/infinity or invent a zero-longitude/latitude center
   when supplied geometry/center data is invalid.
6. Add geometry kind/reason to section metadata and KML description; expose concise
   approximation text at corridor actions. Distinguish sampled curved corridor,
   oriented rectangle and bbox fallback. Keep width/cap heuristics unless a minimal
   reproduction proves a defect against existing behavior. A valid shape is not
   validated sensor coverage or an operational survey boundary.
7. For a reproduced topology defect, add a minimal test and either make a bounded
   repair consistent with the existing contract or report the corridor unavailable/
   approximate. Invalid polygons must not silently become precise-looking shapes.
   Preserve valid numerical mileage when export-only validation fails. Do not add
   speculative global polygon unions, new swath rules or runtime topology packages.
8. Report numerical validity, model/approximation limits and `Google Earth review
   pending` separately. Valid local coordinates do not certify seam/render behavior.

**Done when:** invalid injected coordinates cannot produce invalid KML, existing valid
cases export, fallback type/reason is visible, and mileage remains unchanged. Every
gallery case has reproducible checks and an outcome; serialization is tested as well
as internal geometry. Unknown production topology defects remain investigations,
not predetermined failures. R5 owns external rendering; R2 owns new geometry purposes.
Sources: [geometry construction](src/pipeline_calculator/core/overlap.py),
[KML export](src/pipeline_calculator/export/corridor_kml.py),
[export tests](tests/test_overlap_kml.py).

## A8 â€” Recoverable corridor-launch failures

**Basis:** V7 and swallowed row callbacks in
[overlap_tab.py](src/pipeline_calculator/gui/tabs/overlap_tab.py). **Dependency:** A0;
align final geometry-error messages with A7.

1. Separate generation, file write and launching. Add a shared outcome with saved
   path, `requested`/`failed` state and useful error. Preserve the current string
   wrapper if needed: return path on accepted launch, raise a typed error retaining
   the path on failure. Both GUIs use the outcome-aware helper/recovery handler.
2. Catch Windows `os.startfile` errors. Check macOS/Linux `open`/`xdg-open` exit codes,
   bounded stderr and timeout (initially 10 s). Run blocking launch requests outside
   Tk; reuse callback/close discipline without attaching them to a completed analysis.
3. Say `Opening requested` only after OS acceptance. On failure explain that KML was
   saved but opening failed; show path/reason and Copy Path, Save As, Retry, Close.
   Retry uses the existing file. Write failure has no fictional saved path; Save As
   failure retains the original recoverable file and reports the new error.
4. Close file handles before launching. Keep temp KML after launch/failure, dialog
   close and app exit. Use OS temp storage; document OS cleanup and Save As for a
   durable copy. Avoid new age-based deletion until ownership/read timing is assured.
5. Route legacy `view_overlap_kml`, modular actions and table dispatch through shared
   recovery. Narrow/remove blanket `except: pass`; handle stale/deleted row IDs
   separately without hiding callback errors or showing duplicate dialogs.

**Tests:** serialization/write failure, missing launcher, nonzero exit, timeout,
Windows association error, accepted request, copy/retry, Save As success/failure,
row callback and close with launch pending. Mock platforms; do not launch Google
Earth/change associations. Cover spaces/Unicode paths, UTF-8, file reuse and legacy.

**Done when:** write-success/open-failure always offers a usable path, errors remain
visible, and no accepted request claims confirmed rendering. A hanging launcher
cannot freeze Tk. Sources:
[launch action](src/pipeline_calculator/gui/actions/open_kml_action.py),
[modular caller](src/pipeline_calculator/gui/main_window.py),
[legacy caller](src/pipeline_calculator_v3.py).

## A9 â€” Automated completion and follow-up handoff

**Dependency:** automated outcomes recorded, including no-change profiling decisions
and specifically unavailable platform checks.

1. Use focused tests per change. After integration run the full Python 3.11 suite,
   `python -m compileall -q src`, corpus/reference/gallery checks and applicable
   preview smoke. Run `git diff --check`. System-runtime comparison can supplement
   but not replace the intended build runtime.
2. Update README for cancellation, real progress, launch recovery and approximation.
   Reconcile unsupported progress/performance claims with measured evidence and
   retain sampled/heuristic calculation limitations.
3. Write `docs/validation/automated-improvements.md`: source/patch identity, task
   outcomes, tests/skips, benchmark/error/gallery commands and results, cancellation
   latency, overhead, artifacts and remaining limitations. Keep raw private data local.
4. Populate the runbook with exact prepared files, outstanding owner questions and
   unavailable checks. Remove already-resolved questions; hand off once rather than
   interrupting for each optional project, setting or technical decision.
5. Mark tasks `complete`, `complete - no optimization justified`, or a specifically
   identified external check deferred to Rn. In-scope failing checks require fixes;
   they are not human tasks merely because they failed. When a business/model rule
   is required, preserve current semantics and document the concrete choice in R2.

Completion means available improvements/checks and their evidence are finished.
It does not certify reviewed real-project accuracy, macOS visuals, Google Earth
rendering, or permission to distribute.

## Commands and proof reproduction

Existing commands used in this review:

```powershell
python -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q  # Missing pytest during planning.
git rev-parse HEAD
git ls-files .danny
```

Implemented validation interfaces:

```powershell
.\.venv\Scripts\python.exe scripts/validation/run_corpus.py --suite local --output build/validation/corpus
.\.venv\Scripts\python.exe scripts/benchmarks/run.py --suite smoke --repeat 5 --output build/validation/benchmarks
.\.venv\Scripts\python.exe scripts/validation/compare_reference.py --output build/validation/reference
.\.venv\Scripts\python.exe scripts/validation/build_gallery.py --output build/validation/gallery
```

Use a prepared implementation workspace; retain evidence outside directories that
build scripts clean. Keep commands/flags consistent as tools land. Preserve the
pre-change baseline independently of after-results.

Reproduce V7/V8 without opening external applications using the system Python from
V1. In PowerShell, pipe a single-quoted here-string containing this code to `python -`:

```python
import sys
import subprocess
from unittest.mock import patch
sys.path.insert(0, 'src')
from pipeline_calculator.gui.actions import open_kml_action as action
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml

with patch.object(action, 'write_corridor_kml_tempfile', return_value='saved.kml'), \
     patch.object(action, 'open_path', side_effect=OSError('no handler')):
    print(action.open_overlap_corridor({}, 1))  # Current: saved.kml; error lost.
with patch.object(action.sys, 'platform', 'darwin'), \
     patch.object(action.subprocess, 'run', return_value=subprocess.CompletedProcess(['open'], 7)):
    print(action.open_path('saved.kml'))  # Current: None; exit 7 ignored.
section = dict(pipeline_1='A', pipeline_2='B', bundled_length_miles=1,
               average_separation=2,
               corridor_polygon=[(float('nan'), 0), (1, 0), (1, 1), (float('nan'), 0)])
print('nan' in build_overlap_corridor_kml(section, 1))  # Current: True.
```

For V5/V6, verify the hash first, then call
`extract_features_from_file_with_diagnostics('.danny/Centerlines.kmz')` and
`PipelineAnalyzer().analyze_complete('.danny/Centerlines.kmz', progress_callback=events.append)`.
Record environment and parameters with each result. These probes establish current
behavior; future acceptance requires the focused tests and reports above.
