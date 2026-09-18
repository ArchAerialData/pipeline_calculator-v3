# Corridor buffer implementation evidence

Updated September 18, 2026. **Implemented, verified and published to local Windows `dist`.** Numerical comparison, fixture auditing, geometry, performance, source/UI and frozen Windows checks have passed. macOS distribution requires separate platform verification. The governing specification is the [implementation runbook](../../CORRIDOR_GEOMETRY_IMPLEMENTATION_PLAN.md).

## Implemented behavior

Corridors follow complete qualifying paths with **5 m padding**, round joins and round caps. Bends, disconnected components and holes survive map previews and exports. The builder slices original GRS80 geodesic paths at qualified sample chainages; it does not connect separate paths or include unqualified trailing samples. Mileage, qualification, shared-border allocation and savings retain their existing calculation contracts.

Numerical sections and savings finish before optional map construction. Projection, buffer, normalization, clipping or resource failures omit the affected map and produce a warning without erasing mileage or savings. Cancellation aborts the job. Canonical geometry has schema version 1, policy `qualified_path_buffer_v1`, and explicit `ready`/`omitted` status. Empty, omitted, invalid or unknown canonical geometry cannot revive a legacy rectangle. Legacy dictionaries without canonical fields retain their previous validated behavior. See [builder](../../src/pipeline_calculator/core/corridor_buffer.py), [orchestration](../../src/pipeline_calculator/core/overlap.py), and [canonical decisions](../../src/pipeline_calculator/core/corridor_geometry.py).

Ordinary previews serialize polygons only, with every component and hole at full floating-point precision. State maps use authoritative clipped polygons. Excel appends a Corridor Map column and reuses Analysis Details for the padding policy. Unsupported legacy center/width fields stay blank; geometry text over 30,000 characters is omitted whole, never truncated. JSON validates finite native values. The UI keeps existing controls, explains that padding does not affect mileage, and disables unavailable map actions. See [consumer tests](../../tests/test_corridor_consumers.py) and [workbook tests](../../tests/test_export_workbook.py).

## Geometry and bounded state handling

Construction uses bounded local ellipsoidal radial charts, adaptive source/output densification, dateline splitting, complete polygon unions, and a two-sided neighborhood check against finer generating support. The numerical display target is **0.05 m relative to that construction model**, not surveyed ownership accuracy. The JPG boundary application is not a runtime dependency.

State clipping retains original boundary edge endpoints in its local mask. Shortening those edges to a query rectangle first had changed slopes by floating-point roundoff. A seeded 120-case regression checks exact agreement with full-boundary intersection and empty outside difference. GEOS itself can leave microscopic outside residue; at most eight budgeted intersections with the same mask may remove it. Retries are restricted to roundoff-sized displacement/area change, and nonempty final outside difference is never accepted. There is no outward buffer, snapping or accepted leakage. See [state clipping](../../src/pipeline_calculator/core/geography/corridors.py) and [integration regressions](../../tests/test_corridor_buffer_integration.py).

Single-state numerical reuse retains `source_runs.scope = "Combined"` to identify original support. These indices must not be relabeled as state-path indices: partitioning can discard a zero-length original path and compact the state path list. State metadata separately identifies the clipped output scope.

| Resource | Ceiling |
| --- | --- |
| Construction/output points per section | 100,000 |
| Parts / total rings per section | 4,096 / 8,192 |
| Charts per section / job | 1,024 / 10,000 |
| New native construction operands / input vertices | 32 / 4,096 |
| Job construction work / retained output vertices | 2,000,000 / 1,000,000 |
| Immutable full-boundary certification vertices / queries | 2,000,000 / 10,000 |

Budgets span Combined and states, charge failed work, preflight native output expansion, count ring-closing coordinates, and release replaced state output before retaining its clipped replacement. Full-boundary certification has separate explicit bounds and cancellation checks. Measured call times below are observations, not operating-system scheduling guarantees.

## Baseline, fingerprints and numerical evidence

The clean baseline is commit `9aa8698da15097eea60ee5fe5890a79042252186`. Its source archive was captured before implementation at `.validation-output/corridor-buffer/baseline/source-9aa8698.zip`, SHA256 `082cb209f20bcb702d329523f17bc7b11c23aca16bbb25c828152377164d9a9a`. The local source manifest and retained comparison receipts identify individual files. The candidate is an uncommitted working tree; the commit alone does not identify tested code.

[Numerical comparison](corridor-buffer/numeric-comparison.json) passed **25 cases**: 24 require exact numerical parity; one explicitly verifies recovery from a legacy visualization failure that erased overlap results. The comparator excludes narrowly named visual fields and map diagnostics while retaining identities, qualification, mileage, savings, geography accounting and numerical failures. Mileage goldens were not regenerated from the candidate.

Numerical/performance comparisons used Python 3.13.3, NumPy 2.2.6, SciPy 1.16.0, pyproj 3.7.1 and Shapely 2.0.7 on Windows 10. Source tests and isolated geometry measurements also exercised the build environment, Python 3.11.9 with Shapely 2.1.2. Each receipt owns its environment and source fingerprint. The comparison boundary-resource SHA256 is `55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539`.

## Verification receipts

| Check | Recorded result | Evidence |
| --- | --- | --- |
| Independent fixture audit | 106 adversarial tests; 7 fixtures; 10,250 application checks; 2,050 stress checks; zero known/unexpected production failures | [Audit](corridor-buffer/fixture-audit-report.json) |
| Numerical baseline | 25 passing cases under the policies above | [Comparison](corridor-buffer/numeric-comparison.json) |
| Geometry gallery | 19 reviewed rows: 18 ready maps and one deliberate unavailable map; independent 2 m geodesic support, 128-quadrant reference | [Receipt](corridor-buffer/gallery-report.json), [sheet 1](corridor-buffer/contact-sheet-1.png), [sheet 2](corridor-buffer/contact-sheet-2.png), [before/after](corridor-buffer/before-after.png) |
| Isolated construction | 50 cases; normal cases ready; 100-bend serpentine intentionally omitted at native-input limit | [Geometry](corridor-buffer/geometry-gates.json) |
| Native construction latency | Maximum measured call 0.001010 s; maximum cancellation latency 0.015225 s; isolated peak working set 81,289,216 bytes | [Geometry](corridor-buffer/geometry-gates.json) |
| Boundary certification | 116 cases across 51 jurisdictions; 696 calls; maximum measured call 0.005210 s | [Boundary](corridor-buffer/boundary-gates.json) |
| Broad source regression | 711 passed, 47 deselected; native UI and separate layout probes excluded | Local `.validation-output/corridor-buffer/pytest-source-final.log` |
| Native UI regression | 47 marked native tests; 31 layout tests; six repeated narrow/high-DPI probes; 30 final session/progress tests passed | [Test receipts](corridor-buffer/test-results.json), [layout regression](corridor-buffer/layout-regression.json) |
| Source smoke | Both new and legacy passed offline; 20 Summary returns each, zero callback errors, four contained state polygons, holes/multipart, parity and round trips | Local `.validation-output/corridor-buffer/source-smoke/{new,legacy}.json` |
| Frozen Windows smoke | Both entrypoints passed offline; bundled resources, repair provenance, polygon holes/multipart, state containment, numerical parity, map round trips and repeated UI navigation | [New](corridor-buffer/windows-smoke/new.json), [legacy](corridor-buffer/windows-smoke/legacy.json), [summary](corridor-buffer/windows-smoke/summary.json) |

The gallery validates all rings/components, not just a valid exterior. The fixture audit checks independent expectations and source/fixture fingerprints with no known-failure allowance. Extended packaged smoke requires the new policy and actual curved, hole and multipart geometry, so a legacy rectangle cannot pass it.

## End-to-end performance

Each baseline and candidate case used three fresh subprocesses with unchanged inputs and matching environments. Time ceiling: baseline median plus `max(1 second, 25%)`. Peak working set ceiling: baseline plus 25%. All five cases passed, retained equal numerical outputs, and were stable across runs. Memory includes imports; isolated builder time excludes clipping/exports. [Comparison](corridor-buffer/performance-comparison.json), [runs/fingerprints](corridor-buffer/performance-runs.json).

| Input / mode | Baseline median (s) | Candidate median (s) | Candidate peak bytes |
| --- | ---: | ---: | ---: |
| Three-state corridors / ordinary | 0.180 | 0.347 | 91,238,400 |
| Three-state corridors / state | 1.263 | 1.547 | 162,095,104 |
| Disconnected networks / state | 1.467 | 1.827 | 162,693,120 |
| Branching stress network / state | 3.965 | 4.188 | 175,448,064 |
| Adamas centerlines / state | 19.477 | 18.644 | 383,356,928 |

The centerlines input has no qualifying corridor sections; it measures realistic
analysis parity and total runtime, not buffer construction. The other four inputs
exercise the builder. Their isolated candidate builder medians were respectively
0.2101, 0.4112, 0.6600 and 2.1208 seconds. Those figures exclude state clipping and
exports. Baseline isolated builder timing was unavailable and is recorded as null,
not zero, in the retained runs.

## Reproduction

Run from the repository root. Keep UI and performance work serial. Use the Python environment recorded in the receipt for reproducible comparisons; the example uses the recorded Python 3.13 interpreter. The baseline checkout must be extracted from the archived clean source first.

```powershell
$corridorAuditPython = 'C:\Program Files\Python313\python.exe'
$env:PROJ_NETWORK = 'OFF'
& $corridorAuditPython scripts/validation/corridor_numeric.py generate --output .validation-output/corridor-buffer/inputs
& $corridorAuditPython scripts/validation/corridor_numeric.py capture --inputs .validation-output/corridor-buffer/inputs/cases.json --source-root .validation-output/corridor-buffer/baseline/checkout --output .validation-output/corridor-buffer/baseline/numeric.json
& $corridorAuditPython scripts/validation/corridor_numeric.py capture --inputs .validation-output/corridor-buffer/inputs/cases.json --output .validation-output/corridor-buffer/candidate/numeric.json
& $corridorAuditPython scripts/validation/corridor_numeric.py compare --before .validation-output/corridor-buffer/baseline/numeric.json --after .validation-output/corridor-buffer/candidate/numeric.json --output .validation-output/corridor-buffer/numeric-comparison.json
& $corridorAuditPython scripts/validation/corridor_numeric.py benchmark --source-root .validation-output/corridor-buffer/baseline/checkout --output .validation-output/corridor-buffer/baseline/performance --runs 3
& $corridorAuditPython scripts/validation/corridor_numeric.py benchmark --output .validation-output/corridor-buffer/candidate/performance --runs 3
& $corridorAuditPython scripts/validation/corridor_numeric.py compare-performance --before .validation-output/corridor-buffer/baseline/performance/performance.json --after .validation-output/corridor-buffer/candidate/performance/performance.json --output .validation-output/corridor-buffer/performance-comparison.json
& $corridorAuditPython tests/fixtures/geography/pipeline_kmz_regression_suite/validation/run_audit.py
.venv/Scripts/python.exe scripts/validation/build_gallery.py --output .validation-output/corridor-buffer/gallery
.venv/Scripts/python.exe -m pytest -q -m 'not native_gui' --ignore=tests/fixtures
.venv/Scripts/python.exe -m pytest -q -m native_gui --ignore=tests/fixtures
.venv/Scripts/python.exe -m pytest -q tests/test_ui_layout.py
```

The gallery command resets visual review to pending; inspect its new rendering before recording approval. Instrumented probes are retained as [geometry measurements](../../scripts/validation/measure_corridor_geometry.py) and [boundary measurements](../../scripts/validation/measure_corridor_boundaries.py). Run either with `.venv/Scripts/python.exe`; they write to `.validation-output/corridor-buffer/`. Durable regressions are [buffer tests](../../tests/test_corridor_buffer.py), [qualified-run tests](../../tests/test_corridor_qualified_runs.py), and [integration tests](../../tests/test_corridor_buffer_integration.py).

Source smoke (repeat with `legacy` and its output filename):

```powershell
$env:PIPELINE_CALCULATOR_IMPL = 'new'
$env:PROJ_NETWORK = 'OFF'
.venv/Scripts/python.exe scripts/validation/gui_process.py --timeout 60 -- .venv/Scripts/python.exe src/pipeline_calculator_entry.py --smoke-test .validation-output/corridor-buffer/source-smoke/new.json
```

## Final source and Windows completion

The final source verification covers **789 application tests across three disjoint suites** (711 broad, 47 marked native, 31 layout), with no remaining failure. An additional 30 session/progress tests passed after the layout fix. [Commands and retained logs](corridor-buffer/test-results.json).

The first broad run exposed three stale visual expectations, corrected without changing numerical assertions, plus an existing intermittent layout hang. The untouched baseline also timed out at 512x288/250%. Tracing found an inactive uniform grid column competing for width in the caution dialog, repeatedly changing text wrapping. Removing that inactive column from the uniform group resolved the loop: all 31 layout tests and six serial repeats passed, including both entrypoints. [Diagnosis, baseline evidence and regression checks](corridor-buffer/layout-regression.json).

The final [82-file source manifest](corridor-buffer/final-source-manifest.json) identifies the exact candidate. Only this isolated dialog layout fix followed the numerical, geometry, export and performance receipts; all their implementation files remain byte-identical. Frozen-build verification is recorded separately below.

| Completion field | Final value |
| --- | --- |
| Final broad source command/result | `pytest -q -m 'not native_gui' --ignore=tests/fixtures --ignore=tests/test_ui_layout.py`: **711 passed, 47 deselected** |
| Final native/layout command/result | 47 native tests, 31 layout tests, six repeated probes and 30 final session/progress tests passed; see linked receipts above. |
| Final tested source fingerprint | [Source manifest](corridor-buffer/final-source-manifest.json), including the single UI-only difference from the calculation audits. |
| Sample state export manifest/workbook inspection | [Sample manifest](corridor-buffer/sample-export.json): complete NM/OK/TX analysis, one workbook with seven sheets, JSON, Combined and three state KMZ maps; all workbook text fits Excel's cell limit. Package retained under `.validation-output/corridor-buffer/sample-exports`. |
| Windows build/version | `4.24-dev.9aa8698da150.dirty`; [build log](corridor-buffer/logs/windows-build.log). Local uncommitted development build. |
| Frozen offline smoke, new / legacy | Both passed with `PROJ_NETWORK=OFF`; [summary](corridor-buffer/windows-smoke/summary.json). |
| Published executable and fixture location | [Usual executable](../../dist/Pipeline_Calculator_v4.exe) and [versioned executable](../../dist/Pipeline_Calculator_v4.24-dev.9aa8698da150.dirty.exe). Existing synthetic [KMZ fixtures](../../tests/fixtures/geography/pipeline_kmz_regression_suite/fixtures/) remain in the repository. |
| Executable SHA256 and size | `418C2363C54545DD02EFF8202B22E05801A1B1CCC61AEA2251A8D6AFAA48BC3B`; 102,622,252 bytes. Tested build, versioned copy and usual executable match. [Publication receipt](corridor-buffer/windows-publication.json). |

The existing executable was retained under `dist/archive/Pipeline_Calculator_v4_before_corridor_buffer_20260918_100729.exe`. Existing running instances were not closed; reopen the app to load the new build. Prior versioned executables and fixture files were preserved.

Executed isolated Windows build and verification:

```powershell
$corridorBuildRoot = Join-Path (Get-Location) '.validation-output\corridor-buffer\package'
& '.\scripts\windows\build_exe.ps1' -OutputRoot $corridorBuildRoot
$corridorVersion = (Get-Content -LiteralPath (Join-Path $corridorBuildRoot 'build\version.json') -Raw | ConvertFrom-Json).version
$corridorExe = Join-Path $corridorBuildRoot ('dist\Pipeline_Calculator_v' + $corridorVersion + '.exe')
.venv/Scripts/python.exe scripts/validation/check_packaged_smoke.py $corridorExe --output-directory (Join-Path $corridorBuildRoot 'smoke') --expected-version $corridorVersion
Get-FileHash -LiteralPath $corridorExe -Algorithm SHA256
```

## Remaining limits

These are approximate display areas around qualifying centerlines, not rights-of-way or surveyed ownership. Boundary accuracy and transformation limitations remain those of the bundled 2025 Census resource. Complex, extreme projection-domain or over-budget maps are explicitly omitted while numbers remain available; the measured 100-bend serpentine is one example.

The local Windows build passed frozen checks; that does not clear macOS. Verify native dependencies, offline resources, both GUI entrypoints, high-DPI layout and map opening on a Mac before distribution. No remote release, signing or notarization was performed.
