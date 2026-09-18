# State boundary audit resolution

The five confirmed feature fixes and fixture regressions are implemented on baseline
`6bc6e5f7aa083604edbaa25f177f671f58ea15d6`. The full suite passes **437 tests**, with
zero failures, errors or skips (363.85 seconds). A fresh Windows executable passes
offline smoke checks in **both modern and legacy GUI modes**, with empty stderr.
Exact source hashes, test outcomes and artifact identity are retained in
[state-boundary-resolution.json](state-boundary-resolution.json).
The user authorized the [resolution plan](../../STATE_BOUNDARY_ANALYSIS_RESOLUTION_PLAN.md)
after the other task's changes were committed. Existing readiness documents and
fixture data are preserved. No commit, push, or distribution is part of this work.

## Implemented fixes

| Finding | Resolution |
| --- | --- |
| SB-01 / SB-02 | Local ellipsoidal AEQD coordinates now come directly from GRS80 inverse distance and azimuth. Exact shared arcs use canonical source endpoints; independently rounded stations no longer create false unresolved tails. Numerical uncertainty remains explicit. |
| SB-03 / SB-04 | One core corridor utility selects preferred/oriented/bounding representations, normalizes dateline polygons, and records fallback/omission decisions. State clipping preserves holes and certified coordinates. Map serialization splits source lines at the antimeridian on their original geodesic. |
| SB-04 reporting | Combined visualization decisions are prepared before publishing the analysis. Map export preflights a separate snapshot before writing workbook/JSON. Unusable visuals produce diagnostics while source lines and reports remain available; filesystem errors still fail the atomic package. |
| SB-05 | Public float callbacks use the same aggregate progress as desktop contexts, including state stages. Completion follows the assembled result; cancellation produces no completion event. Ordinary-mode callback behavior is retained. |
| Fixture regression | Both original user archives are now consumed by automated tests using independent XML/GRS80 expectations. |

Shared-border verification follows the declared models: source edges are geodesics,
and boundary edges are coordinate-linear. A positive common arc must be geometrically
verified; closeness alone does not assign a nearby line to a shared border. Genuine
short crossings remain positive intervals. Boundary provenance and source accuracy
limitations are unchanged.

## Verification

- Geometry and state accounting: 50 focused tests passed. The same 500 bundled
  shared-border directions used in the audit now produce no unresolved fragments or
  diagnostics, with every reconciliation passing. The pre-fix set had 53 unresolved
  cases; this comparison is not a population failure-rate estimate.
- Corridor/export verification: 104 affected checks passed, followed by a final
  27-test visualization/export run including 17 new cases. Tests cover the actual
  dateline input, fallback and omitted visuals, holes/multipart shapes, non-round
  state borders, containment after serialization, immutable snapshots and diagnostics.
- Independent code review found no additional confirmed production issue. A separate
  200-direction dateline probe preserved source mileage with a maximum observed
  error of approximately `2.3e-9 m`; repeated preflight preserved geometry and did not
  duplicate diagnostics.

- Public aggregate progress and existing execution/state contracts: 46 passed,
  one native case excluded from this focused run. A subsequent 24-test progress/state
  integration run passed after preparing Combined visualization diagnostics.
- Real fixture regressions: six passed. The polygon archive contributes no pipeline
  mileage in either mode. The centerline archive preserves all 281 sources / 4,827
  paths, their independently expected state ownership and original lengths, zero
  crossings, and four Combined/state KMZ mileage round trips.
- Unavailable corridor actions: three tests passed, including a native row-action
  check on the isolated Windows desktop. An omitted visualization is labelled
  “Map unavailable” and cannot launch through pointer, keyboard, or direct routing.
- Both GUI input routes: two native tests passed, retaining actual input-page wiring
  and parameter dialogs while replacing the file picker, drop transport and worker.
  They verify Browse/retry/drop options, busy suppression, Apply/Cancel behavior,
  preference reload in a new application instance, and concise unsupported-input status.
- Durable sample packages: real Centerlines, shared Texas/New Mexico, the default
  antimeridian input, and a Texas/Oklahoma overlap pair. All nine map mileage round
  trips pass; maximum observed difference is `4.89e-9 m`. Workbook organization,
  accounting, JSON, polygon validity and state containment were checked. Sample
  evidence records the `covers`/empty-set-difference containment fallback explicitly.
- Visual inspection: Combined and Texas views, the comparison table, shared allocation
  notes and the real polygon-only error state were captured on the isolated Windows
  desktop. Labels remain readable; lower content is accessible by scrolling. No
  callback or Tcl stderr errors were recorded. Images are in
  `.validation-output/state-boundary-resolution/ui-review/`.

## RV-01 investigation

The 100-replacement scenario is making steady progress rather than hanging. Corrected
diagnostic runs completed all original ownership, callback and heartbeat assertions
in 74.98 seconds without a scheduled dump and 77.22 seconds with the old 20-second
traceback dump. Both exited successfully. The original 45-second process deadline
was shorter than this healthy workload. These are observations on this Windows host,
not strict performance benchmarks.

Only this long correctness scenario now has a 120-second process deadline and a
110-second diagnostic traceback. Other native tests retain their 45/20-second limits.
Fatal-error traceback capture remains enabled, progress is printed every ten completed
replacements, and all existing 100-cycle cleanup, binding/style, callback and heartbeat
assertions remain. The native harness now also rejects Tcl errors on stderr even when
the process exits with zero. No production widget or scrollbar replacement was adopted.
The comparison heartbeat maxima were approximately 699 ms and 562 ms: they pass the
ordinary two-second correctness watchdog, and do not meet or certify the separate
250 ms strict reference target.

The earlier Windows access violation was observed again during an instrumented probe,
but its cause remains unestablished. Successful runs both with and without scheduled
dumps do not establish dumping as its cause. This work does not claim to have fixed a
proven application freeze or identified that intermittent native fault. Two consecutive
post-change 100-cycle acceptance runs passed, followed by the full 437-test suite.
The focused lifecycle/shared suite passed 11 tests. Each acceptance run released every
retired view, retained one style variant, recorded all ten progress checkpoints and
reported no callback/Tcl stderr errors. Native fault diagnosis remains open.

Detailed outputs are under `.validation-output/state-boundary-resolution/`,
`.validation-output/state-boundary-fixes/` and
`.validation-output/state-boundary-implementation/`. These are focused runs, not
additive counts of unique tests. The final executable is
`4.18-dev.6bc6e5f7aa08.dirty`, SHA-256
`91152451e738cb39dba1d132815fdb8788100ddbc1eb7e45e536188b84945116`.
Its smoke checks verified all 51 bundled jurisdictions, Texas/Oklahoma state analysis,
reconciliation and map reimports, and 20 Summary/disclosure cycles in each GUI mode,
with `PROJ_NETWORK=OFF`. Previous `dist/` artifacts were preserved.

Review artifacts:

- [Windows executable](../../.validation-output/state-boundary-resolution/package/dist/Pipeline_Calculator_v4.18-dev.6bc6e5f7aa08.dirty.exe).
- [Real Centerlines workbook](../../.validation-output/state-boundary-resolution/samples/adamas_ng_pipeline_row_centerlines_analysis_20260915_102238/analysis.xlsx).
- [Sample package accounting and map checks](../../.validation-output/state-boundary-resolution/samples/sample-evidence.json).
- [Combined comparison table](../../.validation-output/state-boundary-resolution/ui-review/Combined-details.png) and [Texas shared-mileage view](../../.validation-output/state-boundary-resolution/ui-review/Texas.png).

Before release signoff, retain the historical native-crash investigation as open,
run strict timing checks on a reference host, and complete actual macOS packaged and
visual validation. A repeat native fault should be captured with its exact artifact,
OS/Tk versions and a native exception dump in the isolated test process before choosing
a production change. Passing current functional checks does not explain the earlier
access violation. Native macOS testing was unavailable on this Windows host.

Evidence handling note: the ignored readiness geometry `recheck.json` was accidentally
overwritten by a post-fix probe. Use the tracked
[readiness summary](state-boundary-readiness.json) and original audit evidence for
pre-fix observations; the geometry implementation evidence is retained separately
under `.validation-output/state-boundary-fixes/geometry/`.
