# Follow-up runbook: owner input and external validation

Prepared September 10, 2026 against `b058ac08ed08ed9e0069d5942328252c69cefb1a`.
Status: **automated prerequisites complete; further acceptance testing deferred**.
Owner clarified priorities and corridor purpose on September 10, 2026. Remaining
data review, platform/viewer checks and performance acceptance are future work;
they are not requests for immediate input. No release was published.

This is the follow-up to [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md). It contains
only work that currently needs unavailable evidence/access, an interactive platform
review, or an owner decision. It does not block the synthetic tools, cancellation,
progress, launcher fixes or available local packaging work. The automated task completed without owner intervention. The next task can work
from the prepared evidence below.

## Owner decisions recorded September 10, 2026

- First priority: accurately calculate the total distance of all source polylines
  representing the pipeline system. Keep this original total distinct from any
  estimated reduction due to grouping/overlap.
- Grouping and overlap are supplemental estimates whose accuracy remains valuable.
  Corridors show the rough location and extent of identified bundled segments;
  they are visual guides, not surveyed boundaries or flight/sensor coverage.
  No sensor width or flight assumptions are required from the owner.
- Favor reliable operation and reasonable visual approximation. This does not
  establish a numerical error allowance for source mileage or permission to hide
  missing geometry. Retain usable valid data with clear incomplete/error notices
  where supported; do not fabricate distances to finish a calculation.
- Defer additional testing and owner acceptance until a future task. Previously
  completed automated checks remain evidence; deferred checks are not marked passed.
- GitHub Actions provides the macOS runner. Defer its execution/artifact follow-up
  to R3; do not ask the owner to supply a Mac for CI builds.
- R4 is an optional check of real project speed, not a proposed extra step in the
  everyday app workflow. No runtime target or hardware decision is needed now.

Subsequent owner direction: R1 real files are explicitly deferred. Current speed
is satisfactory; R4's automated addition is a warning only for extreme workloads.
R5's further code investigation and fixes precede manual review. Collect remaining
manual tests into one future session after autonomous fixes and artifact preparation;
do not ask the owner to test follow-ups piecemeal. See
[workload/corridor hardening](docs/validation/workload-corridor-hardening.md).

## Entry checklist and handoff packet

The following evidence is prepared and verified:

| Automated work | Prepared evidence |
| --- | --- |
| A0/A9/A10/A11 | [Latest completion report](docs/validation/workload-corridor-hardening.md): 193 passing tests, exact source identity and limitations |
| A5 | [Manifest template](docs/validation/corpus-manifest.md), [final corpus results](docs/validation/corpus-final.json); tracked input explicitly unreviewed |
| A3 | [Performance/output comparison](docs/validation/comparison.md), [cancellation timings](docs/validation/cancellation.json); raw profiles in `.validation-output/` |
| A4 | [Reference model](docs/validation/reference-model.md), [error table](docs/validation/reference-results.md), [280 comparisons](docs/validation/reference-results.json) |
| A7/A11 | [Refreshed gallery](docs/validation/hardening-gallery.md), [geometry evidence](docs/validation/hardening-gallery.json); source/overlay KMLs in `.validation-output/hardening-gallery/` |
| A6/A10/A11 | [Refreshed preview hashes/versions/smoke](docs/validation/hardening-packages.json); warning fixture hash included |
| A8 | Recovery, retry, saved-file preservation and platform failures covered by automated tests; actual OS association/viewer review remains below |

Windows previews are under `%TEMP%/pipeline-calculator-hardening-20260910/new/dist`
and `legacy/dist`. Both report `4.3-dev.b058ac08ed08.dirty`; use the recorded hashes
to identify their exact content. The existing repository build/dist artifacts were
preserved. No macOS executable was built on this Windows host.

| Follow-up | Current disposition |
| --- | --- |
| R1 | Deferred: representative real files and any known expected distances, when convenient; no immediate input needed |
| R2 | Purpose/priority resolved: source mileage first, supplemental grouping, rough visual corridors; numerical acceptance testing deferred |
| R3 | Deferred to a future task: macOS via existing GitHub Actions runner and native interactive checks; Windows build/smoke evidence retained |
| R4 | Extreme-workload warning implemented automatically; interaction review joins the consolidated deferred session; performance tuning optional |
| R5 | Additional autonomous geometry fixes implemented; Google Earth review joins the consolidated deferred session |
| R6 | No access/signing blocker encountered for local work; release publication remains optional and separately authorized |

Update the disposition table as follow-up evidence arrives. Do not call an unrun
check passed; send reproducible implementation defects back to an autonomous fix task.

## Consolidated deferred manual session

Prepare one packet after autonomous hardening and available package checks finish:

1. Record exact preview versions, source fingerprint and file hashes. Use refreshed
   hardening artifacts, not the earlier previews with the same development version.
   macOS CI/artifact follow-up remains a future task.
2. When testing resumes, collect R1 examples once, including the formerly shortened
   group and any trusted distance comparisons. No real files are required now.
3. On those files, combine R2 source-distance checks, R3 import/display/progress/
   cancel/retry/export, and R5 corridor rendering. Check source distance first and
   assess approximate group visuals separately.
4. Include the extreme-workload Continue/Cancel interaction (R4) using a prepared
   synthetic fixture; the owner need not create a giant private project.
5. Record one issue list with file/settings/artifact identities. Return confirmed
   defects for autonomous fixes and repeat only affected checks. Unknown real-data
   and viewer behavior stays pending until that session occurs.

R6 distribution remains optional and separate. No release is required for testing.

## R1 — Approve and review representative project data

**Plain-language request, deferred:** when ready, identify a few real KML/KMZ files
that represent your work, especially one whose grouping looked too short. Tell us
which area looked wrong and any distance you already know from another trusted
source. Exact expected answers are optional; you do not need to calculate them or
design tests. Automation handles comparisons and records remaining uncertainty.
Existing tracked data can continue to be investigated locally; permission is only
an unresolved issue for new sharing/redistribution or additional private data.

**Why owner input is needed:** a tracked sample exists, but neither its provenance
nor its correct source/adjusted totals are documented. Automation cannot establish
permission to redistribute project geometry or invent real-world expected results.
Original task 5 and production portions of tasks 3/4/7 land here.

Evidence: [tracked KMZ](.danny/Centerlines.kmz), roadmap V5/V6,
[parser](src/pipeline_calculator/parsers/kml_kmz.py),
[analyzer](src/pipeline_calculator/core/analyzer.py).

Owner/data custodian provides:

- Provenance/intended use of `.danny/Centerlines.kmz`; whether it may be used in
  shared fixtures/reports. It is already tracked; this runbook does not instruct
  deletion or history rewriting.
- Approved examples of ordinary projects, dense bundles, curves/branches, multipart
  geometry, linked exports and known troublesome inputs. One file may cover several
  categories; no arbitrary minimum file count is necessary.
- An approved local/private storage location, handling constraints and the person
  qualified to review geometry and expectations. Avoid credentials in documents.
- Independent source mileage/feature counts where available, known overlap intervals,
  diagnostics expected from source data, and uncertainty where truth is unknown.

Procedure after the files/input are available:

1. Copy the prepared manifest template; record hashes, purpose, settings, provenance,
   basis of expectations and reviewer. Leave unknown expectations unasserted.
2. Run the prepared corpus tool against the private root. Keep raw geometry/names
   local. Compare original lengths independently of adjusted mileage; agreement
   with an older app is a regression baseline, not proof of correct mileage.
3. Review discrepancies with source GIS evidence. Classify parser loss, source-length
   disagreement, sampling/group choice, known source problem, or unknown. Record
   disputed intervals and parameters so they can be reproduced.
4. Convert confirmed defects into small synthetic reproductions where possible;
   return those to an autonomous fix task. Retain a private check on the original.
5. Mark a fixture reviewed only after the expected fields and uncertainties have
   actually been reviewed. Run strict reviewed-corpus mode for acceptance.

**Exit evidence:** reviewed manifest entries, comparison results, reviewer/date,
coverage categories and unresolved deviations. If files are unavailable, retain
synthetic-only status; do not describe the app as production-survey validated.

## R2 — Operational accuracy and any changed calculation/geometry rules

**Purpose resolved; numerical acceptance deferred:** the owner has specified source
polyline distance as the priority and corridors as rough group visualizations.
No additional purpose clarification is needed. Tests can quantify error within a
declared model; a future material change in business rules or numerical accuracy
tradeoff needs concrete examples before any remaining owner decision is raised.
Original task 4's operational acceptance belongs here.
The current all-pairs grouping/minimum rules are already established and do not
need to be asked again.

Sources: [current calculation contract](README.md#minimum-parallel-length),
[segmentation](src/pipeline_calculator/core/segmentation.py),
[bundling](src/pipeline_calculator/core/bundling.py),
[method metadata](src/pipeline_calculator/core/analyzer.py).

Automation brings a concrete decision packet:

- Meter/relative errors across segment lengths, threshold cases and reviewed jobs;
  separate source-length, sampling, projection and grouping-heuristic effects.
- Specific unsupported cases, if any (for example branch/turn correspondence in the
  reference). Do not ask the owner to choose an algorithm without examples.
- For any proposed changed rule, current/proposed totals and intervals on the same
  cases, beneficiaries, tradeoffs, compatibility impact and recommended option.

The prepared error table includes large qualification discontinuities at exact
range/minimum boundaries and coarse hairpin cases. Compare section sums with unique
interval coverage before interpreting the size of a reported difference. The
savings gap combines sampling/grouping and does not isolate heuristic loss.

Owner decisions, only where still unresolved:

1. After testing resumes, assess observed discrepancies against the stated source-
   mileage priority. Ask about an acceptable error only if a concrete unresolved
   tradeoff remains; do not require a percentage or meter budget in advance.
2. Whether the measured current approximation is acceptable; if not, which
   error/performance tradeoff is acceptable. Do not silently change segment defaults.
3. Whether any new grouping/coverage rule should replace the existing conservative
   heuristic. Optimal grouping is not automatically the chosen flight-planning rule.
4. Corridor purpose is settled: show the rough extent of identified groups. Judge
   endpoints and shape readability against that purpose. Surveyed boundaries,
   sensor coverage and flight planning are outside this scope.

Procedure:

1. Review analytical evidence before representative examples; separate facts from
   operational judgments. Record decisions with the exact report revision.
2. If current behavior is accepted, document supported use/limitations and close
   this gate without unnecessary production changes.
3. If a change is selected, create a follow-up implementation scope with explicit
   numerical acceptance cases. Version computation-method metadata when semantics
   change and document how historical results are interpreted. Preserve original
   mileage and unrelated contracts.

**Exit evidence:** owner/date, tolerances, supported geometry/use, selected rule or
explicit decision to retain current behavior. Engineering oracle epsilon and
projection checks must never be presented as owner-approved survey tolerances.

## R3 — Interactive packaged Windows/macOS acceptance

**Why this is separate:** automation can build and test source/withdrawn widgets,
but this review has only a Windows host and did not inspect actual packaged display,
focus, scaling, drag-and-drop or installed associations. macOS execution requires
the existing GitHub Actions runner (`build-macos`, `macos-14`, Python 3.11).
The owner explicitly deferred this follow-up to a future task. Original task 6's
native review lands here; no workflow execution is requested now.

Sources: [main GUI](src/pipeline_calculator/gui/main_window.py),
[legacy GUI](src/pipeline_calculator_v3.py),
[Windows build](scripts/windows/build_exe.ps1),
[macOS build](scripts/macos/build_app.sh),
[workflow](.github/workflows/build.yaml).

Required resources when resumed: prepared exact-source Windows previews; GitHub
Actions macOS results/artifacts for the intended source revision; an interactive
test session for remaining visual checks. First inspect the existing CI run and
artifact evidence; execute missing CI checks in that future task as authorized.
CI build success does not establish visual acceptance. Signing secrets
or privileged machine changes belong to R6, not an improvised workaround.

Procedure on each platform:

1. Record OS/architecture, screen resolution/scaling, artifact path/hash/version,
   implementation choice and source/patch identity. Use the prepared synthetic
   inputs. Keep preexisting build outputs outside script-cleaned directories.
2. Test the default GUI, then legacy differences: browse/import, drag-and-drop,
   corrected parameters, long paths/names, reanalysis and successful export.
3. Test progress, cancellation during several stages, retry after cancellation,
   close during work, startup failure, empty/point-only input and incomplete results.
4. Review small/large layouts, scrollbars, pagination past 20 rows, final-page controls,
   clipped critical values, keyboard navigation, dialogs and focus restoration.
5. Test native KML association success/failure and recovery (Copy Path/Save As/Retry).
   Do not change system associations just to manufacture a failure; mocked failure
   coverage already exists. Coordinate real viewer checks with R5.
6. Attach annotated screenshots and exact steps for each failure; send reproducible
   defects back to an automated fix task and repeat only the affected checks.

**Exit evidence:** per-platform checklist with outcome, artifact identity, screenshots
where useful and exact not-run reasons. A passing build or withdrawn widget test
cannot substitute for this review. Testing previews does not authorize publishing.

## R4 — Representative performance and responsiveness budgets

**Current decision:** ordinary performance is acceptable. A10 adds the requested
warning at 750,000 estimated segments or 10,000,000 estimated neighbor inspections.
After import/source measurement (and indexing for density), pause before costly
overlap comparisons with Continue anyway, Cancel and split/simplify-copy guidance.
Preparation/waiting stays cancellable. Existing hard caps still apply. Its manual
interaction check joins the consolidated session; no hardware/speed decision is
needed now. The older optional performance procedure below remains deferred.

**Suggested approach, optional and deferred:** if normal use reveals a slow job,
use that file and an ordinary project to measure where time is spent. Optimize
verified bottlenecks while preserving mileage. The owner need only identify a
slow file and, if relevant, what wait feels disruptive; automation gathers machine
details and timings. Formal time/memory budgets are optional.

**Actual user workflow:** choose the KML/KMZ, adjust settings if needed, click
Analyze, see progress, then review/export results. Cancel remains available during
long work. Only exceptionally large/dense jobs add the Continue/Cancel choice.
No benchmark screen or setup wizard is proposed.

**Why input is needed:** A3 can optimize measured costs, but production workload
sizes, available hardware and acceptable wait/memory use are owner constraints.
No representative production budget was established by this review.

Required evidence: A3 synthetic/local measurements, R1 reviewed project categories,
and the target machine/typical maximum project description.

1. Use available ordinary and slow/large examples. Ask for a missing file only when
   needed; record the machine automatically. Request a wait-time constraint only
   if necessary to resolve a demonstrated performance/accuracy tradeoff.
2. Automation runs the prepared benchmark on approved inputs/machines using recorded
   settings. Separate native memory from Python allocations; separate warm runs,
   profiler overhead and single-call cancellation delays.
3. Compare median/range, peak memory, completion/diagnostics and output equivalence
   against proposed budgets. Report tail cases rather than only average speed.
4. If budgets are missed, prepare a measured optimization scope. Changes to data
   structures within existing semantics can run autonomously; accuracy/default
   tradeoffs return to R2. Do not merely raise processing limits.

**Exit evidence:** workload/hardware-specific budgets, measured outcomes and a list
of exceptions. If no budget is requested, retain a measured performance report
without inventing a guaranteed runtime or blocking other acceptance checks.

## R5 — Google Earth corridor presentation

**Status:** further viewer testing deferred. The visual purpose is resolved in R2.
Additional automated investigation is implemented in A11: rectangle fallbacks
enclose qualified samples across bends, curved ends are padded, unused malformed
bounding boxes do not break valid exports, and large self-crossing or excessively
costly outlines use disclosed fallbacks. Grouping/mileage rules are unchanged.
Use the refreshed gallery and previews for the consolidated session. See
[new findings and evidence](docs/validation/workload-corridor-hardening.md).

Recent-change investigation (September 10, 2026; code and Git diffs inspected):

- September 9, `39f3260` (Bug Fixes): corridor joins previously replaced the last
  accumulated point repeatedly, discarding earlier extent. Joins now append their
  points. The [extent regression](tests/test_calculation_regressions.py) checks that
  a 300 m straight section retains approximately its full length in the outline.
  This is directly relevant to previously shortened shapes.
- September 9, `b058ac0` (Bug Fixes 2): finite-segment overlap comparisons replaced
  a midpoint-distance requirement that missed nearby lines whose sample positions
  did not align. This can recover missed sections. See the offset regression in
  [follow-up tests](tests/test_followup_audit.py) and [audit](FOLLOWUP_AUDIT.md).
  The same commit tightened multi-pipeline savings groups to require every pair
  to qualify; it is not a general instruction to make corridor outlines longer.
- Initial automated batch (before A11): KML output validates outlines and uses
  a disclosed rectangle fallback for invalid curved shapes. This can change the
  displayed shape, but does not extend qualifying groups or change source mileage.
  Six benchmark analysis outputs stayed equivalent, excluding added metadata and
  input path normalization ([comparison](docs/validation/comparison.md)).
- A specific old short grouping is not confirmed fixed without its source file and
  settings. Sampling, gaps, turns and qualification thresholds can still limit the
  displayed extent. Keep that example for deferred R1/R5 review; do not infer a
  mileage loss solely from the approximate outline.

**Why external review is needed:** local geometry/KML checks do not prove Google
Earth renders seams, bends, centers and distinct paths as intended. This requires
an available viewer and interactive inspection. Original task 7's visual gate and
part of task 8's real launch behavior land here.

Sources: [geometry](src/pipeline_calculator/core/overlap.py),
[KML exporter](src/pipeline_calculator/export/corridor_kml.py),
[launcher](src/pipeline_calculator/gui/actions/open_kml_action.py).

Concrete gallery findings: right-angle, hairpin and loop preferred outlines can
self-cross; their exports now disclose valid rectangle approximations. Review
whether those broad shapes communicate the intended purpose. All ring sizes now
receive bounded topology checks; excessive work uses a disclosed simpler fallback.
Local checks are not viewer approval.

Procedure with A7's prepared gallery:

1. Record viewer/version/OS, app artifact, fixture ID, settings, KML hash and numerical
   validity/approximation status. Open synthetic files first, then approved R1 examples.
2. Overlay qualified source paths with the exported shape. Inspect endpoints, bends,
   hairpins, loops, short/disconnected sections, reversed paths, dateline and poles.
   Check that centers, labels and approximation descriptions remain understandable.
3. Inspect world-spanning seam artifacts, self-crossing fills, clipped extents and
   erroneous links between distinct parts. Verify saved-file recovery manually if
   launching fails; OS acceptance alone does not certify rendering.
4. Capture each issue with source/shape overlay, camera position or clear reproduction
   steps and exact KML. Classify numerical/export defect, rendering issue, source
   issue or operational-purpose disagreement. The last category goes to R2.
5. Return confirmed reproductions for bounded automatic fixes, regenerate affected
   files and repeat those checks. Do not use a visually plausible polygon as proof
   of exact mileage or sensor coverage.

**Exit evidence:** gallery table with per-case visual outcome, approximation status,
reviewer/date and screenshots for failures. Unavailable viewer cases remain pending.

## R6 — External access, signing and optional distribution

**Why authorization/resources may be needed:** macOS hosts, credentials, signing,
notarization, CI access and release publication are separate from local improvement
work. Existing [build workflow](.github/workflows/build.yaml) publishes tagged
builds; [CODE_SIGNING.md](CODE_SIGNING.md) describes signing setup.

1. First enumerate the exact unavailable resource or desired distribution action.
   Routine local dependencies/build fixes stay in A0/A6; do not ask for credentials
   or publishing approval speculatively.
2. For access-only checks, obtain the appropriate environment through the approved
   mechanism and run the prepared command. Keep secrets out of manifests/reports.
3. If distribution is wanted, present the exact tested artifact/source, validation
   summary, remaining limitations and destination for authorization. Preview-build
   permission is not release publication permission.
4. Follow existing Git-derived version/tag/signing rules. Do not bypass main/tag
   signing requirements, reuse published tags, or change version history to ship.

**Exit evidence:** access checks completed, or a separately authorized release record.
If distribution is not requested, mark publication out of scope; it does not prevent
completion of the automated improvements or validation runbook.

## Follow-up completion record

Maintain one row per item: `R-ID | owner/reviewer | input/artifact | decision/outcome |
evidence | date | remaining action`. A decision can retain the current behavior.
Record unavailable checks and uncertainty explicitly. New fixes should carry their
own source revision and affected revalidation list rather than silently changing
an already-reviewed artifact.
