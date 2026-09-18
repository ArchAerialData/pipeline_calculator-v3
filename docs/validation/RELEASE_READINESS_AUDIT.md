# Sales rollout audit — September 18, 2026

## Scope and current decision

Audit baseline: `1562fa3d63c6662ea3f314b1bade3edae72dd52a` (`KMZ Repair 5`).
Sales comparison baseline: `5da6b8677c86b3d6a856509526df68cac5be68b7`
(`Version 4.0.0 Updates`, May 26, 2026).

**Automated audit complete; identified critical defects fixed and final native
build/runtime gates passed.** No unresolved critical calculation defect was
reproduced in the exercised scenarios. Before Sales distribution, complete Mac
notarization/stapling and downloaded-app acceptance, including a current retail
Golden Gate Mac. The customer metadata discrepancy below also requires client
clarification before that file is used as a mileage benchmark.

The review covers GRS80 original mileage and US survey mile conversion, sampled
overlap savings, state partitions/shared allocations, corridor polygons and
exports, safe repair/source snapshots, GUI result/error states, and native Windows
and macOS distribution. Tests and probes run against local synthetic fixtures;
the customer KMZ stays local and unchanged.

## Confirmed defects addressed

| Priority | Defect | Resolution and proof |
| --- | --- | --- |
| P1 | Valid XML could contain multiple coordinate blocks, nested coordinate markup, or nested Placemarks; parsing could omit or double-count geometry while reporting Complete. | Both direct parsing and prepared-source analysis now use a shared structural guard. Ambiguous files are rejected with client-correction findings. 24 new compatibility/safety regressions; 225 focused parser/repair tests passed. |
| P1 | The signed app could not start on Golden Gate: macOS 27 rejected an older SciPy PROPACK native library. | Reproduced in both packaged entrypoints and traced to the official upstream wheel. Pin NumPy 1.26.4 and SciPy 1.17.1; the official replacement passed direct native macOS 27 import/numerical probes. No binary patching or skipped imports. Full packaged verification is recorded below. |
| P2 | A closed loop followed by a retraced edge could lose its main corridor map. | Split temporary map construction at repeated source vertices, retaining exact generating support and all certification tolerances. All 72 loop/retrace probes pass; source geometry, mileage and savings are unchanged. |
| P2 | Corridor union batches could exceed the native vertex limit even when their operand count was small. | Batch by both limits. The actual macOS failure was reproduced locally with sub-nanometer coordinate perturbations; nine long-route variants now pass without increasing a resource limit. |
| P2 | Combined overlap failure appeared as “No bundled sections” in the Overlap tab. | Show analysis unavailable with a Diagnostics reference. Successful empty-overlap and single-pipeline results retain their correct messages. A native regression starts with an actual injected analyzer failure. |
| P2 | Exact sampling boundaries could produce different overlap savings on Windows and macOS. | Apply a terminal-only allowance bounded by 1 µm and one millionth of the sample length, clamping to the actual endpoint. Independent cumulative-chainage reconstruction and exact golden assertions verify the amended sampling policy. See [terminal sampling policy](terminal-sampling-policy.md). |
| P2 | Collapsing Summary details and switching tabs before layout settled could leave Summary permanently blank. | Reproduced with real native navigation and an unmapped canvas child. Recover the viewport on canvas mapping and maintain it based on canvas visibility. Native navigation regressions pass; final packaged-platform verification is recorded below. |
| P2 | A narrow, high-DPI, short window could clip repair notices and leave the result pages unmapped. | Compact the context header and footer labels only when needed; recover Summary spacing from its parent even before the canvas is visible. Native checks include wider fonts, native macOS point scaling, and Windows 250% scaling. An actual 380×230 logical viewport retains controls, scrolling, the complete mileage value and its units; normal labels and spacing return on resize. |
| P2 | Rapid repair/reimport or Summary teardown could leave queued callbacks referring to a retired source or destroyed view. | Capture the displayed repair identity/status and prevent old controls from acting on replacement sources; retire all scroll scheduling when its content is disposed. Both exact failures reproduced locally before the fixes. Fifteen repair/header checks passed; the subsequent combined layout/scroll/header suite passed all 58 checks. These suites overlap and are not additive. |

## Independent evidence

- State audit: 130 focused tests; seven frozen KMZs with 10,250 independent
  assertions across 25 exported scopes; 204 additional forward/reverse partitions
  across all 51 supported jurisdictions. Maximum independent allocation difference
  was 0.0000002263 m. No critical state-partition defect was reproduced.
- Corridor audit after fixes: 169 focused tests; 84 independent randomized
  neighborhood/topology checks; 72 loop/retrace cases; nine perturbed long routes;
  six actual KMZ map roundtrips with maximum mileage error 1.357e-9 m.
- Starting local test runs: 717 broad tests, 54 native GUI tests, and 31 startup/
  layout checks passed. These are separate baseline runs, not a final patched-build
  count. Targeted suites overlap with the broad run and must not be summed.
- Fresh own-window captures reviewed for Summary, Overlap, state/repair header,
  and expanded human-readable repair details. Native checks exercise high DPI,
  narrow windows, control containment, text fit, keyboard use, and both entrypoints.
- Independent original-mileage probe: 500 global paths, compared with GeographicLib
  using the same GRS80 ellipsoid. Maximum difference: 1.1175870895385742e-8 m.
  All 64 analytic/bounded overlap controls passed. In four-line chains, reversal
  changed conservative estimated savings by at most 15 m; source order changed none.
- All 106 optional fixture/reference/adversarial tests passed without skips in the
  separate Python 3.13 reference environment. This supplements the application
  environment and packaged tests; it does not replace them.
- Numerical dependency comparison: NumPy 1.24.3/SciPy 1.15.3 and the replacement
  NumPy 1.26.4/SciPy 1.17.1 produced identical full-precision results for three
  independent ECEF neighborhood/count checks, six overlap cases and six canonical
  clipping ledgers. Both nested JSON equality and SHA256 match. Eleven focused
  regressions also passed on each stack; see `corridor/scipy-compat/findings.md`
  in the local evidence directory.
- A full local source run passed 912 tests with 11 dependency-related skips.
  The optional fixture tests and real Mach-O tests were also run separately with
  their required dependencies. Later Mac lifecycle findings still require the
  final patched candidate to pass native CI; this local run alone is not release
  approval. The final patched native results are recorded below.

Detailed local evidence is retained under `.validation-output/release-audit/`,
including `repair/README.md`, `state/README.md`, `corridor/findings.md`, machine-
readable probe outputs, test logs and screenshots. This directory is not uploaded
as application data or bundled into the EXE.

### Customer mileage discrepancy

The local `Pipelines August 2023.kmz` contains 42 pipeline records, 44 LineString
paths and 2,139 vertices. Independent parsing confirms every longitude/latitude
vertex and path order; repair verification also preserves coordinate payloads.
The original SHA256 is unchanged. Independent GRS80 summation gives
920,747.8695152602 m, matching the application at **572.1250571784 US survey miles**.

Its 42 descriptive `MILES` fields sum to **648.601713**. These totals do not agree.
The serialized-geometry proof verifies that repair omitted no input line, but the
metadata's source geometry, unit convention and measurement method are unverified.
Do not present the field total as a verified reference or claim that it matches.
The difference is 76.4766558216 miles (11.791% below the attribute total), beyond
rounding. Client confirmation or a fresh authoritative export is needed before
using this particular file as an accepted system-mileage reference. Preserving
the supplied geometry does not certify completeness of the client's GIS system.

## Workflow and distribution controls

The actual baseline workflow run failed on both platforms:
[run 35364941214](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/35364941214).
It exposed the numerical/corridor issues above and GUI tests that assumed the
requested window size rather than the viewport actually granted by the runner.

A subsequent Windows run exposed a diagnostic-only native crash: a timed
`faulthandler` stack walk racing Shapely's native call. A standalone ten-point
polygon workload reproduced it in all three timed-dump runs; all three otherwise
identical runs without timed dumps passed. Windows retains the crash handler and
external subprocess/job deadlines while disabling the unsafe timed stack walk.
This does not bypass numerical or application tests.

A separate Windows smoke-test startup race was reproduced with the actual CTk
titlebar initialization and a busy event queue. The smoke now starts after the
root is mapped, retaining its 15-second deadline, visibility checks and 20 Summary
returns. This test-ordering fix is separate from the user-facing scrolling defect.

Sales confirmed M1–M4 machines running Sequoia/Tahoe and requested Golden Gate
support. Final CI builds the Apple Silicon app on macOS 15, then verifies that
same archived app on macOS 26 and 27; Intel is outside this rollout's scope.
The first signed candidate passed Sequoia and Tahoe but failed both entrypoints
on Golden Gate 27.0 (26A5406e). Direct inspection of SHA256-verified upstream
SciPy wheels confirmed the failing library metadata was present before packaging.
An isolated native [dependency probe](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/35375659551)
verified the replacement official NumPy 1.26.4/SciPy 1.17.1 stack. The suspected
SDK-dependent loader policy is an inference, not required to establish the fix;
the actual supported-OS runtime gates remain authoritative. See [SciPy's release
notes](https://docs.scipy.org/doc/scipy/release/1.17.0-notes.html) and the retained
`state/scipy-wheel-proof/README.md` evidence. The independent reference fixture
environment retains its own pinned dependencies and provenance.
The Golden Gate job uses the current `xcode-27` preview runner, asserts the actual
OS major version, and records its build number. See [Apple's release notice](https://support.apple.com/en-ca/149035)
and [the published GitHub image](https://github.com/actions/runner-images/releases).
The runner's preview status and exact OS build remain part of the receipt; a pass
on that image alone does not establish downloaded-app acceptance on every Golden
Gate release or future update. Include a current retail Golden Gate Mac in the
final downloaded/notarized-app review.

Workflow changes include explicit architecture labels, version assertions, an offline
smoke check after signing, app ZIPs preserving permissions/symlinks, explicit
release-token permissions, bounded job duration, and mandatory artifact presence.
Runner selection follows the [GitHub architecture table](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Archiving the app before upload addresses [documented artifact permission loss](https://github.com/actions/upload-artifact#permission-loss).

The TkinterDnD hook previously copied all platforms' native libraries into every
bundle. It now packages only the directory selected by the runtime loader,
retaining its Tcl files and shared data. Fourteen resource-selection tests pass.
After stamping version and minimum-OS metadata, the build restores and verifies
the app's ad-hoc signature before its first frozen check. The signed distribution
path subsequently replaces it with Developer ID signing and repeats the check.

The deployment-target environment variable did not establish macOS 13 support.
Direct inspection of official pyproj 3.7.2 wheel binaries found a minimum of
macOS 14 on ARM and 13 on Intel. A pre-sign gate now inventories **all** bundled
Mach-O libraries, requires the target architecture, and stamps the maximum actual
header requirement into `LSMinimumSystemVersion`. It retains a JSON inventory
with CI artifacts. Header requirements are necessary limits, not proof of older
OS runtime support; each supported OS must pass the packaged runtime gate. See
[PyInstaller compatibility guidance](https://pyinstaller.org/en/stable/usage.html#making-macos-apps-forward-compatible)
and [Apple's minimum-version key](https://developer.apple.com/library/archive/documentation/General/Reference/InfoPlistKeyReference/Articles/LaunchServicesKeys.html#//apple_ref/doc/uid/20001402-113253).

Tags now create **draft releases**. Mac notarization/stapling remains manual using
`scripts/macos/notarize_dmg.sh`; the signed app passing CI does not establish
Gatekeeper acceptance of a downloaded distribution. Notarize/staple the DMG,
replace its draft asset, and verify downloaded launch on target Macs before
publishing. No tag, main merge or sales release is created by this audit.

## Accuracy boundaries

- Original mileage is ellipsoidal path distance, not a stored client attribute or
  planar polygon length. Elevation is not included.
- Overlap savings remain an estimate based on fixed-length samples and qualifying
  continuous runs. Genuine short tails stay in original mileage. Grouping uses
  the existing conservative nearest-first pairwise-compatible heuristic; it is
  not an optimal flight-route solver.
- State attributed original mileage reconciles to combined original mileage with
  explicit coverage exceptions. State overlap qualification is independent, so
  state adjusted mileage need not sum to combined adjusted mileage. Shared-border
  allocation receives no state overlap discount.
- State ownership follows the bundled Census boundary model and its provenance;
  numerical precision is not surveyed ownership accuracy.
- Corridor maps are certified visualization buffers, not the source of mileage
  or savings. An unavailable map must be reported rather than substituted with
  unverified geometry.

## Pre-v5 verification receipt

Final application candidate: `8a8649654aab655ad965dd9b3018763886e444d2`, on the isolated
`codex/release-audit-verification-20260918` branch. All 64 application/build/test
snapshot files matched the working files at that checkpoint (including the bundled README), verified
by Git object hashes. The user's branch and index are unchanged.

[Native CI run 35375766969](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/35375766969)
passed every build/runtime job. The tag-only draft release job was correctly
skipped; no public release or tag was created.

| Native platform | Final evidence |
| --- | --- |
| Sequoia 15.7.9 (24G830), ARM64 | 901 source tests passed, 30 expected platform/optional-dependency skips, zero failures/errors. App built; all 261 native binaries passed architecture/minimum-OS inventory; both frozen entrypoints passed before and after Developer ID signing. DMG and app ZIP produced. |
| Tahoe 26.6.2 (25G83), ARM64 | The archived signed app passed strict signature/version/architecture checks and both frozen entrypoints, with PROJ networking OFF. |
| Golden Gate 27.0 (26A5406e), ARM64 preview runner | The identical archived signed app passed the same checks and both frozen entrypoints. This is actual macOS 27 execution, not only an Xcode 27 SDK check. |
| Windows x64 (Windows Server 2025 hosted runner) | 919 source tests passed, 14 expected platform/optional-dependency skips, zero failures/errors. EXE built and both frozen entrypoints passed. Final local Windows EXE also passed as detailed below. |

The Tahoe and Golden Gate jobs used the same app ZIP SHA256:
`f4630c8550bbdd69f67ee51c252811282a23d7c76c1b0393b38c059f21126222`.
CI embedded version: `4.39-dev.8a8649654aab`. The bundle records macOS 14.0.0
as its native binary minimum; supported-OS evidence is the three runs above.
Downloaded test/runtime/signing reports are retained in `final-ci-*.zip` under
the local evidence directory; all are also artifacts of the linked CI run.
`final-ci-receipt.json` records job conclusions, artifact IDs and checksums.
The platform suites overlap and their counts must not be added as unique tests.
Skips cover the other platform's native helpers/DPI probes and optional reference
dependencies, whose additional local verification is listed above.

The final local Windows build passed both frozen entrypoints with PROJ networking
disabled, including boundary resources, state reconciliation/map roundtrips,
repair provenance/saved-copy verification, curved/holed/multipart corridor exports,
drag-and-drop library loading, and 20 Summary returns per entrypoint without
callback errors.

- Local version: `4.27-dev.1562fa3d63c6.dirty`.
- EXE: `dist/Pipeline_Calculator_v4.27-dev.1562fa3d63c6.dirty.exe` and the identical
  convenient alias `dist/Pipeline_Calculator_v4.exe`.
- NumPy 1.26.4; SciPy 1.17.1.
- Size: 99,433,461 bytes.
- SHA256: `ac0f21b51ae0b4f2ab3b5ded2f0bb2c7929458696cc02a8180a52651195faecd`.
- Receipts: `.validation-output/release-audit/windows-build-receipt.json`,
  `source-snapshot-parity.json`, and `windows-packaged-smoke/`.

CI versions differ from the local preview because the verification branch has
additional snapshot commits; the matching source content and embedded version
checks establish which build was tested. Manual macOS notarization/stapling and
downloaded-app acceptance remain separate distribution requirements.

The standalone Windows developer setup helper was hardened separately after the
application snapshot: `scripts/windows/setup_windows.ps1` now requires Python
3.11, preserves incompatible environments, and checks every native installation,
import and dependency-consistency command. All 16 tests in
`tests/test_windows_setup.py` passed locally on Windows. These two files are not
part of the 64-file CI snapshot or the packaged app; neither hosted build invokes
this helper. README and all packaged application files matched that candidate at
the checkpoint. Historical root setup scripts are not the supported build path.

## Version 5 naming follow-up

The requested major-version change separates the clean application title (`5.0`)
from the complete build identity retained in run details, JSON and workbook
Analysis Details. Windows artifacts now use `Pipeline_Calculator_v5...exe`;
macOS uses `Pipeline_Calculator_v5.app` and versioned v5 DMG/app ZIP filenames.
The macOS bundle identifier and user preference locations are unchanged.

The first commit introducing this major on main's first-parent history is 5.0;
real-Git tests cover a normal merge after concurrent main changes, dirty previews,
release rejection and full embedded identity. Subsequent commits retain the
existing automatic minor-version progression. No main merge, tag or public
release has been created by this verification.

Local Windows verification completed:

- Both packaged GUI entrypoints passed, including a clean `5.0` title and full
  build identity in analysis results and the exported workbook.
- Full build: `5.0-dev.1562fa3d63c6.dirty`; 99,436,090 bytes.
- EXE: `dist/Pipeline_Calculator_v5.exe`, identical to
  `dist/Pipeline_Calculator_v5.0-dev.1562fa3d63c6.dirty.exe`.
- SHA256: `ecf906e34ce8100b8070c0f627421788898bd78083005fe7b2b83c7eba0623c5`.
- Receipts: `.validation-output/v5-windows-build-receipt.json`,
  `v5-packaged-smoke/`, and `v5-source-parity.json`.

The v5 native CI snapshot is `deace1f03e0050607ad1a8af64b7496bdb8912a9`.
All 77 explicitly included application/build/test files matched the local source
by Git object hash. This includes the Windows setup helper and its tests.
Documentation-only audit/signing notes updated afterward do not affect packages.
All build/runtime jobs passed in
[run 35383162511](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/35383162511).
The release job was correctly skipped. CI embedded version is
`5.0-dev.deace1f03e00`; its clean display version is `5.0`.

| Platform | v5 verification |
| --- | --- |
| Windows x64 | 939 source tests passed, 14 expected skips; EXE built and both packaged GUI entrypoints passed. |
| Sequoia ARM64 build runner | 905 source tests passed, 46 expected skips; all 261 native binaries checked; both packaged entrypoints passed before and after Developer ID signing. |
| Tahoe 26.6.2 (25G83), ARM64 | The distributed v5 app passed version/signature/architecture checks and both packaged entrypoints. |
| Golden Gate 27.0 (26A5406e), ARM64 preview runner | The identical v5 app passed the same checks and both packaged entrypoints. |

The added macOS skips include the 16 Windows-only setup-helper tests. Test counts
overlap between platforms and are not unique-test totals. The newer macOS jobs
verified app ZIP SHA256
`e54bbffc1fb0675824bb67432bfe57864616facff38f17257a6d6ad11a32178f`,
matching the downloaded local artifact.

Signed macOS outputs are also available locally:

- `dist/Pipeline_Calculator_v5.0-dev.deace1f03e00_arm64.dmg`
- `dist/Pipeline_Calculator_v5.0-dev.deace1f03e00_arm64.app.zip`

The ZIP contains `Pipeline_Calculator_v5.app`; keeping the archive intact on
Windows preserves macOS permissions and symlinks. Both downloaded outer artifact
hashes were checked against GitHub's recorded SHA256. The DMG SHA256 is
`cf47846d78d5c89b4be973d978eb822b493c8982e6fd4df03afe7e5945e4d716`.
These previews are signed but have not been notarized; the existing separate
notarization/stapling step is still required for distribution.

Evidence is in `.validation-output/v5-ci-receipt.json`,
`v5-macos-build-receipt.json`, and the `v5-ci-*.zip` test/runtime reports.
Previous local v4 binaries remain available for rollback.
