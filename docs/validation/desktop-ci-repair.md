# Desktop CI repair — September 11, 2026

Status: **complete and verified on GitHub Actions**, on branch
`codex/fix-desktop-ci`; not merged to main.

## Verified causes and changes

- Native Tk tests previously shared an interpreter with worker-thread tests. The
  macOS suite hung for an hour; a local full-suite run also timed out in warning
  controls with a stale Tk callback. Native GUI scenarios now run in separate
  processes with a 45-second timeout and 20-second thread dumps. They still run
  on macOS; this does not skip failing GUI coverage. Windows-only checks retain
  their explicit platform skips. The isolated macOS suite completed in 15 seconds.
- macOS returned `9.909715631392153e-11` meters for the zero-length fixture.
  Source mileage and segmentation now bypass the geodesic backend for exactly
  identical endpoints. No tolerance rounds away real pipeline lengths. A regression
  test rejects any backend call for identical endpoints, including tiny segment sizes.
- The Windows runner has a 1024x768 display. At 250% scaling, the overlap table
  received only 56 physical pixels. Redundant vertical margins around its table
  and navigation are removed. An additional locally constrained 1280x720 physical
  window reproduces the failure before the fix and passes afterward.
- Once the overlap check passed, CI exposed a collapsed settings viewport when
  returning to import. The footer is now reserved before the expanding drop zone,
  with a height budget and compact header on short windows. Regression probes
  constrain physical windows to 1020x720 and 1280x720 at 250% scaling.
- App shutdown cancels pending Tk callbacks before destroying the window.
- CI now performs a bounded real Tk initialization preflight, prints Python/Tcl/Tk
  locations and screen dimensions, emits individual test results and stack dumps,
  and saves JUnit reports. Distribution uploads require successful preceding
  steps; missing executables no longer obscure the original test failure.

## Validation

- Local full suite on `8b7b021`: **244 passed in 62.32 seconds**.
- [Diagnostic run](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34641345171)
  proved the macOS hang was removed and exposed the identical-endpoint assertion.
- [macOS packaging verification](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34641553010):
  macOS passed tests, built and signed the app, and uploaded the app and DMG.
  Windows still failed its layout check in this intermediate run; the spacing fix
  is subsequent to it.
- [Final application build verification](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34642239763)
  on `8b7b021`: **Windows 244 passed**, EXE uploaded; **macOS 229 passed,
  15 platform-specific skips**, app signed and verified with codesign, app and DMG
  uploaded. Both build jobs succeeded. Release creation was intentionally skipped.
- JUnit reports were initially placed under `build`, which packaging clears.
  Their destination is now `.validation-output/ci-test-results.xml`, outside
  packaging cleanup. The upload action includes only that explicitly selected
  hidden-path file.
- [Final CI/report verification](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34642656453)
  on `9daab29`: both jobs succeeded; Windows EXE, signed macOS app/DMG, and both
  platform JUnit reports are available as artifacts. Windows: **244 passed**;
  macOS: **229 passed, 15 platform-specific skips**. No tests were disabled to
  obtain a passing build. Later documentation-only commits do not change tested code.

## Remaining acceptance

Real-project accuracy review and interactive macOS/DPI acceptance remain in the
follow-up runbook. Successful CI packaging is not evidence of interactive launch,
Gatekeeper/notarization acceptance, or field-data validation. No release or main
branch merge is performed by this repair task.
