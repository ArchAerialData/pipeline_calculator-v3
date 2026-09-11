# Desktop CI repair — September 11, 2026

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
- App shutdown cancels pending Tk callbacks before destroying the window.
- CI now performs a bounded real Tk initialization preflight, prints Python/Tcl/Tk
  locations and screen dimensions, emits individual test results and stack dumps,
  and saves JUnit reports. Distribution uploads require successful preceding
  steps; missing executables no longer obscure the original test failure.

## Validation

- Local full suite after calculation/layout/shutdown fixes: **240 passed**.
- [Diagnostic run](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34641345171)
  proved the macOS hang was removed and exposed the identical-endpoint assertion.
- [macOS packaging verification](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34641553010):
  macOS passed tests, built and signed the app, and uploaded the app and DMG.
  Windows still failed its layout check in this intermediate run; the spacing fix
  is subsequent to it.
- Final cross-platform build verification: pending the next run.

## Remaining acceptance

Real-project accuracy review and interactive macOS/DPI acceptance remain in the
follow-up runbook. Successful CI packaging is not evidence of interactive launch,
Gatekeeper/notarization acceptance, or field-data validation. No release or main
branch merge is performed by this repair task.
