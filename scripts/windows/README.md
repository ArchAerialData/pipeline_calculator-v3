# Windows scripts

These scripts are for engineers building/running Pipeline Calculator v5 from source on Windows, and for producing a self-contained `.exe` for distribution.

## Development (from source)
- Setup dev machine: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\setup_windows.ps1`
- Run GUI: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\run_gui.ps1`
- Basic checks: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\run_tests.ps1`

These scripts use a repo-local virtualenv at `.venv\\`.

## GUI tests without desktop interruptions

Normal `python -m pytest` and `run_tests.ps1` runs launch Windows GUI checks on
an undisplayed Windows desktop. The real Tk windows still open, resize, map,
and run their event loops there. Existing layout, DPI, dialog, and normal-startup
visibility assertions remain enabled. This also covers both source smoke entrypoints.
The launcher never switches your desktop, suppresses application focus behavior,
or withdraws windows to bypass visibility checks. Isolation failures fail the test;
they never fall back to visible execution. Timeout cleanup terminates the test's
process tree, including venv launcher children.

Standalone `tests/ui_layout_probe.py`, `tests/ui_startup_probe.py`, the CI Tcl/Tk
check, and `scripts/validation/check_windows_startup.py` use the same isolation
by default. Window-specific layout captures are still supported. To wrap another
GUI validation command (including a built EXE):

```powershell
.venv/Scripts/python.exe scripts/validation/gui_process.py --timeout 60 -- C:/path/to/app.exe --smoke-test C:/path/to/smoke.json
```

For deliberate on-screen visual review only, temporarily set
`$env:PIPELINE_GUI_TEST_MODE = 'interactive'`; remove it afterward with
`Remove-Item Env:PIPELINE_GUI_TEST_MODE`. That mode can interrupt the desktop.
Actual monitor dragging, external viewer interaction, and desktop-compositor
appearance still need a deliberate interactive review or a dedicated Windows VM;
the isolated tests do not claim to replace those checks. Other operating systems
retain their existing GUI test behavior.

Implementation references: Microsoft's
[desktop assignment](https://learn.microsoft.com/en-us/windows/win32/winstation/thread-connection-to-a-desktop),
[STARTUPINFO.lpDesktop](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/ns-processthreadsapi-startupinfow),
and [job object cleanup](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

## Distribution build (no Python needed on user machines)

### Corridor launch configuration

`src/pipeline_calculator/gui/config.py` contains the backend-only
`SHOW_CORRIDOR_LAUNCH_DIALOG` switch, defaulting to `False`. View Corridor
creates and opens its KML directly without the preparation/status modal.
Actual launch failures still display an error and the saved KML path when available.
Set the switch to `True` and rebuild to restore the existing dialog, including
its retry and save-copy actions. Neither GUI exposes this switch.

Build a self-contained `.exe` with PyInstaller:
- `powershell -ExecutionPolicy Bypass -File scripts\\windows\\build_exe.ps1`

To build the legacy monolithic GUI instead of the default modular v5 GUI, set:

```powershell
$env:PIPELINE_CALCULATOR_BUILD_IMPL = "legacy"
powershell -ExecutionPolicy Bypass -File scripts\\windows\\build_exe.ps1
```
# Isolated preview validation

Use `./scripts/windows/build_exe.ps1 -OutputRoot C:\path\to\preview` from the repo
root to build the current source into separate `build`/`dist` directories. Existing
repository artifacts are untouched. The selected output's build/dist directories
are cleaned; choose a dedicated directory. Both output paths are checked before
cleanup. Build failures propagate as nonzero exit codes.

Set `PIPELINE_CALCULATOR_BUILD_IMPL=legacy` for the compatibility entrypoint;
the default is modular. Test a generated EXE with `--smoke-test OUTPUT.json` to
exercise frozen imports, embedded resources, a synthetic calculation, and the live
results transition under the main event loop. Use the isolated desktop launcher
described above and impose a test-process timeout. To validate a specific dataset,
set `PIPELINE_SMOKE_INPUT` to its KML/KMZ path in the test process environment; this
setting is read only in explicit smoke mode. Corridor actions are checked without
opening an external application.
This smoke check does not replace native visual review or authorize publication.
