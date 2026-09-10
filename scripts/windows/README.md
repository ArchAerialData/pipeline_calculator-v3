# Windows scripts

These scripts are for engineers building/running Pipeline Calculator v4 from source on Windows, and for producing a self-contained `.exe` for distribution.

## Development (from source)
- Setup dev machine: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\setup_windows.ps1`
- Run GUI: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\run_gui.ps1`
- Basic checks: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\run_tests.ps1`

These scripts use a repo-local virtualenv at `.venv\\`.

## Distribution build (no Python needed on user machines)

Build a self-contained `.exe` with PyInstaller:
- `powershell -ExecutionPolicy Bypass -File scripts\\windows\\build_exe.ps1`

To build the legacy monolithic GUI instead of the default modular v4 GUI, set:

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
exercise frozen imports, embedded resources, a withdrawn widget tree and a synthetic
calculation without opening an external application. Impose a test-process timeout.
This smoke check does not replace native visual review or authorize publication.
