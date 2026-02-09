# Windows scripts

These scripts are for engineers building/running Pipeline Calculator v3 from source on Windows, and for producing a self-contained `.exe` for distribution.

## Development (from source)
- Setup dev machine: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\setup_windows.ps1`
- Run GUI: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\run_gui.ps1`
- Basic checks: `powershell -ExecutionPolicy Bypass -File scripts\\windows\\run_tests.ps1`

These scripts use a repo-local virtualenv at `.venv\\`.

## Distribution build (no Python needed on user machines)

Build a self-contained `.exe` with PyInstaller:
- `powershell -ExecutionPolicy Bypass -File scripts\\windows\\build_exe.ps1`

