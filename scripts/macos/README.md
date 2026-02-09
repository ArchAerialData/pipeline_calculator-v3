# macOS scripts

These scripts are for engineers building/running Pipeline Calculator v3 from source on macOS, and for producing a self-contained `.app` for distribution.

## Development (from source)
- Setup dev machine: `bash scripts/macos/setup_macos.sh`
- Run GUI: `bash scripts/macos/run_gui.sh`
- Basic checks: `bash scripts/macos/run_tests.sh`

These scripts use a repo-local virtualenv at `.venv/`.

## Distribution build (no Python needed on pilot machines)

Build a self-contained `.app` with PyInstaller:
- `bash scripts/macos/build_app.sh`

Then package a DMG:
- `bash scripts/macos/package_dmg.sh`

Note: CI produces **signed but not notarized** DMGs (matches the reference repo behavior). Gatekeeper prompts are expected outside the Mac App Store unless you notarize.

