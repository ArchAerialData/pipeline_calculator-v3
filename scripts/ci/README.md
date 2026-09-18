# GitHub Actions CI (macOS)

This folder contains CI scripts used by the GitHub Actions workflow. The workflow file itself must live under `.github/workflows/` (GitHub requirement), but build/sign/package logic is kept here to reduce workflow clutter.

## Scripts
- `macos_build.sh` — creates `.venv`, installs deps, runs tests, builds the `.app`, and verifies both implementations in the frozen application.
- `windows_build.ps1` — installs dependencies, runs tests, builds the versioned executable, and verifies both implementations in the frozen application.
- `macos_package.sh` — packages the `.app` into a `.dmg`.
- `macos_sign_and_package.sh` — signs the `.app` and creates a DMG containing the signed app (no notarization; only runs if secrets are provided).

## Build impl selection
CI defaults to building the modular (refactored) GUI by setting:
- `PIPELINE_CALCULATOR_BUILD_IMPL=new`

Set `PIPELINE_CALCULATOR_BUILD_IMPL=legacy` to build the original monolithic GUI instead.

The CI frozen smoke gate requires the modular distribution, which contains both
implementations and supports switching with `PIPELINE_CALCULATOR_IMPL`. A legacy-only
build cannot pass its modular implementation check.

## Packaged smoke gate

After building, both CI scripts run `scripts/validation/check_packaged_smoke.py`.
It launches the actual packaged executable twice, selecting `new` and `legacy`,
with `PROJ_NETWORK=OFF`. Windows uses an isolated desktop; macOS uses its normal
subprocess launch. The gate checks the frozen flag, 51 bundled jurisdictions,
the Texas/Oklahoma crossing, mileage reconciliation, and combined/state KMZ
roundtrips. Failure or timeout prevents distribution artifact upload.

Reports, captured output, and the gate summary are saved in
`.validation-output/packaged-smoke/` and uploaded alongside the test reports even
when a smoke run fails. Re-run locally with:

```bash
python scripts/validation/check_packaged_smoke.py dist/Pipeline_Calculator_v5.app
```

On Windows, pass the exact `dist/Pipeline_Calculator_v<version>.exe` path instead.

## Secrets used by CI (optional for code signing)
- `MACOS_CERT_P12` (base64 of Developer ID Application `.p12`)
- `MACOS_CERT_PASSWORD`

### Setting secrets with GitHub CLI

If you have `gh` installed and authenticated, you can set secrets from the gitignored `apple_dev/` folder:

```bash
bash scripts/ci/set_github_secrets.sh owner/repo
```

### Generating `MACOS_CERT_P12` (base64)

On macOS/Linux:

```bash
python3 - <<'PY'
import base64, pathlib
p = pathlib.Path("PATH/TO/DeveloperID.p12")
print(base64.b64encode(p.read_bytes()).decode("ascii"))
PY
```

(GitHub secrets can store multi-line values, but CI expects a single base64 blob; the Python version above prints a single line.)

### Optional notarization keys

Notarization is **not** enabled in this repo (mirrors the reference repo setup), but if you later enable it you’ll likely need:
- App Store Connect API key `.p8` (base64)
- API Key ID (looks like `ABCD123456`)
- Issuer ID (UUID)

## Notes
- Unsigned DMG artifacts are allowed only for non-`main` and non-tag builds (PR/debug builds).
- This repo is configured to produce **signed but not notarized** DMGs (mirrors the reference repo setup). Gatekeeper will typically require a manual override on first launch for apps distributed outside the Mac App Store.
