#!/usr/bin/env bash
# Verify the SAME built app on another native OS; do not build or install app dependencies.
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 APP_ZIP EXPECTED_OS_MAJOR EXPECTED_VERSION OUTPUT_DIRECTORY" >&2
  exit 2
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$1"
EXPECTED_OS="$2"
EXPECTED_VERSION="$3"
OUTPUT="$4"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="runner validation"
mkdir -p "${OUTPUT}"

finish() {
  status="$1"
  trap - EXIT
  if [[ -f "${OUTPUT}/runtime.json" ]]; then
    "${PYTHON_BIN}" - "${OUTPUT}/runtime.json" "${status}" "${STAGE}" <<'PY' || true
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text())
value.update(status='passed' if sys.argv[2] == '0' else 'failed',
             exit_code=int(sys.argv[2]), last_stage=sys.argv[3])
path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')
PY
  fi
  exit "${status}"
}
trap 'finish "$?"' EXIT

OS_VERSION="$(sw_vers -productVersion)"
OS_BUILD="$(sw_vers -buildVersion)"
"${PYTHON_BIN}" - "${ARCHIVE}" "${EXPECTED_OS}" "${EXPECTED_VERSION}" "${OUTPUT}" "${OS_VERSION}" "${OS_BUILD}" <<'PY'
import hashlib, json, os, platform, stat, sys, zipfile
from pathlib import Path, PurePosixPath
archive, expected_os, expected_version, output, actual_os, build = sys.argv[1:]
archive, output = Path(archive).resolve(strict=True), Path(output).resolve(strict=True)
machine = platform.machine()
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
manifest = dict(schema_version=1, archive=str(archive), archive_sha256=digest,
                expected_os_major=expected_os, actual_os_version=actual_os,
                actual_os_build=build, architecture=machine,
                expected_version=expected_version, status='running')
(output/'runtime.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
if machine != 'arm64':
    raise SystemExit(f'Expected native arm64 execution, found {machine}')
if not expected_os.isdecimal() or actual_os.split('.')[0] != expected_os:
    raise SystemExit(f'Expected macOS {expected_os}, but actual runtime is {actual_os} ({build})')
if not expected_version.strip():
    raise SystemExit('Expected application version must not be empty')
# Fail before extraction on absolute/traversal entries or entries below a
# symlink. An output from an earlier run must never satisfy this run's gate.
with zipfile.ZipFile(archive) as source:
    names, links = [], set()
    for item in source.infolist():
        path = PurePosixPath(item.filename)
        if ('\\' in item.filename or path.is_absolute() or '..' in path.parts
                or not path.parts or path.parts[0] not in ('Pipeline_Calculator_v5.app', '__MACOSX')):
            raise SystemExit(f'Unexpected archive path: {item.filename}')
        names.append(path)
        if stat.S_ISLNK(item.external_attr >> 16):
            links.add(path)
    if any(parent in links for path in names for parent in path.parents):
        raise SystemExit('Archive contains entries beneath a symbolic link')
(output/'app').mkdir()
PY

STAGE="archive extraction"
ditto -x -k "${ARCHIVE}" "${OUTPUT}/app"
APP_PATH="${OUTPUT}/app/Pipeline_Calculator_v5.app"
STAGE="extracted app validation"
"${PYTHON_BIN}" - "${APP_PATH}" "${EXPECTED_VERSION}" "${OUTPUT}/runtime.json" <<'PY'
import json, plistlib, sys
from pathlib import Path
app, expected, report = Path(sys.argv[1]).resolve(strict=True), sys.argv[2], Path(sys.argv[3])
root = report.parent.resolve()/ 'app'
if not app.is_relative_to(root):
    raise SystemExit('Extracted app escapes the isolated output directory')
for path in app.rglob('*'):
    if path.is_symlink() and not path.resolve(strict=True).is_relative_to(app):
        raise SystemExit(f'App contains an external symbolic link: {path}')
with (app/'Contents/Info.plist').open('rb') as stream:
    info = plistlib.load(stream)
actual = info.get('PipelineCalculatorVersion')
if actual != expected:
    raise SystemExit(f'Embedded version mismatch: expected {expected}, got {actual}')
value = json.loads(report.read_text())
value.update(app=str(app), embedded_version=actual,
             minimum_macos=info.get('LSMinimumSystemVersion'))
report.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')
PY

STAGE="signature verification"
codesign --verify --deep --strict "${APP_PATH}"
STAGE="frozen application smoke"
"${PYTHON_BIN}" "${REPO_DIR}/scripts/validation/check_packaged_smoke.py" "${APP_PATH}" \
  --expected-version "${EXPECTED_VERSION}" --output-directory "${OUTPUT}/smoke"
STAGE="complete"
