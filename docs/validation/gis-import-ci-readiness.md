# GIS import launcher and CI verification

Checked November 16, 2026. The changes remain local and uncommitted on
`codex/fix-desktop-ci`. Machine-readable receipts are in
[gis-import-ci-readiness.json](gis-import-ci-readiness.json).

## Bash launcher

`calc` in `C:/Users/rbake/.bashrc:5` already invokes this checkout's
`.venv/Scripts/python.exe` and `src/pipeline_calculator_entry.py`.
`.bash_profile` loads `.bashrc`. It therefore launches the current source,
including GIS updates, without depending on a versioned executable filename.
No alias edit was needed.

A fresh Git Bash login ran `calc --smoke-test` with projection networking off.
It reported `5.9-dev.7d6e68314148.dirty`, the modern interface, and passing SHP,
direct ZIP, GPKG layer selection, GeoJSON, projected CRS, and source-snapshot
checks. The two harmless job-control notices came from invoking an interactive
shell without a terminal; the application exited successfully.
After the portability fixes, another fresh login launched `calc` successfully
from `C:/Users/rbake`, confirming it does not depend on the repository being the
current directory. All six GIS smoke checks passed again.

## GitHub baseline and workflow updates

The latest published [run 37058844990](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/37058844990)
passed Windows, macOS 15, signed macOS packaging, and the distributed app on
macOS 26 and 27. It tested commit
`7d6e68314148f659798e7ffb75f287b3f2a6f7fe` on October 2, 2026 and **does not
include the new local GIS code**. The tag-only release job was skipped.

The current [workflow](../../.github/workflows/build.yaml) was checked against
upstream metadata and moved to these Node 24 action releases:

| Action | Verified release |
| --- | --- |
| Checkout | [7.0.1](https://github.com/actions/checkout/blob/v7.0.1/action.yml) |
| Python setup | [7.1.0](https://github.com/actions/setup-python/blob/v7.1.0/action.yml) |
| Artifact upload | [7.0.2](https://github.com/actions/upload-artifact/blob/v7.0.2/action.yml) |
| Artifact download | [8.0.2](https://github.com/actions/download-artifact/blob/v8.0.2/action.yml) |
| Release creation | [3.0.4](https://github.com/softprops/action-gh-release/blob/v3.0.4/action.yml) |

Actionlint rejected the old release action's obsolete runtime. The other old
actions declared Node 20; GitHub now runs JavaScript actions with Node 24.
The replacement releases explicitly support that runtime.
[GitHub runtime announcement](https://github.blog/changelog/2026-09-23-node-20-is-no-longer-available-in-github-actions/).

Artifact names, ZIP nesting, permissions, signing gates, and release conditions
are preserved. Direct-file upload mode remains disabled; the updated download
action fails on digest mismatches. `xcode-27` is an official GitHub-hosted ARM64
label, although Actionlint 1.7.12's built-in table lacks it. The local lint
configuration adds that exact recognized label without changing scheduling.
[Runner catalog](https://github.com/actions/runner-images#available-images).

## Portability fixes and local verification

- GPKG's SQLite audit now handles Python builds that omit the optional
  `enable_load_extension` API; when available, extension loading is explicitly
  disabled. [Python documentation](https://docs.python.org/3.11/library/sqlite3.html#sqlite3.Connection.enable_load_extension).
- Loose SHP sidecars normalize Unicode names before matching and still reject
  ambiguous equivalent names. Four new portability cases and an existing exact
  attribute regression passed.
- All ten macOS shell scripts passed Bash syntax and ShellCheck 0.11.0. The five
  Windows/CI PowerShell scripts passed the PowerShell parser. Actionlint 1.7.12
  with ShellCheck passed the complete workflow.
- The existing macOS bundle, distribution, runtime-helper, and drag/drop-hook
  suites passed **47 tests** on Windows with the optional Mach-O parser enabled.
  A new GIS wheel-layout integration test plus the bundle suite passed **20
  tests**; these groups overlap and must not be added together.
- The actual Pyogrio 0.13.0 CPython 3.11 ARM64 wheel contains six ARM64 binaries
  requiring macOS 12 or later. Its GDAL library has only system dependencies;
  the hook includes that library plus GDAL/PROJ data. The Pyproj 3.7.2 ARM64 wheel
  targets macOS 14, within the macOS 15 build target. Inspection checks binary
  structure and packaging, not native execution.
- Cross-platform pip resolution found CPython 3.11/macOS 15 ARM64 wheels for the
  requirements. Pip's conditional dependency markers still reflect the Windows
  host, so this is wheel-availability evidence, not a native macOS installation.

## Fresh Windows dependency environment

The existing local environment used older unpinned packages than a new CI
runner would install. A separate ignored virtual environment installed the
actual requirements afresh without replacing the user's working environment.
Only the optional `macholib` inspection parser was added to the existing `.venv`;
its runtime and build package versions were retained. The fresh environment selected
Pandas 3.0.6, Shapely 2.2.0, PyInstaller 6.22.3, hooks-contrib 2026.10, Pillow
13.0.0, and pytest 9.2.0; the pinned GIS/numerical versions stayed unchanged.

- Initial focused imports, exports, build helpers, and both source entrypoints:
  **11 passed** in 57.96 seconds.
- Complete non-native suite: **1,099 passed, 15 skipped, 108 deselected**, 374.51
  seconds. Five skips were optional geography-audit modules; ten were Mach-O
  inspection cases whose optional parser is absent from a normal Windows install.
- After installing only the optional `macholib` parser, the macOS GIS bundle and
  general bundle tests passed **20/20** with PyInstaller 6.22.3. This includes the
  ten previously skipped inspection cases; ten other cases overlap the full run.
- Complete native GUI suite: **108 passed, 5 skipped, 1,109 deselected**, 452.18
  seconds. Combining testcase identities across the non-native, native, and
  optional Mach-O runs gives **1,217 distinct passing tests and five distinct
  remaining skips**, all optional geography-audit modules. This is a combined
  result, not a single monolithic pytest invocation.
- Four deprecation warnings came from the existing boundary-data preparation
  script's use of `shapely.ops.transform`; no warnings were suppressed.
- The Windows production build script succeeded using the fresh interpreter,
  with build outputs isolated under `.validation-output/gis-ci-clean-windows-build`.
  A recorded harness changed only its repository/interpreter assignments; all
  PyInstaller arguments and build checks were unchanged. Both frozen interfaces
  passed the standard offline smoke, with expected version enforced. The
  121,633,956-byte artifact's SHA-256 is
  `4cd8c8c96fbf3e12e44ec562e140171cab6b73c3d631943a30a1ce96f4373fca`.

The fresh environment's complete dependency receipt, test counts, build harness
hashes, artifact hash, and both full frozen-smoke reports are preserved in the
[machine-readable evidence](gis-import-ci-readiness.json).

## Remaining platform proof

These local checks cannot establish native macOS launch, signatures, or package
behavior for the new code. A commit containing the GIS changes and fixtures
must reach GitHub to run the existing macOS 15/26/27 and Windows jobs against it.
No older successful run is used as a substitute, and no new remote run or
published release is claimed here.

GitHub repository metadata confirms this repository is **public**. Pushing the
byte-identical customer ZIP fixtures would publish their contents. That external
publication requires confirmation before the current changes can be sent to the
existing PR with those fixtures included.
