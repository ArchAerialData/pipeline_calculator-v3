# Pipeline KMZ regression suite

**Four main KMZs and three variations are reference-verified.** They contain real synthetic
pipeline inputs, an offline deterministic generator, two independent references, expected
results, interval/overlap/crossing ledgers, preview maps and executable intent checks.
Application comparisons are recorded separately from the independent references.
Latest comparison against the fingerprinted application sources: **10,250 checks, 0 mismatches**, including exports.

## Run

Run these commands from the repository root with CPython 3.13.3:

```powershell
python -m pip install -r tests/fixtures/geography/pipeline_kmz_regression_suite/requirements.txt
# Regenerate all core archives, previews and independent expected results:
python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py regenerate
# Independently reconstruct and check every core archive and ledger:
python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py validate
# Compare fixed expectations with the application and exercise KMZ exports:
python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py compare
# Run adversarial tests, fresh references, exports and known-defect classification:
python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py audit
# Generate and measure the optional larger branching-overlap profile:
python tests/fixtures/geography/pipeline_kmz_regression_suite/validation/run_stress.py
# Remeasure the existing stress case without replacing its frozen expectation:
python tests/fixtures/geography/pipeline_kmz_regression_suite/validation/run_stress.py --validate-existing
```

Dependency installation is the only step that may require a network. With the pinned packages
and bundled boundary resource supplied, all commands operate offline. Use a separate virtual
environment if the application needs different package pins. Regeneration permits unrelated
later commits but rejects contract source changes, including uncommitted changes, until reviewed.
It stages all core outputs, validates them, then publishes with rollback on ordinary I/O errors.
Validation and application comparison can run against later application revisions.
Missing inputs, empty selections, nonfinite values and stale evidence are errors. Focused
`--only` runs use separate reports. Saved reports fingerprint assets and validation code.

Ordinary repository test discovery reports skips if optional audit dependencies are absent.
The dedicated audit command rejects skipped tests or missing collection; install the suite
requirements before using its receipt as evidence.

Baseline: `71da499d5756648ae395660f0a241ea00edbea4f`. Generator: `1.0.0`; seed: `71499`.
Boundary SHA-256: `55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539`.
The generator and references each verify the bundled archive and its individual WKB hashes.
The [complete boundary manifest](validation/boundary_manifest.json) preserves Census attribution,
2025 vintage, prior transformation operations, accuracy qualifications and original vertex model.
No new transformation or simplified boundary is substituted.

## Results

Displayed values are rounded for reading. JSON/CSV values retain numerical precision.
State original below means **attributed original**, including its equal shared allocation.
Pairwise qualifying coverage is separate from savings. State adjusted totals need not sum
to Combined adjusted mileage because state sampling and qualification restart on clipped interiors.

### 01_three_state_disconnected_networks

[KMZ](fixtures/01_three_state_disconnected_networks.kmz) · [Expected JSON](expected/01_three_state_disconnected_networks.expected.json) · [Preview](previews/01_three_state_disconnected_networks.png)

48 sources · 54 paths · 753 vertices · 0 crossing events · 0 endpoint touches.

| Scope | Original m | US survey mi | Pairwise qualifying m | Savings m | Adjusted m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined | 46,211.085106 | 28.714180 | 15,300 | 10,800 | 35,411.085106 |
| LA | 15,403.695035 | 9.571393 | 5,100 | 3,600 | 11,803.695035 |
| TX | 15,403.695035 | 9.571393 | 5,100 | 3,600 | 11,803.695035 |
| WY | 15,403.695036 | 9.571393 | 5,100 | 3,600 | 11,803.695036 |

### 02_three_state_transverse_crossings

[KMZ](fixtures/02_three_state_transverse_crossings.kmz) · [Expected JSON](expected/02_three_state_transverse_crossings.expected.json) · [Preview](previews/02_three_state_transverse_crossings.png)

20 sources · 20 paths · 292 vertices · 9 crossing events · 1 endpoint touches.

| Scope | Original m | US survey mi | Pairwise qualifying m | Savings m | Adjusted m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined | 50,359.470315 | 31.291862 | 3,300 | 3,300 | 47,059.470315 |
| NM | 13,796.654468 | 8.572826 | 1,100 | 1,100 | 12,696.654468 |
| OK | 13,456.065855 | 8.361195 | 1,100 | 1,100 | 12,356.065855 |
| TX | 23,106.749993 | 14.357840 | 1,100 | 1,100 | 22,006.749993 |

Transverse crossing angles: 89.772207–89.974853 degrees; each event is in the crossing ledger.

### 03_parallel_corridors_crossing_borders

[KMZ](fixtures/03_parallel_corridors_crossing_borders.kmz) · [Expected JSON](expected/03_parallel_corridors_crossing_borders.expected.json) · [Preview](previews/03_parallel_corridors_crossing_borders.png)

20 sources · 20 paths · 481 vertices · 13 crossing events · 0 endpoint touches.

| Scope | Original m | US survey mi | Pairwise qualifying m | Savings m | Adjusted m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined | 29,873.952865 | 18.562777 | 9,775 | 8,930 | 20,943.952865 |
| NM | 10,105.783665 | 6.279430 | 2,535 | 2,535 | 7,570.783665 |
| OK | 5,097.138120 | 3.167208 | 900 | 895 | 4,202.138120 |
| TX | 14,671.031079 | 9.116138 | 4,900 | 4,195 | 10,476.031079 |

Transverse crossing angles: 89.773014–89.975667 degrees; each event is in the crossing ledger.

### 04_shared_border_and_near_border

[KMZ](fixtures/04_shared_border_and_near_border.kmz) · [Expected JSON](expected/04_shared_border_and_near_border.expected.json) · [Preview](previews/04_shared_border_and_near_border.png)

11 sources · 11 paths · 28 vertices · 2 crossing events · 1 endpoint touches.

| Scope | Original m | US survey mi | Pairwise qualifying m | Savings m | Adjusted m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined | 3,771.350000 | 2.343404 | 5,700 | 1,865 | 1,906.350000 |
| NM | 1,676.665000 | 1.041829 | 425 | 425 | 1,251.665000 |
| TX | 2,094.685000 | 1.301574 | 425 | 425 | 1,669.685000 |

Transverse crossing angles: 90.000000–90.000000 degrees; each event is in the crossing ledger.

## Controls and interpretation

- **01:** TX/LA/WY each has positive pair, compatible-trio and incompatible-outside-pair
  controls; perpendicular crossings, diverging branches, loops and isolated lines; two
  disconnected 123.25 m design runs per multipart source. Actual samples qualify separately.
  Full-coverage source groups are verified with explicit sets; perfectly aligned diagonal
  eligibility is asserted only in the hand-worked controls. State savings are 3,600 m each
  and add to Combined because the original independent path inputs are preserved.
- **02:** Continuous NM/TX/OK visits, sparse-edge crossings, reentry, a genuine short state
  visit and an exact native-vertex endpoint touch. Three interior pairs supply positive savings.
- **03:** Long/short/asymmetric border splits, a joining trio, distinct diverge/rejoin
  sections, an opposite-state pair, and nonaligned starts/tails/opposite digitization.
  The short split saves 300 m Combined and zero in either state. The opposite-state pair
  saves 705 m Combined and zero in its separate state inputs. Connectors and detached
  feeder controls are included in full accounting and the global pair search.
- **04:** Verified TX/NM meridian sharing; shared intervals above and below the overlap
  minimum; a partner that must not use shared mileage to qualify state savings; 3 cm
  exclusive offsets; a 10 cm crossing; endpoint touch; reversal and collinear-vertex variants.
  Shared physical length is approximately 1,133.25 m, stored once and allocated equally.

The seven files execute 383 numerical intent assertions plus nine hand-worked controls.
The [coverage matrix](validation/coverage_matrix.md) links each assertion to source keys
and measured values. Repeated display names and OBJECTIDs never merge source identities.
The source-order variant preserves metadata by stable key. Reverse/extra-vertex variants
are measured independently; their unchanged 04 savings are an observed phase-safe result,
not a general reversal invariant. Renaming is not asserted invariant.

## Evidence, uncertainty and limitations

The [geometric reference](reference/README.md) parses final XML directly, measures GRS80
geodesics with GeographicLib, crosschecks each original edge using pyproj/PROJ, and solves
native coordinate-linear boundary equations with curvature bounds and root refinement.
The sampled reference uses cumulative chainage interpolation, exhaustive bounded arrays,
explicit coverage sets and disjoint clique grouping. Neither imports application code.

Per-source and whole-fixture conservation must pass `max(0.001 m, original * 1e-10)`.
Cuts must be certified within 0.01 m; achieved maximum endpoint bounds are about 0.0000022 m.
Bounds include a 2 micrometer floor and a coordinate-evaluation allowance divided by the
certified local crossing derivative. Shallow crossings receive larger bounds or fail closed.
These are engineering certificates backed by convergence and crosschecks, not formal interval-arithmetic proofs.
State bounds sum actual incident endpoint uncertainties, including allocation fractions.
Exact sample counts and savings use only 1e-7 m floating arithmetic tolerance, never ±5 m.
Native boundary ownership is a numerical model, not surveyed legal ownership. The references
fail closed on ambiguous near tangencies; their certified source domain is nonpolar edges
at most 200 km. Core files do not test dateline/Hawaii ownership or actual boundary holes.
The 04 file is intentionally smaller than the main-network target so centimeter controls
remain understandable. No finite collection proves universal correctness.

## Application comparison and exports

The [application report](validation/application_comparison.json) is produced only after
independent expectations are fixed. All original/state mileage and exact savings comparisons
are checked within their stated bounds. Latest comparison against the fingerprinted application sources: **10,250 checks, 0 mismatches**, including exports.

The initial application baseline exposed two defect classes. The retained reproductions
document that history; the current receipt above determines whether they still occur:

1. **02 canonical boundary precision:** The baseline application round-tripped native coordinates
   through radians/degrees during longitude unwrapping. At the exact touch vertex this
   moved the boundary one floating-point unit west, creating a spurious TX attribution
   and reporting 10 crossing events where the unchanged native resource establishes 9.
2. **04 endpoint arithmetic:** A 4 m exclusive TX line ending exactly on the TX/NM border
   produced a spurious `8.881784197001252e-16 m` unresolved fragment in one direction.
   That fragment caused an incomplete state analysis; reversal was complete. The reference
   proves an endpoint touch and zero unresolved mileage.

The audit corrected overstrict export checks: ordinary boundary rounding is evaluated against
explicit local floating-point bounds. The historical [polygon residual](validation/reproductions/polygon_residual.json)
and [strict endpoint-certificate diagnostic](validation/reproductions/export_precision.json) remain
raw evidence, but do not represent failing acceptance checks. Fixture 03 now passes completely.

See [minimal boundary/endpoint reproductions](validation/reproductions/README.md) and
[polygon residual evidence](validation/reproductions/polygon_residual.json). A draft 02
touch was independently found to be slightly across an oblique boundary and was corrected
to an exact native vertex. That reference/construction correction is documented separately;
expectations were not tuned to an application result. Fixture creation did not modify
application code; subsequent application repairs are tracked separately.

`suite.py compare` intentionally exits nonzero while the recorded discrepancies persist.
A fixture can be reference-verified while exposing an application failure. Incomplete app
analysis is a failed comparison, never a zero-savings pass.

`suite.py audit` runs adversarial tests, fresh reference reconstruction, a fresh application
comparison including exports, and the existing stress profile when present. It passes only
with no unexpected regressions. Known failures require exact archive fingerprints and narrow
interval/value signatures; all unrelated assertions remain mandatory. A production fix is
accepted as a passing case. See the [audit findings](validation/audit/README.md) and
[machine-readable audit receipt](validation/audit/audit_report.json).

Export checks reimport Combined original and state exclusive mileage, ignore polygons as
pipeline geometry, and compare every fragment to independent source/path ownership spans.
Endpoints must match within the 1 cm cut target; each exported vertex must lie within 10 µm
of its original source geodesic. Source identities, exact qualifying sample ranges, public
attribution, positive interval lengths, coverage and conservation are checked independently.
Corridor polygons must match their expected source-pair sections, cover qualifying sample
midpoints, and stay within independently bounded section extents. Containment retains holes
and permits only a local 64-coordinate-ULP boundary strip
with its corresponding perimeter-based area bound. Foreign polygons fail even when tiny.
Shared geometry must occur once Combined and never in state maps. The no-empty-map rule
for allocation-only states is documented but not exercised by these main cases.

## Optional stress profile

The saved 24-group profile contains **96 sources, 27,120 samples,
135.895 km and 96 qualifying sections**.
It saves 74,640 m and completed application analysis in
8.11 s on the recorded host, including profiling overhead.
The [stress report](validation/stress_report.json) records actual candidate/neighbor counts,
hardware, process peak memory, runtime and independent/application agreement. These values
are machine-specific observations, not a performance guarantee.

The generator accepts `--profile stress --stress-groups N`; the measured wrapper accepts
`--groups N` for 2–64 groups. Each group contains overlapping branches, not just isolated
padding. The application retains its 1,000,000 segment, 5,000,000 candidate-check and
20,000,000 neighbor-visit limits. Resource-limit rejection is not a successful stress result.

## Folder guide

- `generator/`: deterministic construction, serialization and preview code.
- `reference/`: independent geometric and sampled-contract implementations.
- `fixtures/`: four main inputs; `variants/` and `stress/` keep extra inputs separate.
- `expected/`: full-precision goldens and CSV ledgers; separate optional stress results.
- `previews/`: overview maps; `details/` resolves meter/centimeter controls with explicit scales.
- `validation/`: provenance, reports, executable assertions, app comparisons and minimal reproductions.

Field definitions are in [SCHEMA.md](SCHEMA.md); original requirements are in the
[fixture prompt](../../../../docs/validation/kmz-fixture-agent-prompt.md).
