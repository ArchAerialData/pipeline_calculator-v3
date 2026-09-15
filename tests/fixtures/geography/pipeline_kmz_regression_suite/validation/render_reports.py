"""Rebuild human-readable tables and coverage matrix from fixed reference files."""
from __future__ import annotations

import json
from pathlib import Path
import sys

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.suite import BASELINE, BOUNDARY_SHA256, digest, read_json, select_fixtures, verify_report_freshness


def read(path):
    return read_json(path)


def main():
    select_fixtures(SUITE)
    expected = [read(p) for p in sorted((SUITE / 'expected').glob('*.expected.json'))]
    mains = [e for e in expected if '/' not in e['fixture']]
    report = read(SUITE / 'validation/reference_report.json')
    verify_report_freshness(report, SUITE)
    matrix = ['# Numerical coverage matrix', '',
              'Generated from the fixed independent expectations. Every row is an executed assertion;',
              'full section/sample sets and unrounded evidence remain in the linked JSON. Application',
              'mismatches are separate from these reference results.', '']
    for e in mains:
        stem = Path(e['fixture']).stem
        matrix += [f'## {stem}', '', f'[Complete numerical evidence](../expected/{stem}.expected.json)', '',
                   '| Check / motif | Source keys | Numerical assertion | Measured evidence | Result |',
                   '| --- | --- | --- | --- | --- |']
        for check in e['coverage_checks']:
            measured = json.dumps(check['measured'], sort_keys=True, separators=(',', ':')).replace('|', '\\|')
            keys = ', '.join(check['source_keys']) or 'entire fixture'
            matrix.append(f'| {check["id"]} / {", ".join(check.get("motifs", []))} | {keys} | {check["assertion"]} | `{measured}` | {"PASS" if check["passed"] else "FAIL"} |')
        matrix += ['']
    matrix += ['## Variations', '', '| Variant | Largest source/allocated change (m) | Savings changes (m) | Pass |',
               '| --- | ---: | --- | --- |']
    for variation in report['variant_checks']:
        matrix.append(f'| {variation["fixture"]} | {variation["maximum_source_length_or_allocation_difference_meters"]:.12g} | `{json.dumps(variation["savings_difference_meters_by_scope"], sort_keys=True)}` | {variation["passed"]} |')
    matrix += ['', 'Only source-order savings invariance is required generally. Reversed and redundant-vertex',
               'expectations are independently recalculated. The saved 04 variants happen to preserve savings.', '',
               '## What these cases cannot establish', '',
               'The core suite does not establish arbitrary winding-river coincidence, polar/near-tangent clipping,',
               'Alaska dateline or Hawaii ownership, native boundary-hole behavior, surveyed legal ownership,',
               'or successful analysis at every workload. No state here is represented solely by shared allocation;',
               'the no-empty-map rule is a reusable export requirement, not a demonstrated case. Hand-worked',
               'single-source retracing is tested in the sampled reference; the main input loops are simple.', '']
    (SUITE / 'validation/coverage_matrix.md').write_text('\n'.join(matrix), encoding='utf-8')

    text = ['# Pipeline KMZ regression suite', '',
            '**Four main KMZs and three variations are reference-verified.** They contain real synthetic',
            'pipeline inputs, an offline deterministic generator, two independent references, expected',
            'results, interval/overlap/crossing ledgers, preview maps and executable intent checks.',
            'Application comparisons are recorded separately; see the known discrepancies below.', '',
            '## Run', '', 'Run these commands from the repository root with CPython 3.13.3:', '',
            '```powershell',
            'python -m pip install -r tests/fixtures/geography/pipeline_kmz_regression_suite/requirements.txt',
            '# Regenerate all core archives, previews and independent expected results:',
            'python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py regenerate',
            '# Independently reconstruct and check every core archive and ledger:',
            'python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py validate',
            '# Compare fixed expectations with the application and exercise KMZ exports:',
            'python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py compare',
            '# Run adversarial tests, fresh references, exports and known-defect classification:',
            'python tests/fixtures/geography/pipeline_kmz_regression_suite/suite.py audit',
            '# Generate and measure the optional larger branching-overlap profile:',
            'python tests/fixtures/geography/pipeline_kmz_regression_suite/validation/run_stress.py',
            '# Remeasure the existing stress case without replacing its frozen expectation:',
            'python tests/fixtures/geography/pipeline_kmz_regression_suite/validation/run_stress.py --validate-existing',
            '```', '',
            'Dependency installation is the only step that may require a network. With the pinned packages',
            'and bundled boundary resource supplied, all commands operate offline. Use a separate virtual',
            'environment if the application needs different package pins. Regeneration permits unrelated',
            'later commits but rejects contract source changes, including uncommitted changes, until reviewed.',
            'It stages all core outputs, validates them, then publishes with rollback on ordinary I/O errors.',
            'Validation and application comparison can run against later application revisions.',
            'Missing inputs, empty selections, nonfinite values and stale evidence are errors. Focused',
            '`--only` runs use separate reports. Saved reports fingerprint assets and validation code.', '',
            'Ordinary repository test discovery reports skips if optional audit dependencies are absent.',
            'The dedicated audit command rejects skipped tests or missing collection; install the suite',
            'requirements before using its receipt as evidence.', '',
            f'Baseline: `{BASELINE}`. Generator: `1.0.0`; seed: `71499`.',
            f'Boundary SHA-256: `{BOUNDARY_SHA256}`.',
            'The generator and references each verify the bundled archive and its individual WKB hashes.',
            'The [complete boundary manifest](validation/boundary_manifest.json) preserves Census attribution,',
            '2025 vintage, prior transformation operations, accuracy qualifications and original vertex model.',
            'No new transformation or simplified boundary is substituted.', '',
            '## Results', '',
            'Displayed values are rounded for reading. JSON/CSV values retain numerical precision.',
            'State original below means **attributed original**, including its equal shared allocation.',
            'Pairwise qualifying coverage is separate from savings. State adjusted totals need not sum',
            'to Combined adjusted mileage because state sampling and qualification restart on clipped interiors.', '']
    for e in mains:
        stem = Path(e['fixture']).stem
        geo, archive = e['geometry'], e['archive']
        text += [f'### {stem}', '',
                 f'[KMZ](fixtures/{stem}.kmz) · [Expected JSON](expected/{stem}.expected.json) · [Preview](previews/{stem}.png)', '',
                 f'{archive["source_count"]} sources · {archive["path_count"]} paths · {archive["vertex_count"]} vertices · '
                 f'{geo["crossing_count"]} crossing events · {len(geo["endpoint_touches"])} endpoint touches.', '',
                 '| Scope | Original m | US survey mi | Pairwise qualifying m | Savings m | Adjusted m |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for scope in ['Combined'] + sorted(k for k in e['analyses'] if k != 'Combined'):
            row = e['analyses'][scope]
            original = row.get('attributed_original_meters', row['original_meters'])
            text.append(f'| {scope} | {original:,.6f} | {original / 1609.347218694:,.6f} | {row["qualifying_pairwise_meters"]:,.0f} | {row["savings_meters"]:,.0f} | {row["adjusted_meters"]:,.6f} |')
        angles = [c['angle_degrees'] for c in geo['crossings'] if c['angle_degrees'] is not None and not c['via_shared_interval']]
        if angles:
            text += ['', f'Transverse crossing angles: {min(angles):.6f}–{max(angles):.6f} degrees; each event is in the crossing ledger.']
        text += ['']
    text += ['## Controls and interpretation', '',
             '- **01:** TX/LA/WY each has positive pair, compatible-trio and incompatible-outside-pair',
             '  controls; perpendicular crossings, diverging branches, loops and isolated lines; two',
             '  disconnected 123.25 m design runs per multipart source. Actual samples qualify separately.',
             '  Full-coverage source groups are verified with explicit sets; perfectly aligned diagonal',
             '  eligibility is asserted only in the hand-worked controls. State savings are 3,600 m each',
             '  and add to Combined because the original independent path inputs are preserved.',
             '- **02:** Continuous NM/TX/OK visits, sparse-edge crossings, reentry, a genuine short state',
             '  visit and an exact native-vertex endpoint touch. Three interior pairs supply positive savings.',
             '- **03:** Long/short/asymmetric border splits, a joining trio, distinct diverge/rejoin',
             '  sections, an opposite-state pair, and nonaligned starts/tails/opposite digitization.',
             '  The short split saves 300 m Combined and zero in either state. The opposite-state pair',
             '  saves 705 m Combined and zero in its separate state inputs. Connectors and detached',
             '  feeder controls are included in full accounting and the global pair search.',
             '- **04:** Verified TX/NM meridian sharing; shared intervals above and below the overlap',
             '  minimum; a partner that must not use shared mileage to qualify state savings; 3 cm',
             '  exclusive offsets; a 10 cm crossing; endpoint touch; reversal and collinear-vertex variants.',
             '  Shared physical length is approximately 1,133.25 m, stored once and allocated equally.', '',
             'The seven files execute ' + str(sum(len(e['coverage_checks']) for e in expected)) + ' numerical intent assertions plus nine hand-worked controls.',
             'The [coverage matrix](validation/coverage_matrix.md) links each assertion to source keys',
             'and measured values. Repeated display names and OBJECTIDs never merge source identities.',
             'The source-order variant preserves metadata by stable key. Reverse/extra-vertex variants',
             'are measured independently; their unchanged 04 savings are an observed phase-safe result,',
             'not a general reversal invariant. Renaming is not asserted invariant.', '',
             '## Evidence, uncertainty and limitations', '',
             'The [geometric reference](reference/README.md) parses final XML directly, measures GRS80',
             'geodesics with GeographicLib, crosschecks each original edge using pyproj/PROJ, and solves',
             'native coordinate-linear boundary equations with curvature bounds and root refinement.',
             'The sampled reference uses cumulative chainage interpolation, exhaustive bounded arrays,',
             'explicit coverage sets and disjoint clique grouping. Neither imports application code.', '',
             'Per-source and whole-fixture conservation must pass `max(0.001 m, original * 1e-10)`.',
             'Cuts must be certified within 0.01 m; achieved maximum endpoint bounds are about 0.0000022 m.',
             'Bounds include a 2 micrometer floor and a coordinate-evaluation allowance divided by the',
             'certified local crossing derivative. Shallow crossings receive larger bounds or fail closed.',
             'These are engineering certificates backed by convergence and crosschecks, not formal interval-arithmetic proofs.',
             'State bounds sum actual incident endpoint uncertainties, including allocation fractions.',
             'Exact sample counts and savings use only 1e-7 m floating arithmetic tolerance, never ±5 m.',
             'Native boundary ownership is a numerical model, not surveyed legal ownership. The references',
             'fail closed on ambiguous near tangencies; their certified source domain is nonpolar edges',
             'at most 200 km. Core files do not test dateline/Hawaii ownership or actual boundary holes.',
             'The 04 file is intentionally smaller than the main-network target so centimeter controls',
             'remain understandable. No finite collection proves universal correctness.', '',
             '## Application comparison and exports', '',
             'The [application report](validation/application_comparison.json) is produced only after',
             'independent expectations are fixed. All original/state mileage and exact savings comparisons',
             'agree within their stated bounds. Two baseline defect classes remain visible:', '',
             '1. **02 canonical boundary precision:** The application round-trips native coordinates',
             '   through radians/degrees during longitude unwrapping. At the exact touch vertex this',
             '   moves the boundary one floating-point unit west, creating a spurious TX attribution',
             '   and reporting 10 crossing events where the unchanged native resource establishes 9.',
             '2. **04 endpoint arithmetic:** A 4 m exclusive TX line ending exactly on the TX/NM border',
             '   produces a spurious `8.881784197001252e-16 m` unresolved fragment in one direction.',
             '   That fragment causes an incomplete state analysis; reversal is complete. The reference',
             '   proves an endpoint touch and zero unresolved mileage.', '',
             'The audit corrected overstrict export checks: ordinary boundary rounding is evaluated against',
             'explicit local floating-point bounds. The historical [polygon residual](validation/reproductions/polygon_residual.json)',
             'and [strict endpoint-certificate diagnostic](validation/reproductions/export_precision.json) remain',
             'raw evidence, but do not represent failing acceptance checks. Fixture 03 now passes completely.', '',
             'See [minimal boundary/endpoint reproductions](validation/reproductions/README.md) and',
             '[polygon residual evidence](validation/reproductions/polygon_residual.json). A draft 02',
             'touch was independently found to be slightly across an oblique boundary and was corrected',
             'to an exact native vertex. That reference/construction correction is documented separately;',
             'expectations were not tuned to an application result. No application code was changed.', '',
             '`suite.py compare` intentionally exits nonzero while the recorded discrepancies persist.',
             'A fixture can be reference-verified while exposing an application failure. Incomplete app',
             'analysis is a failed comparison, never a zero-savings pass.', '',
             '`suite.py audit` runs adversarial tests, fresh reference reconstruction, a fresh application',
             'comparison including exports, and the existing stress profile when present. It passes only',
             'with no unexpected regressions. Known failures require exact archive fingerprints and narrow',
             'interval/value signatures; all unrelated assertions remain mandatory. A production fix is',
             'accepted as a passing case. See the [audit findings](validation/audit/README.md) and',
             '[machine-readable audit receipt](validation/audit/audit_report.json).', '',
             'Export checks reimport Combined original and state exclusive mileage, ignore polygons as',
             'pipeline geometry, and compare every fragment to independent source/path ownership spans.',
             'Endpoints must match within the 1 cm cut target; each exported vertex must lie within 10 µm',
             'of its original source geodesic. Source identities, exact qualifying sample ranges, public',
             'attribution, positive interval lengths, coverage and conservation are checked independently.',
             'Corridor polygons must have the expected source-pair identities and cover qualifying sample',
             'midpoints. Containment retains holes and permits only a local 64-coordinate-ULP boundary strip',
             'with its corresponding perimeter-based area bound. Foreign polygons fail even when tiny.',
             'Shared geometry must occur once Combined and never in state maps. The no-empty-map rule',
             'for allocation-only states is documented but not exercised by these main cases.', '',
             '## Optional stress profile', '']
    stress_path = SUITE / 'validation/stress_report.json'
    if stress_path.exists():
        stress = read(stress_path)
        current_stress = (stress.get('status') == 'complete' and stress.get('passed') and
                          stress.get('sha256') == digest(SUITE / 'fixtures/stress/stress_branching_network.kmz') and
                          stress.get('expectation_sha256') == digest(SUITE / 'expected/stress/stress_branching_network.expected.json'))
    else:
        current_stress = False
    if current_stress:
        text += [f'The saved {stress["groups"]}-group profile contains **{stress["source_count"]} sources, {stress["independent_sample_count"]:,} samples,',
                 f'{stress["independent_original_meters"] / 1000:.3f} km and {stress["independent_qualifying_section_count"]} qualifying sections**.',
                 f'It saves {stress["independent_savings_meters"]:,.0f} m and completed application analysis in',
                 f'{stress["application_runtime_seconds"]:.2f} s on the recorded host, including profiling overhead.',
                 'The [stress report](validation/stress_report.json) records actual candidate/neighbor counts,',
                 'hardware, process peak memory, runtime and independent/application agreement. These values',
                 'are machine-specific observations, not a performance guarantee.', '']
    else:
        text += ['No complete stress measurement matches the current assets; run the stress wrapper to measure them.', '']
    text += ['The generator accepts `--profile stress --stress-groups N`; the measured wrapper accepts',
             '`--groups N` for 2–64 groups. Each group contains overlapping branches, not just isolated',
             'padding. The application retains its 1,000,000 segment, 5,000,000 candidate-check and',
             '20,000,000 neighbor-visit limits. Resource-limit rejection is not a successful stress result.', '',
             '## Folder guide', '',
             '- `generator/`: deterministic construction, serialization and preview code.',
             '- `reference/`: independent geometric and sampled-contract implementations.',
             '- `fixtures/`: four main inputs; `variants/` and `stress/` keep extra inputs separate.',
             '- `expected/`: full-precision goldens and CSV ledgers; separate optional stress results.',
             '- `previews/`: overview maps; `details/` resolves meter/centimeter controls with explicit scales.',
             '- `validation/`: provenance, reports, executable assertions, app comparisons and minimal reproductions.', '',
             'Field definitions are in [SCHEMA.md](SCHEMA.md); original requirements are in the',
             '[fixture prompt](../../../../docs/validation/kmz-fixture-agent-prompt.md).', '']
    (SUITE / 'README.md').write_text('\n'.join(text), encoding='utf-8')


if __name__ == '__main__':
    main()
