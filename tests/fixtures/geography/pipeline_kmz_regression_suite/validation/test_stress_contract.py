"""Publication and receipt safeguards for the optional stress workload."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest

for dependency in ('geographiclib', 'matplotlib', 'psutil'):
    pytest.importorskip(dependency, reason='Install the KMZ suite requirements to run its optional audit tests')

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.validation import run_stress as stress


def provenance():
    return {'baseline_contract_unchanged': True, 'changed_contract_files': []}


@pytest.fixture(scope='session')
def reference_pair(tmp_path_factory):
    """A small real reference, independent of optional delivered stress assets."""
    from pipeline_kmz_regression_suite.generator.generate import main as generate

    root = tmp_path_factory.mktemp('kmz-stress-contract-reference')
    generate(['--profile', 'stress', '--stress-groups', '2', '--output', str(root)])
    reference = stress.build_reference(root / stress.FIXTURE, 2)
    stress.dump(root / stress.EXPECTATION, reference)
    return root


@pytest.fixture
def frozen_pair(tmp_path, reference_pair):
    for relative in (stress.FIXTURE, stress.EXPECTATION):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference_pair / relative, target)
    return tmp_path


def fingerprints(root):
    return {relative: stress.digest(root / relative) for relative in (stress.FIXTURE, stress.EXPECTATION)}


def test_validate_existing_never_generates_or_rewrites_goldens(frozen_pair, monkeypatch):
    before = fingerprints(frozen_pair)
    verified = []

    def verify(root):
        verified.append(root)
        return {'status': 'reference-verified', 'passed': True}

    def forbidden(*args, **kwargs):
        pytest.fail('Read-only stress verification must not invoke the generator')

    monkeypatch.setattr(stress, 'validate_stress_reference', verify)
    monkeypatch.setattr(stress, 'contract_provenance', provenance)
    monkeypatch.setattr(stress.subprocess, 'run', forbidden)
    reference, sources, proof = stress.prepare_reference(validate_existing=True, root=frozen_pair)
    assert verified == [frozen_pair]
    assert fingerprints(frozen_pair) == before
    assert len(sources) == 4 * reference['groups']
    assert proof['reference_design']['state_inputs_equal_complete_paths']


@pytest.mark.parametrize('verification', [
    {'status': 'not_generated', 'passed': True},
    {'status': 'reference-verified', 'passed': False},
])
def test_validate_existing_requires_an_actual_verified_pair(tmp_path, monkeypatch, verification):
    monkeypatch.setattr(stress, 'contract_provenance', provenance)
    monkeypatch.setattr(stress, 'validate_stress_reference', lambda root: verification)
    with pytest.raises(ValueError, match='complete frozen stress'):
        stress.prepare_reference(validate_existing=True, root=tmp_path)
    assert not (tmp_path / stress.EXPECTATION).exists()


def test_changed_contract_prevents_generation(frozen_pair, monkeypatch):
    before = fingerprints(frozen_pair)
    monkeypatch.setattr(stress, 'contract_provenance', lambda: {
        'baseline_contract_unchanged': False, 'changed_contract_files': ['core/overlap.py']})

    def forbidden(*args, **kwargs):
        pytest.fail('Contract guard must run before generator invocation')

    monkeypatch.setattr(stress.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='changed calculation contract'):
        stress.prepare_reference(root=frozen_pair)
    assert fingerprints(frozen_pair) == before


def test_failed_staged_verification_preserves_delivered_artifacts(frozen_pair, monkeypatch, reference_pair):
    before = fingerprints(frozen_pair)
    reference = stress.read_json(frozen_pair / stress.EXPECTATION)
    staged_roots = []

    def generate(command, **kwargs):
        stage = Path(command[command.index('--output') + 1])
        staged_roots.append(stage)
        (stage / stress.FIXTURE).parent.mkdir(parents=True)
        shutil.copy2(reference_pair / stress.FIXTURE, stage / stress.FIXTURE)

    def reject(stage):
        assert stage != frozen_pair
        assert fingerprints(frozen_pair) == before
        raise ValueError('Independent reference detects missing overlap')

    monkeypatch.setattr(stress, 'contract_provenance', provenance)
    monkeypatch.setattr(stress.subprocess, 'run', generate)
    monkeypatch.setattr(stress, 'build_reference', lambda path, groups: reference)
    monkeypatch.setattr(stress, 'validate_stress_reference', reject)
    with pytest.raises(ValueError, match='missing overlap'):
        stress.prepare_reference(2, root=frozen_pair)
    assert fingerprints(frozen_pair) == before
    assert staged_roots and all(not stage.exists() for stage in staged_roots)


def test_publication_waits_for_independent_staged_verification(tmp_path, monkeypatch, reference_pair):
    reference = stress.read_json(reference_pair / stress.EXPECTATION)
    for relative in (stress.FIXTURE, stress.EXPECTATION):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'previous artifact')
    before = fingerprints(tmp_path)
    verified = []

    def generate(command, **kwargs):
        stage = Path(command[command.index('--output') + 1])
        (stage / stress.FIXTURE).parent.mkdir(parents=True)
        shutil.copy2(reference_pair / stress.FIXTURE, stage / stress.FIXTURE)

    def verify(stage):
        assert fingerprints(tmp_path) == before
        assert stage != tmp_path
        assert (stage / stress.EXPECTATION).is_file()
        verified.append(stage)
        return {'status': 'reference-verified', 'passed': True}

    monkeypatch.setattr(stress, 'contract_provenance', provenance)
    monkeypatch.setattr(stress.subprocess, 'run', generate)
    monkeypatch.setattr(stress, 'build_reference', lambda path, groups: reference)
    monkeypatch.setattr(stress, 'validate_stress_reference', verify)
    stress.prepare_reference(2, root=tmp_path)
    assert len(verified) == 1
    assert stress.digest(tmp_path / stress.FIXTURE) == stress.digest(reference_pair / stress.FIXTURE)
    assert stress.read_json(tmp_path / stress.EXPECTATION) == reference


def test_failure_receipt_preserves_detailed_checks_and_measurements(tmp_path, monkeypatch):
    def fail(groups, **kwargs):
        kwargs['report'].update(checks=[{'assertion': 'Wrong source sample range', 'passed': False}],
                                actual_overlap_pass_work=[{'segment_count': 20}],
                                application_runtime_seconds=1.2)
        raise ValueError('Sample contract mismatch')

    monkeypatch.setattr(stress, 'SUITE', tmp_path)
    monkeypatch.setattr(stress, 'run_profile', fail)
    with pytest.raises(SystemExit, match='Sample contract mismatch'):
        stress.main(['--validate-existing'])
    receipt = stress.read_json(tmp_path / stress.REPORT)
    assert receipt['status'] == 'failed' and receipt['passed'] is False
    assert receipt['checks'] == [{'assertion': 'Wrong source sample range', 'passed': False}]
    assert receipt['actual_overlap_pass_work'] == [{'segment_count': 20}]
    assert receipt['application_runtime_seconds'] == 1.2


@pytest.mark.parametrize('mutation', ['lost_group_overlap', 'changed_state_path'])
def test_stress_design_rejects_aggregate_only_success(mutation, reference_pair):
    reference = stress.read_json(reference_pair / stress.EXPECTATION)
    sources = stress.read_kmz(reference_pair / stress.FIXTURE)
    assert stress.validate_stress_design(reference, sources)['passed']
    damaged = deepcopy(reference)
    if mutation == 'lost_group_overlap':
        # The old positive aggregate savings still looks plausible.
        damaged['sampled']['motif_savings'][sources[0]['motif']] = 0.0
    else:
        damaged['geometry']['state_inputs']['TX'][0]['paths'][0][0][0] += .001
    with pytest.raises(ValueError):
        stress.validate_stress_design(damaged, sources)


@pytest.mark.parametrize('actual,target', [(float('nan'), 0.0), (float('inf'), 1.0),
                                         ({'count': True}, {'count': 1})])
def test_stress_comparison_rejects_nonfinite_and_boolean_numeric_results(actual, target):
    assert not stress._equal(actual, target)


def test_failure_receipt_can_record_nonfinite_output(tmp_path):
    value = {'actual': [float('nan'), float('inf')], 'passed': False}
    stress.dump(tmp_path / 'receipt.json', stress._safe_value(value))
    assert stress.read_json(tmp_path / 'receipt.json') == {'actual': ['nan', 'inf'], 'passed': False}
