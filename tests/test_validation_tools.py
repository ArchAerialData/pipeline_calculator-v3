from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validation.common import digest, parallel, write_fixture
from scripts.validation.run_corpus import local_manifest, run_manifest, validate_manifest


def test_deterministic_kmz(tmp_path):
    first = write_fixture(tmp_path / 'one.kmz', parallel(), linked=2)
    second = write_fixture(tmp_path / 'two.kmz', parallel(), linked=2)
    assert digest(first) == digest(second)


def test_manifest_hash_missing_and_unknown(tmp_path):
    data = local_manifest(tmp_path)
    item = data['fixtures'][0]
    item['sha256'] = '0' * 64
    assert run_manifest({'schema_version': 1, 'fixtures': [item]}, tmp_path)[0]['status'] == 'failed'
    item['path'] = 'not-present-private-fixture.kmz'
    assert run_manifest({'schema_version': 1, 'fixtures': [item]}, tmp_path)[0]['status'] == 'missing-private-input'
    assert run_manifest({'schema_version': 1, 'fixtures': [item]}, tmp_path, strict=True)[0]['status'] == 'failed'
    item['status'] = 'invented'
    with pytest.raises(ValueError, match='status'):
        validate_manifest({'schema_version': 1, 'fixtures': [item]})


def test_all_synthetic_corpus_and_exports(tmp_path):
    data = local_manifest(tmp_path)
    data['fixtures'] = [i for i in data['fixtures'] if i['status'] == 'analytic']
    assert all(r['status'] == 'passed' for r in run_manifest(data, tmp_path))
    assert list(tmp_path.glob('export-*.xlsx'))


def test_manifest_needs_expectation_basis_and_reviewer(tmp_path):
    item = local_manifest(tmp_path)['fixtures'][0]
    item.pop('basis')
    with pytest.raises(ValueError, match='basis'):
        validate_manifest({'schema_version': 1, 'fixtures': [item]})

    item['basis'] = 'Independent GIS measurements'
    item['status'] = 'reviewed'
    with pytest.raises(ValueError, match='reviewer'):
        validate_manifest({'schema_version': 1, 'fixtures': [item]})


def test_bounded_process_timeout_stops_its_child(tmp_path):
    import os
    import subprocess
    from scripts.validation.processes import run_bounded
    marker=tmp_path/'child.pid'
    code=("import subprocess,sys,time;from pathlib import Path;"
          "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
          f"Path({str(marker)!r}).write_text(str(p.pid));time.sleep(60)")
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable,'-c',code],timeout=1)
    assert marker.exists()
    pid=int(marker.read_text())
    if os.name=='nt':
        check=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/NH','/FO','CSV'],capture_output=True,text=True,timeout=10)
        assert f'"{pid}"' not in check.stdout
