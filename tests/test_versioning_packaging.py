import subprocess
import json
import pytest
from pipeline_calculator import versioning


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, 'init', '-b', 'main')
    git(tmp_path, 'config', 'user.email', 'test@example.com')
    git(tmp_path, 'config', 'user.name', 'Test')
    git(tmp_path, 'commit', '--allow-empty', '-m', 'baseline')
    return tmp_path, git(tmp_path, 'rev-parse', 'HEAD')


def test_main_merges_and_rebuilds(repo):
    path, baseline = repo
    derive = lambda: versioning.derive_version(path, baseline=baseline, env={})
    assert derive()['version'] == '4.0'
    git(path, 'checkout', '-b', 'feature')
    for i in range(3):
        git(path, 'commit', '--allow-empty', '-m', str(i))
    preview = derive()
    assert preview['version'] == f"4.3-dev.{preview['commit'][:12]}"
    git(path, 'checkout', 'main')
    git(path, 'merge', '--no-ff', 'feature', '-m', 'merge')
    assert derive()['version'] == '4.1'
    assert derive() == derive()
    for i in range(9):
        git(path, 'commit', '--allow-empty', '-m', str(i))
    assert derive()['version'] == '4.10'


def test_dirty_and_pr(repo):
    path, baseline = repo
    assert '-dev.' in versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/pull/1/merge'})['version']
    (path / 'untracked.txt').write_text('change')
    assert versioning.derive_version(path, baseline=baseline, env={})['version'].endswith('.dirty')
    with pytest.raises(ValueError, match='clean working tree'):
        versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/heads/main'})


def test_tags(repo):
    path, baseline = repo
    git(path, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
    git(path, 'checkout', '--detach')
    assert versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/tags/v4.0'})['version'] == '4.0'
    with pytest.raises(ValueError, match='Release tag'):
        versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/tags/v5.0'})
    git(path, 'commit', '--allow-empty', '-m', 'off main')
    with pytest.raises(ValueError, match='Release tag'):
        versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/tags/v4.1'})


def test_old_branch_and_shallow_clone(repo, tmp_path):
    path, old = repo
    git(path, 'commit', '--allow-empty', '-m', 'new baseline')
    baseline = git(path, 'rev-parse', 'HEAD')
    git(path, 'checkout', '-b', 'old-feature', old)
    assert '-dev.' in versioning.derive_version(path, baseline=baseline, env={})['version']
    clone = tmp_path / 'shallow'
    git(path, 'clone', '--depth', '1', path.as_uri(), str(clone))
    with pytest.raises(ValueError, match='Full Git history'):
        versioning.derive_version(clone, baseline=old, env={})


def test_embedded_version_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr(versioning, '__file__', str(tmp_path / 'versioning.py'))
    (tmp_path / 'version.json').write_text(json.dumps({'version':'4.12'}))
    monkeypatch.setattr(versioning, 'derive_version', lambda *a, **k: pytest.fail('Git called'))
    assert versioning.get_version() == '4.12'
