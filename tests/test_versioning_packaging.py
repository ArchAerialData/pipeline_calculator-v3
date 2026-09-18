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
    assert derive()['version'] == '5.0'
    git(path, 'checkout', '-b', 'feature')
    for i in range(3):
        git(path, 'commit', '--allow-empty', '-m', str(i))
    preview = derive()
    assert preview['version'] == f"5.3-dev.{preview['commit'][:12]}"
    git(path, 'checkout', 'main')
    git(path, 'merge', '--no-ff', 'feature', '-m', 'merge')
    assert derive()['version'] == '5.1'
    assert derive() == derive()
    for i in range(9):
        git(path, 'commit', '--allow-empty', '-m', str(i))
    assert derive()['version'] == '5.10'


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
    assert versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/tags/v5.0'})['version'] == '5.0'
    with pytest.raises(ValueError, match='Release tag'):
        versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/tags/v6.0'})
    git(path, 'commit', '--allow-empty', '-m', 'off main')
    with pytest.raises(ValueError, match='Release tag'):
        versioning.derive_version(path, baseline=baseline, env={'GITHUB_REF':'refs/tags/v5.1'})


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
    (tmp_path / 'version.json').write_text(json.dumps({'version':'5.12'}))
    monkeypatch.setattr(versioning, 'derive_version', lambda *a, **k: pytest.fail('Git called'))
    assert versioning.get_version() == '5.12'


def write_major(path, major):
    source = path / versioning.VERSION_SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f'MAJOR = {major}\n')
    git(path, 'add', versioning.VERSION_SOURCE)


def test_major_introduction_resets_on_main_merge(repo):
    path, _ = repo
    write_major(path, 4)
    git(path, 'commit', '-m', 'old major')
    git(path, 'checkout', '-b', 'v5')
    write_major(path, 5)
    git(path, 'commit', '-m', 'introduce v5')
    assert versioning.derive_version(path, env={})['numeric_version'] == '5.0'
    git(path, 'commit', '--allow-empty', '-m', 'preview followup')
    assert versioning.derive_version(path, env={})['numeric_version'] == '5.1'
    git(path, 'checkout', 'main')
    git(path, 'commit', '--allow-empty', '-m', 'concurrent main work')
    git(path, 'merge', '--no-ff', 'v5', '-m', 'release v5')
    assert versioning.derive_version(path, env={})['version'] == '5.0'
    git(path, 'commit', '--allow-empty', '-m', 'next release')
    assert versioning.derive_version(path, env={})['version'] == '5.1'


def test_uncommitted_major_bump_is_preview(repo):
    path, _ = repo
    write_major(path, 4)
    git(path, 'commit', '-m', 'old major')
    write_major(path, 5)
    result = versioning.derive_version(path, env={})
    assert result['version'].startswith('5.0-dev.')
    assert result['version'].endswith('.dirty')
    assert result['version_baseline'] is None
    with pytest.raises(ValueError, match='clean working tree'):
        versioning.derive_version(path, env={'GITHUB_REF': 'refs/heads/main'})


def test_clean_display_preserves_full_build_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(versioning, '__file__', str(tmp_path / 'versioning.py'))
    full = '5.0-dev.123456abcdef.dirty'
    (tmp_path / 'version.json').write_text(json.dumps({'version': full, 'numeric_version': '5.0'}))
    assert versioning.get_display_version() == '5.0'
    assert versioning.get_version() == full
