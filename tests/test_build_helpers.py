from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform!='win32',reason='Native Windows PowerShell helper execution')
@pytest.mark.parametrize('compile_exit,test_exit,expected',[(0,0,0),(7,0,1),(0,7,1)])
def test_windows_helper_native_failures(tmp_path,compile_exit,test_exit,expected):
    script=ROOT/'scripts/windows/run_tests.ps1'
    dest=tmp_path/'scripts/windows/run_tests.ps1'
    dest.parent.mkdir(parents=True)
    (tmp_path/'src').mkdir()
    fake=tmp_path/'python.cmd'
    fake.write_text(f'@echo off\nif "%2"=="compileall" exit /b {compile_exit}\nexit /b {test_exit}\n')
    # Keep the actual helper body; redirect only its interpreter to an isolated fake.
    text=script.read_text().replace('$Py = Join-Path $VenvDir "Scripts\\python.exe"',f"$Py = '{fake}'")
    dest.write_text(text)
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(dest)],capture_output=True,text=True,timeout=30)
    assert (result.returncode!=0)==bool(expected)
    assert ('OK' in result.stdout)==(expected==0)


def test_source_smoke_entrypoints(tmp_path):
    import json
    for entry,implementation in [('pipeline_calculator_entry.py','new'),('pipeline_calculator_v3.py','legacy')]:
        output=tmp_path/f'{implementation}.json'
        process=subprocess.run([sys.executable,str(ROOT/'src'/entry),'--smoke-test',str(output)],capture_output=True,text=True,timeout=30)
        assert process.returncode==0,process.stderr
        report=json.loads(output.read_text())
        assert report['status']=='passed' and report['implementation']==implementation


@pytest.mark.skipif(sys.platform!='win32',reason='Native Windows build helper')
def test_build_failure_is_nonzero_and_outputs_are_isolated(tmp_path):
    dest=tmp_path/'scripts/windows/build_exe.ps1'
    dest.parent.mkdir(parents=True)
    fake=tmp_path/'python.cmd'
    fake.write_text('@echo off\nif "%2"=="PyInstaller" exit /b 7\necho 4.2-dev.test\nexit /b 0\n')
    text=(ROOT/'scripts/windows/build_exe.ps1').read_text().replace(
        '$Py = Join-Path $VenvDir "Scripts\\python.exe"',f"$Py = '{fake}'")
    dest.write_text(text)
    sentinel=tmp_path/'dist/keep.exe'
    sentinel.parent.mkdir();sentinel.write_text('preserved')
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(dest),
                           '-OutputRoot',str(tmp_path/'isolated')],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'PyInstaller failed' in result.stderr
    assert sentinel.read_text()=='preserved'


@pytest.mark.parametrize('status',[0,7])
def test_bash_helper_uses_environment_python_and_propagates_failure(tmp_path,status):
    bash=shutil.which('bash')
    git_bash=Path('C:/Program Files/Git/bin/bash.exe')
    if sys.platform=='win32':
        bash=str(git_bash) if git_bash.exists() else None
    if not bash:
        pytest.skip('No local Bash interpreter; macOS native execution is separate')
    dest=tmp_path/'scripts/macos/run_tests.sh';dest.parent.mkdir(parents=True)
    dest.write_text((ROOT/'scripts/macos/run_tests.sh').read_text(),newline='\n')
    bin_dir=tmp_path/'.venv/bin';bin_dir.mkdir(parents=True)
    (bin_dir/'activate').write_text('export PATH="${VENV_DIR}/bin:$PATH"\n',newline='\n')
    fake=bin_dir/'python'
    fake.write_text(f'#!/usr/bin/env bash\nif [[ "$2" == "pytest" ]]; then exit {status}; fi\nexit 0\n',newline='\n')
    fake.chmod(0o755)
    result=subprocess.run([bash,str(dest).replace('\\','/')],capture_output=True,text=True,timeout=30)
    assert (result.returncode!=0)==bool(status),result.stderr
