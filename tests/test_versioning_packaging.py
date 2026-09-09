from __future__ import annotations

from pathlib import Path

import pipeline_calculator
import pipeline_calculator_v3


def test_versions_are_v4() -> None:
    assert pipeline_calculator.__version__ == "4.0.0"
    assert pipeline_calculator_v3.__version__ == "4.0.0"


def test_packaging_artifact_names_are_v4() -> None:
    repo = Path(".")

    assert "Pipeline_Calculator_v4" in (repo / "scripts/windows/build_exe.ps1").read_text()
    assert "Pipeline_Calculator_v4.dmg" in (repo / "scripts/macos/package_dmg.sh").read_text()
    assert "Pipeline_Calculator_v4.dmg" in (repo / ".github/workflows/build.yaml").read_text()
    assert "Pipeline_Calculator_v4.exe" in (repo / ".github/workflows/build.yaml").read_text()
