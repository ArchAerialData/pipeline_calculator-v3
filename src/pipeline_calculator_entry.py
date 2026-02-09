"""PyInstaller-friendly entrypoint for the refactored (modular) GUI.

Why this file exists:
- PyInstaller expects a script path as its analysis entrypoint.
- We want distribution builds to default to the modular GUI implementation.
- We still keep the legacy monolith (`pipeline_calculator_v3`) available as a
  runtime fallback via `pipeline_calculator.app`.

This script sets `PIPELINE_CALCULATOR_IMPL=new` unless it is already set, then
delegates to `pipeline_calculator.app.main()`.
"""

from __future__ import annotations

import os

from pipeline_calculator.app import main


def _force_modular_default() -> None:
    # Allow callers to override explicitly (e.g., force legacy for debugging).
    if not (os.getenv("PIPELINE_CALCULATOR_IMPL") or "").strip():
        os.environ["PIPELINE_CALCULATOR_IMPL"] = "new"


if __name__ == "__main__":
    _force_modular_default()
    raise SystemExit(main())

