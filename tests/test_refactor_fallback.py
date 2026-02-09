from __future__ import annotations

import os


def test_refactor_entrypoint_defaults_to_legacy() -> None:
    # Ensure the refactor package can resolve to legacy without executing the GUI.
    os.environ.pop("PIPELINE_CALCULATOR_IMPL", None)
    from pipeline_calculator.app import resolve_entrypoint

    entry = resolve_entrypoint()
    assert callable(entry)


def test_refactor_entrypoint_unknown_impl_raises() -> None:
    os.environ["PIPELINE_CALCULATOR_IMPL"] = "definitely-not-a-real-impl"
    from pipeline_calculator.app import resolve_entrypoint

    try:
        resolve_entrypoint()
        assert False, "expected ValueError"
    except ValueError:
        pass

