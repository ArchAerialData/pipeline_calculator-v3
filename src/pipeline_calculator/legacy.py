from __future__ import annotations

from collections.abc import Callable


def load_legacy_main() -> Callable[[], int]:
    """Return the legacy GUI entrypoint.

    Import is intentionally inside the function so importing this module does not
    pull GUI dependencies unless the entrypoint is actually needed.
    """
    import pipeline_calculator_v3  # legacy monolith

    return pipeline_calculator_v3.main

