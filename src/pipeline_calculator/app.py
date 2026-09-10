"""Application entrypoint for Pipeline Calculator.

Design goals:
- Keep `src/pipeline_calculator_v3.py` available as a legacy compatibility entrypoint.
- Default to the modular v4 GUI while retaining an explicit legacy switch.
- Provide a controlled switch (`PIPELINE_CALCULATOR_IMPL`) with safe fallback to legacy.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable


def _impl_from_env() -> str:
    return (os.getenv("PIPELINE_CALCULATOR_IMPL") or "new").strip().lower()


def resolve_entrypoint() -> Callable[[], int]:
    impl = _impl_from_env()

    if impl in ("legacy", "v3", "monolith"):
        from pipeline_calculator.legacy import load_legacy_main

        return load_legacy_main()

    if impl in ("new", "refactor", "package"):
        from pipeline_calculator.gui.main_window import main as gui_main

        return gui_main

    raise ValueError(
        f"Unknown PIPELINE_CALCULATOR_IMPL={impl!r}. "
        "Use 'new' (default) or 'legacy'."
    )


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == '--smoke-test':
        from pipeline_calculator.smoke import run
        return run(sys.argv[2], implementation='legacy' if _impl_from_env() in ('legacy', 'v3', 'monolith') else 'new')
    try:
        entrypoint = resolve_entrypoint()
    except (ImportError, NotImplementedError) as e:
        print(
            f"[pipeline_calculator] {type(e).__name__}: {e} Falling back to legacy implementation.",
            file=sys.stderr,
        )
        from pipeline_calculator.legacy import load_legacy_main

        entrypoint = load_legacy_main()
    return int(entrypoint() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
