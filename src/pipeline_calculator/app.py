"""Application entrypoint for the refactor-in-progress package.

Design goals:
- Keep `src/pipeline_calculator_v3.py` as the stable, legacy implementation.
- Allow developers/CI to run the new package entrypoint without breaking builds.
- Provide a controlled switch (`PIPELINE_CALCULATOR_IMPL`) with safe fallback to legacy.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable


def _impl_from_env() -> str:
    # "legacy" is the safe default until the refactor is complete.
    return (os.getenv("PIPELINE_CALCULATOR_IMPL") or "legacy").strip().lower()


def resolve_entrypoint() -> Callable[[], int]:
    impl = _impl_from_env()

    if impl in ("legacy", "v3", "monolith"):
        from pipeline_calculator.legacy import load_legacy_main

        return load_legacy_main()

    if impl in ("new", "refactor", "package"):
        # Future: point this at the refactored app entrypoint.
        # Keep the fallback so developers can opt-in without getting stuck.
        raise NotImplementedError("Refactored implementation not wired yet.")

    raise ValueError(
        f"Unknown PIPELINE_CALCULATOR_IMPL={impl!r}. "
        "Use 'legacy' (default) or 'new' (future)."
    )


def main() -> int:
    try:
        entrypoint = resolve_entrypoint()
    except NotImplementedError as e:
        print(f"[pipeline_calculator] {e} Falling back to legacy implementation.", file=sys.stderr)
        from pipeline_calculator.legacy import load_legacy_main

        entrypoint = load_legacy_main()
    return int(entrypoint() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

