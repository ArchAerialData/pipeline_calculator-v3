from __future__ import annotations

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

from pipeline_calculator.app import main


if __name__ == "__main__":
    raise SystemExit(main())

