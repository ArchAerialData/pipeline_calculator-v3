from __future__ import annotations

import platform
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def resource_root() -> Path:
    """Best-effort base directory for runtime resources (icons, etc.)."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS"))  # type: ignore[arg-type]
    # Running from source: this file is under src/pipeline_calculator/gui/.
    return Path(__file__).resolve().parents[3]


def icon_path() -> Path | None:
    """Return the most appropriate icon path for the current platform, if present."""
    root = resource_root()
    system = platform.system()
    if system == "Windows":
        candidate = root / "icon.ico"
    else:
        candidate = root / ("icon.icns" if system == "Darwin" else "icon.ico")
    return candidate if candidate.exists() else None


def set_window_icon(root) -> None:
    """Configure window icon for supported platforms (best-effort)."""
    try:
        icon = icon_path()
        if icon is None:
            return

        system = platform.system()
        if system == "Windows":
            root.iconbitmap(str(icon))
            return

        # For macOS/Linux, use iconphoto. Keep a reference on the root so it
        # doesn't get GC'd.
        from PIL import Image, ImageTk

        img = Image.open(str(icon))
        photo = ImageTk.PhotoImage(img)
        root._pipeline_calc_icon_image = photo  # noqa: SLF001 - stash ref on Tk root
        root.iconphoto(True, photo)
    except Exception:
        # Icon issues should never prevent the app from launching.
        return
