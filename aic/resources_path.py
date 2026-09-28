"""Locate bundled resources both from source and inside the PyInstaller onedir build."""
import sys
from pathlib import Path


def res(*parts) -> Path:
    base = Path(getattr(sys, "_MEIPASS", "")) / "aic" / "resources"
    if getattr(sys, "_MEIPASS", None) and base.exists():
        return base.joinpath(*parts)
    return Path(__file__).resolve().parent.joinpath("resources", *parts)
