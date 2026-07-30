"""Conventions for generated project artifacts."""

from __future__ import annotations

from pathlib import Path


OUTPUT_DIRECTORY = Path("out")


def prepare_output_path(path: Path) -> Path:
    """Put relative generated artifacts in ``out/`` and create their folder.

    An explicit absolute path remains untouched. Passing ``out/name.glb`` also
    remains untouched rather than becoming ``out/out/name.glb``.
    """
    if path.is_absolute() or (path.parts and path.parts[0].lower() == OUTPUT_DIRECTORY.name):
        result = path
    else:
        result = OUTPUT_DIRECTORY / path
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
