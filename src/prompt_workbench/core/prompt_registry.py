"""Loads prompt templates by name.

The one place that knows where prompt templates live and how to read them.
Callers ask for a template by name and get back its system text; nothing else in
the app opens a YAML file. Keeping templates as data (YAML) rather than code
makes them easy to add, compare, and swap without touching the application.
"""

from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_system_prompt(name: str, *, prompts_dir: str | Path | None = None) -> str:
    """Return the ``system`` text from ``<name>.yaml``.

    ``prompts_dir`` is injectable so tests can point at a temp directory;
    it defaults to the package's ``prompts/`` folder.

    Raises ``FileNotFoundError`` if the template file is missing and
    ``ValueError`` if it has no ``system`` field.
    """
    resolved_dir = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    path = resolved_dir / f"{name}.yaml"

    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    system = data.get("system")
    if not system:
        raise ValueError(f"Prompt '{name}' is missing a 'system' field")
    return system


def available_prompts(*, prompts_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Return metadata for every prompt template on disk.

    Each entry is a dict with ``name`` (the file stem, usable with
    ``load_system_prompt``), ``title`` (human label) and ``technique``.
    Discovery is driven by the ``prompts/`` folder, so adding a YAML file
    makes a new template available with no code change. Sorted by name for a
    stable UI ordering; an empty folder yields an empty list.
    """
    resolved_dir = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    prompts: list[dict[str, str]] = []
    for path in sorted(resolved_dir.glob("*.yaml")):
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        prompts.append(
            {
                "name": path.stem,
                "title": data.get("name", path.stem),
                "technique": data.get("technique", ""),
            }
        )
    return prompts
