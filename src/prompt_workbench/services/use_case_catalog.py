"""Loads the ready use cases from disk.

They are YAML rather than Python for the same reason the prompt templates are:
a use case is prose and mock data, and prose in a source file invites editing it
like code. A new situation is a new file — no import to add, no tuple to extend.

Order is stable and grouped by family, so the dropdown reads as a progression
rather than an alphabet.
"""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from prompt_workbench.models.use_case import MockBlock, UseCase

USE_CASES_DIR = Path(__file__).resolve().parent.parent / "use_cases"

# Families in the order they appear in the dropdown. Grounding first: it is the
# failure most prompts have and the easiest to see happening.
FAMILY_ORDER: tuple[str, ...] = ("grounding", "structure", "state", "agentic", "safety")

FAMILY_LABELS: dict[str, str] = {
    "grounding": "Grounding",
    "structure": "Structured output",
    "state": "Running state",
    "agentic": "Agent loop",
    "safety": "Safety",
}


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _load_one(path: Path) -> UseCase:
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    mocks = tuple(
        MockBlock(
            name=str(block["name"]),
            label=str(block.get("label", block["name"])),
            content=str(block.get("content", "")).rstrip(),
            note=str(block.get("note", "")),
        )
        for block in data.get("mocks", [])
    )
    return UseCase(
        key=str(data.get("key", path.stem)),
        label=str(data["label"]),
        family=str(data.get("family", "grounding")),
        situation=str(data.get("situation", "")).strip(),
        trap=str(data.get("trap", "")).strip(),
        system_prompt=str(data.get("system_prompt", "")).strip(),
        mocks=mocks,
        example_message=str(data.get("example_message", "")).strip(),
        criteria=_as_tuple(data.get("criteria")),
        forbidden=_as_tuple(data.get("forbidden")),
        origin=str(data.get("origin", "")).strip(),
        notes=str(data.get("notes", "")).strip(),
    )


@cache
def all_use_cases() -> tuple[UseCase, ...]:
    """Every shipped use case, grouped by family in dropdown order."""
    loaded = [_load_one(path) for path in sorted(USE_CASES_DIR.glob("*.yaml"))]

    def sort_key(case: UseCase) -> tuple[int, str]:
        family = FAMILY_ORDER.index(case.family) if case.family in FAMILY_ORDER else 99
        return (family, case.label)

    return tuple(sorted(loaded, key=sort_key))


def get(key: str) -> UseCase:
    """One use case by key; raises ``KeyError`` for an unknown one."""
    for case in all_use_cases():
        if case.key == key:
            return case
    raise KeyError(f"No use case {key!r}")


def grouped() -> dict[str, tuple[UseCase, ...]]:
    """Use cases by family label, for a grouped dropdown."""
    groups: dict[str, list[UseCase]] = {}
    for case in all_use_cases():
        groups.setdefault(FAMILY_LABELS.get(case.family, case.family), []).append(case)
    return {label: tuple(cases) for label, cases in groups.items()}
