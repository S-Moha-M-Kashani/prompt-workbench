"""Starting kits: what a kind of job fills the workspace in with.

A kit is the answer to the blank page, not the answer to the case. It supplies a
system prompt, a user prompt, a tool set, an answer shape, metrics with
thresholds, a starting temperature and the approaches worth trying — and every
one of those stays editable, because the kit knows the shape of the job and
nothing at all about the user's own situation.

The content lives as YAML under ``prompts/presets/``, for the same reason the
workbench's own instructions do: a prompt is prose, and prose in a source file
gets edited like code and reviewed like code. Adding a kit is a new file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from prompt_workbench.models.call import OutputStructure, ToolSpec

PRESETS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "presets"


@dataclass(frozen=True)
class StartingKit:
    """One kind of job's editable starting point."""

    key: str
    label: str
    system_prompt: str
    user_prompt: str
    note: str = ""
    tools: tuple[ToolSpec, ...] = ()
    output_structure: OutputStructure | None = None
    #: Metric key to threshold. A metric without a threshold measures nothing.
    metrics: dict[str, float] = field(default_factory=dict)
    temperature: float | None = None
    approach_labels: tuple[str, ...] = ()


@cache
def load(key: str, *, presets_dir: str | Path | None = None) -> StartingKit:
    """The kit named ``key``, or a refusal naming it.

    Cached because a kit is immutable data read many times per render, and the
    files do not change while the app runs.
    """
    resolved = Path(presets_dir) if presets_dir is not None else PRESETS_DIR
    path = resolved / f"{key}.yaml"
    if not path.is_file():
        raise KeyError(f"No starting kit {key!r} in {resolved}")
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return _as_kit(key, data)


def _as_kit(key: str, data: dict[str, Any]) -> StartingKit:
    shape = data.get("output_structure") or None
    return StartingKit(
        key=key,
        label=str(data.get("label", key)),
        system_prompt=str(data.get("system_prompt", "")).rstrip(),
        user_prompt=str(data.get("user_prompt", "")).rstrip(),
        note=str(data.get("note", "")).rstrip(),
        tools=tuple(
            ToolSpec(
                name=str(tool["name"]),
                description=str(tool.get("description", "")),
                input_schema=tool.get("input_schema") or None,
            )
            for tool in data.get("tools", [])
        ),
        output_structure=(
            OutputStructure(
                name=str(shape.get("name", key)), schema=dict(shape.get("schema", {}))
            )
            if shape
            else None
        ),
        metrics={str(k): float(v) for k, v in (data.get("metrics") or {}).items()},
        temperature=(
            float(data["temperature"]) if data.get("temperature") is not None else None
        ),
        approach_labels=tuple(str(label) for label in data.get("approach_labels", [])),
    )


def available() -> tuple[str, ...]:
    """Every kit on disk, by key."""
    return tuple(sorted(path.stem for path in PRESETS_DIR.glob("*.yaml")))
