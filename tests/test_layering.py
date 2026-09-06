"""The layering rule, read off the source rather than trusted.

``core/`` and ``models/`` are the vendor-free rings: no Streamlit, no provider
SDK, no framework SDK. deepeval is not on that list — it is the metric library,
imported lazily by ``core/scoring.py`` precisely because it is optional.
Framework libraries live only in ``llm_call/``, which is the one place besides
``services/openrouter_client`` that opens a connection.

Imports are read with ``ast`` rather than by importing the modules, so the rule
is checked against what the file says and not against which packages happen to
be installed here.
"""

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "src" / "prompt_workbench"

# Packages no vendor-free module may name.
FORBIDDEN_IN_CORE = frozenset(
    {
        "streamlit",
        "openai",
        "anthropic",
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langgraph",
    }
)

# Only these rings may import a framework SDK.
FRAMEWORK_PACKAGES = frozenset(
    {"openai", "anthropic", "langchain", "langchain_openai", "langchain_core", "langgraph"}
)
FRAMEWORK_RINGS = ("llm_call", "services")


def _top_level_imports(path: Path) -> set[str]:
    """Every root package named by an import in ``path``, lazy ones included."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _modules_under(*rings: str) -> list[Path]:
    return sorted(p for ring in rings for p in (SOURCE_ROOT / ring).rglob("*.py"))


@pytest.mark.parametrize("path", _modules_under("core", "models"), ids=lambda p: p.name)
def test_core_and_models_name_no_vendor(path: Path) -> None:
    offenders = _top_level_imports(path) & FORBIDDEN_IN_CORE
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _modules_under("ui"), ids=lambda p: p.name)
def test_the_ui_names_no_framework_sdk(path: Path) -> None:
    offenders = _top_level_imports(path) & FRAMEWORK_PACKAGES
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


def test_framework_sdks_appear_only_in_the_rings_allowed_to_open_connections() -> None:
    for path in SOURCE_ROOT.rglob("*.py"):
        if not _top_level_imports(path) & FRAMEWORK_PACKAGES:
            continue
        ring = path.relative_to(SOURCE_ROOT).parts[0]
        assert ring in FRAMEWORK_RINGS, f"{path} is outside {FRAMEWORK_RINGS}"
