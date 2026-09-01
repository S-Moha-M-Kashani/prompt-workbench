import tomllib
from pathlib import Path

import prompt_workbench


def test_version_is_read_from_pyproject():
    """__version__ must match pyproject.toml (single source of truth),
    not drift as a separately hardcoded value."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        expected = tomllib.load(f)["project"]["version"]

    assert prompt_workbench.__version__ == expected
