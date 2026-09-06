"""The framework list: what is available here, and what to type if it is not."""

import pytest

from prompt_workbench.llm_call import registry


def test_every_framework_is_listed_whether_or_not_it_is_installed() -> None:
    keys = [entry.key for entry in registry.all_frameworks()]
    assert "openai" in keys
    assert keys == sorted(set(keys), key=keys.index), "no duplicate keys"


def test_the_bare_sdk_is_always_available() -> None:
    assert registry.get("openai").is_available() is True


def test_an_unavailable_framework_names_its_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = registry.FrameworkEntry(
        key="ghost",
        label="Ghost",
        install_hint="uv sync --extra ghost",
        import_names=("a_package_that_is_not_installed_anywhere",),
        builder=lambda **kwargs: None,  # type: ignore[arg-type,return-value]
    )
    assert entry.is_available() is False
    assert "uv sync --extra ghost" in entry.unavailable_reason()


def test_availability_is_decided_at_call_time_not_at_import() -> None:
    source = (registry.__file__ or "").replace("registry.py", "registry.py")
    text = open(source).read()
    for framework in ("langchain", "langgraph", "anthropic"):
        assert f"import {framework}" not in text, f"{framework} imported at module level"


def test_an_unknown_framework_key_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="nonsense"):
        registry.get("nonsense")


def test_available_keys_are_a_subset_of_all_keys() -> None:
    available = set(registry.available_keys())
    assert available <= {entry.key for entry in registry.all_frameworks()}
    assert "openai" in available


def test_the_optional_frameworks_are_registered_with_install_commands() -> None:
    for key in ("langchain", "langgraph"):
        entry = registry.get(key)
        assert entry.import_names, f"{key} must be checked for presence"
        assert entry.install_hint.startswith("uv sync --extra")


def test_an_uninstalled_extra_disables_only_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the extra being absent by making its spec lookup fail."""
    import importlib.util

    real = importlib.util.find_spec

    def missing(name: str, *args: object, **kwargs: object):
        if name == "langgraph":
            return None
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(registry.importlib.util, "find_spec", missing)

    assert registry.get("langgraph").is_available() is False
    assert "uv sync --extra langgraph" in registry.get("langgraph").unavailable_reason()
    assert registry.get("openai").is_available() is True
    with pytest.raises(ModuleNotFoundError, match="langgraph"):
        registry.build("langgraph")


def test_every_framework_says_what_it_forwards() -> None:
    for entry in registry.all_frameworks():
        assert entry.forwards, f"{entry.key} does not say what it carries through"
