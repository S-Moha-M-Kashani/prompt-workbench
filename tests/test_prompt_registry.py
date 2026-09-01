import pytest

from prompt_workbench.core import prompt_registry


def _write(directory, name: str, body: str) -> None:
    (directory / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_load_system_prompt_returns_the_system_text(tmp_path):
    _write(tmp_path, "direct", "name: Direct\ntechnique: zero-shot\nsystem: |\n  Answer plainly.\n")

    assert prompt_registry.load_system_prompt("direct", prompts_dir=tmp_path) == "Answer plainly.\n"


def test_load_system_prompt_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        prompt_registry.load_system_prompt("absent", prompts_dir=tmp_path)


def test_load_system_prompt_without_system_field_raises_value_error(tmp_path):
    _write(tmp_path, "broken", "name: Broken\ntechnique: none\n")

    with pytest.raises(ValueError, match="missing a 'system' field"):
        prompt_registry.load_system_prompt("broken", prompts_dir=tmp_path)


def test_available_prompts_lists_metadata_sorted_by_name(tmp_path):
    _write(tmp_path, "reasoning", "name: Reasoning\ntechnique: reasoning-guided\nsystem: think\n")
    _write(tmp_path, "direct", "name: Direct\ntechnique: zero-shot\nsystem: answer\n")

    listed = prompt_registry.available_prompts(prompts_dir=tmp_path)

    assert [entry["name"] for entry in listed] == ["direct", "reasoning"]
    assert listed[0] == {"name": "direct", "title": "Direct", "technique": "zero-shot"}


def test_available_prompts_falls_back_to_the_file_stem(tmp_path):
    _write(tmp_path, "unlabelled", "system: answer\n")

    assert prompt_registry.available_prompts(prompts_dir=tmp_path) == [
        {"name": "unlabelled", "title": "unlabelled", "technique": ""}
    ]


def test_available_prompts_of_an_empty_folder_is_empty(tmp_path):
    """The packaged folder ships empty; discovery must not fail on it."""
    assert prompt_registry.available_prompts(prompts_dir=tmp_path) == []
