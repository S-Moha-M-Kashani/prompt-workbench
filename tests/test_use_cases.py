"""Ready use cases: a situation, a prompt, mocked context, and what good means."""

import pytest

from prompt_workbench.models.use_case import MockBlock, UseCase
from prompt_workbench.services import use_case_catalog


def a_use_case(**overrides: object) -> UseCase:
    fields: dict[str, object] = {
        "key": "grounded_briefing",
        "label": "Grounded briefing",
        "family": "grounding",
        "situation": "You brief a tutor before it answers a learner.",
        "trap": "Inventing a history that was never supplied.",
        "system_prompt": "Use only what is in HISTORY:\n{history}",
        "mocks": (MockBlock(name="history", label="Learner history", content="(empty)"),),
        "example_message": "What should I know about this learner?",
        "criteria": ("says there is no history",),
        "forbidden": ("invents a past event",),
    }
    fields.update(overrides)
    return UseCase(**fields)  # type: ignore[arg-type]


# --- the model ------------------------------------------------------------


def test_a_mock_block_is_named_after_the_placeholder_it_fills() -> None:
    block = MockBlock(name="history", label="Learner history", content="nothing yet")
    assert block.placeholder == "{history}"


def test_mocks_are_substituted_into_the_prompt_under_test() -> None:
    filled = a_use_case().filled_prompt("Use only what is in HISTORY:\n{history}")
    assert filled == "Use only what is in HISTORY:\n(empty)"


def test_substitution_leaves_other_braces_alone() -> None:
    """The prompts under test contain JSON shapes. A format()-style pass would
    choke on them, so only known placeholders are replaced."""
    filled = a_use_case().filled_prompt('Reply {"score": 1} using {history} and {unknown}')
    assert filled == 'Reply {"score": 1} using (empty) and {unknown}'


def test_a_use_case_reports_which_placeholders_its_prompt_still_needs() -> None:
    case = a_use_case()
    assert case.unfilled_placeholders("no placeholders here") == ()
    assert case.unfilled_placeholders("{history} and {candidates}") == ("candidates",)


def test_the_situation_message_names_the_trap_and_the_mocks() -> None:
    message = a_use_case().as_situation_message()
    assert "You brief a tutor" in message
    assert "Inventing a history" in message
    assert "Learner history" in message


def test_a_use_case_becomes_a_ground_truth_case_for_scoring() -> None:
    case = a_use_case().as_ground_truth_case("case-1", user_message="what do you know?")
    assert case.id == "case-1"
    assert case.test_message == "what do you know?"
    assert case.required_criteria == ("says there is no history",)
    assert case.forbidden_behaviours == ("invents a past event",)


def test_a_use_case_needs_criteria_to_be_scoreable() -> None:
    with pytest.raises(ValueError, match="criteri"):
        a_use_case(criteria=())


def test_a_use_case_needs_a_non_empty_starting_prompt() -> None:
    with pytest.raises(ValueError, match="system prompt"):
        a_use_case(system_prompt="   ")


# --- the shipped catalogue ------------------------------------------------


def test_the_catalogue_ships_ten_use_cases() -> None:
    assert len(use_case_catalog.all_use_cases()) == 10


def test_every_use_case_key_and_label_is_unique() -> None:
    cases = use_case_catalog.all_use_cases()
    assert len({c.key for c in cases}) == len(cases)
    assert len({c.label for c in cases}) == len(cases)


def test_every_use_case_is_fully_specified() -> None:
    for case in use_case_catalog.all_use_cases():
        assert case.situation.strip(), case.key
        assert case.trap.strip(), case.key
        assert case.system_prompt.strip(), case.key
        assert case.criteria, case.key
        assert case.example_message.strip(), case.key


def test_every_placeholder_in_a_shipped_prompt_has_a_mock_behind_it() -> None:
    """A prompt with an unfilled placeholder would reach the model with a
    literal {history} in it, which is a bug the user would have to debug."""
    for case in use_case_catalog.all_use_cases():
        assert case.unfilled_placeholders(case.system_prompt) == (), case.key


def test_the_catalogue_covers_several_families_of_prompt_job() -> None:
    families = {case.family for case in use_case_catalog.all_use_cases()}
    assert len(families) >= 4, families


def test_one_use_case_hunts_prompt_injection() -> None:
    """The mocked knowledge base must actually contain the injection, or the
    use case tests nothing."""
    case = use_case_catalog.get("injection_resistance")
    mocked = " ".join(block.content for block in case.mocks).lower()
    assert "ignore" in mocked or "instruction" in mocked


def test_the_grounded_briefing_use_case_supplies_an_empty_history() -> None:
    """Its whole point: the model is given nothing and must say so."""
    case = use_case_catalog.get("grounded_briefing")
    history = next(b for b in case.mocks if "histor" in b.name.lower())
    assert history.content.strip() in ("", "(none)", "(empty)") or len(history.content) < 40


def test_an_unknown_use_case_key_raises() -> None:
    with pytest.raises(KeyError):
        use_case_catalog.get("no_such_use_case")
