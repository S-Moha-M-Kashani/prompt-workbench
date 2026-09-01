"""The user's own case, and the prompts written for it."""

import json
from datetime import UTC, datetime

import pytest

from prompt_workbench.core import case_intake, variants
from prompt_workbench.core.generation import GenerationError
from prompt_workbench.models.case import CaseBrief, EvalCase
from prompt_workbench.models.identifiers import sequential_ids
from prompt_workbench.services import task_catalog

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def replying(*replies: str):  # type: ignore[no-untyped-def]
    sent: list[list[dict[str, str]]] = []
    queue = list(replies)

    def complete(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        sent.append(messages)
        return queue.pop(0) if queue else replies[-1]

    return complete, sent


CASES_REPLY = json.dumps({
    "cases": [
        {"input": "My card was charged twice", "expected_output": "billing",
         "context": [], "notes": "everyday billing case"},
        {"input": "???", "expected_output": "other", "context": [], "notes": "ambiguous"},
    ]
})


# --- test cases -----------------------------------------------------------


def test_a_case_records_which_fields_it_supplies() -> None:
    bare = EvalCase(id="c1", input="hello")
    assert bare.supplied_fields() == {"input", "actual_output"}
    rich = EvalCase(id="c2", input="hi", expected_output="hey", retrieval_context=("doc",))
    assert {"expected_output", "retrieval_context"} <= rich.supplied_fields()


def test_a_case_needs_an_input() -> None:
    with pytest.raises(ValueError, match="input"):
        EvalCase(id="c1", input="   ")


def test_a_brief_reports_only_the_fields_every_case_supplies() -> None:
    """A metric that can score three cases out of eight is not one you can put a
    threshold on."""
    brief = CaseBrief(
        description="d", task_type_key="classification", created_at=WHEN,
        cases=(
            EvalCase(id="c1", input="a", expected_output="x"),
            EvalCase(id="c2", input="b"),
        ),
    )
    assert "expected_output" not in brief.supplied_fields()


def test_a_brief_with_no_cases_supplies_nothing() -> None:
    brief = CaseBrief(description="d", task_type_key="classification", created_at=WHEN)
    assert brief.supplied_fields() == set()


def test_cases_are_generated_for_the_described_job() -> None:
    complete, sent = replying(CASES_REPLY)
    cases = case_intake.generate_cases(
        description="Sort billing complaints into queues",
        task=task_catalog.get("classification"),
        count=2,
        complete=complete,
        model="m",
        settings=None,
        new_id=sequential_ids(),
    )
    assert len(cases) == 2
    assert cases[0].expected_output == "billing"
    prompt = "\n".join(m["content"] for m in sent[0])
    assert "Sort billing complaints" in prompt
    assert "Classification" in prompt


def test_a_case_reply_with_nothing_in_it_is_reported() -> None:
    complete, _ = replying(json.dumps({"cases": []}))
    with pytest.raises(GenerationError, match="no test cases"):
        case_intake.generate_cases(
            description="d", task=task_catalog.get("classification"), count=2,
            complete=complete, model="m", settings=None, new_id=sequential_ids(),
        )


def test_generating_cases_without_a_description_is_refused_before_the_call() -> None:
    def explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("must not call a model with no description")

    with pytest.raises(GenerationError, match="Describe"):
        case_intake.generate_cases(
            description="  ", task=task_catalog.get("classification"), count=2,
            complete=explode, model="m", settings=None, new_id=sequential_ids(),
        )


# --- variants -------------------------------------------------------------


def a_brief() -> CaseBrief:
    return CaseBrief(
        description="Sort incoming support tickets into one of six queues",
        task_type_key="classification",
        created_at=WHEN,
        cases=(EvalCase(id="c1", input="My card was charged twice", expected_output="billing"),),
        output_format="The queue name alone",
    )


def test_one_prompt_is_written_per_approach() -> None:
    task = task_catalog.get("classification")
    complete, _ = replying("You are a ticket classifier.")
    written = variants.generate(
        case=a_brief(), task=task, approaches=task.variants[:3],
        complete=complete, model="m", settings=None,
        new_id=sequential_ids(), clock=lambda: WHEN,
    )
    assert len(written) == 3
    assert [v.approach_key for v in written] == [a.key for a in task.variants[:3]]


def test_each_variant_is_told_which_approach_to_take() -> None:
    """Several prompts converging on the same shape would measure nothing."""
    task = task_catalog.get("classification")
    complete, sent = replying("prompt text")
    variants.generate(
        case=a_brief(), task=task, approaches=(task.variant("enumerate"), task.variant("schema")),
        complete=complete, model="m", settings=None,
        new_id=sequential_ids(), clock=lambda: WHEN,
    )
    assert "Strict enumeration" in sent[0][1]["content"]
    assert "JSON schema" in sent[1][1]["content"]


def test_the_writer_sees_the_users_own_case_and_format() -> None:
    task = task_catalog.get("classification")
    complete, sent = replying("prompt text")
    variants.generate(
        case=a_brief(), task=task, approaches=(task.variants[0],),
        complete=complete, model="m", settings=None,
        new_id=sequential_ids(), clock=lambda: WHEN,
    )
    prompt = sent[0][1]["content"]
    assert "six queues" in prompt
    assert "The queue name alone" in prompt
    assert "My card was charged twice" in prompt


def test_an_empty_generated_prompt_is_an_error_not_a_blank_variant() -> None:
    task = task_catalog.get("classification")
    complete, _ = replying("   ")
    with pytest.raises(GenerationError, match="empty"):
        variants.generate(
            case=a_brief(), task=task, approaches=(task.variants[0],),
            complete=complete, model="m", settings=None,
            new_id=sequential_ids(), clock=lambda: WHEN,
        )


def test_no_approaches_is_refused_before_any_call() -> None:
    def explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("must not call a model with no approach chosen")

    with pytest.raises(GenerationError, match="at least one approach"):
        variants.generate(
            case=a_brief(), task=task_catalog.get("classification"), approaches=(),
            complete=explode, model="m", settings=None,
            new_id=sequential_ids(), clock=lambda: WHEN,
        )


def test_editing_a_variant_keeps_the_approach_it_came_from() -> None:
    """A hand-edited prompt still belongs to its approach, or the comparison
    stops holding up after one round of editing."""
    task = task_catalog.get("classification")
    complete, _ = replying("original")
    variant = variants.generate(
        case=a_brief(), task=task, approaches=(task.variants[0],),
        complete=complete, model="m", settings=None,
        new_id=sequential_ids(), clock=lambda: WHEN,
    )[0]
    edited = variant.with_text("rewritten by hand", edited_at=WHEN)
    assert edited.approach_key == variant.approach_key
    assert edited.revision == 2 and edited.edited
    assert "edited" in edited.label
