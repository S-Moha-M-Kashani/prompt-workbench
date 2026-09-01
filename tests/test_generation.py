"""Discovery, dataset generation, and candidate generation.

Every model call goes through an injected fake, so these assert on what the
workbench *sends* and how it reads what comes back — including the rule that a
few-shot candidate may only be shown cases the user can see.
"""

import json
from datetime import UTC, datetime

import pytest

from prompt_workbench.core import candidate_generation, dataset_generation, discovery
from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.generation import GenerationError
from prompt_workbench.models import (
    BriefSnapshot,
    CaseCategory,
    GroundTruthCase,
    GroundTruthDataset,
    ModelSettings,
    PromptBrief,
    PromptTechnique,
    SourceRef,
    sequential_ids,
)

WHEN = datetime(2026, 3, 1, tzinfo=UTC)
PLATFORM = "You are a prompt engineer."


def a_brief() -> BriefSnapshot:
    return BriefSnapshot(
        id="brief-1",
        revision=1,
        brief=PromptBrief(
            purpose="Summarize incoming support tickets",
            audience="Team leads",
            output_format="Three bullet points",
        ),
        created_at=WHEN,
    )


def a_dataset(cases: tuple[GroundTruthCase, ...] | None = None) -> GroundTruthDataset:
    default = (
        GroundTruthCase(
            id="case-1",
            test_message="My package is late.",
            required_criteria=("names the delay",),
            reference_answer="Your package is delayed.",
        ),
    )
    return GroundTruthDataset(
        id="ds-1",
        revision=1,
        source_brief=SourceRef(id="brief-1", revision=1),
        cases=cases if cases is not None else default,
        created_at=WHEN,
    )


def replying(*replies: str) -> tuple[object, list[list[dict[str, str]]]]:
    """A fake completion returning each reply in turn, plus a call log."""
    sent: list[list[dict[str, str]]] = []
    queue = list(replies)

    def complete(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        sent.append(messages)
        return queue.pop(0) if queue else replies[-1]

    return complete, sent


# --- discovery ------------------------------------------------------------


def test_a_clarification_reply_updates_the_draft_brief_and_asks_one_question() -> None:
    store = ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)
    thread_id = store.open("discovery")
    complete, sent = replying(
        json.dumps(
            {
                "reply": "Who reads these summaries?",
                "brief": {"purpose": "Summarize support tickets"},
                "ready": False,
            }
        )
    )
    result = discovery.clarify(
        store=store,
        thread_id=thread_id,
        user_message="I need summaries of support tickets",
        brief=PromptBrief(),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
    )
    assert result.reply == "Who reads these summaries?"
    assert result.brief.purpose == "Summarize support tickets"
    assert result.ready is False
    assert PLATFORM in sent[0][0]["content"]


def test_the_discovery_call_carries_the_whole_thread_and_records_both_turns() -> None:
    store = ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)
    thread_id = store.open("discovery")
    reply = json.dumps({"reply": "noted", "brief": {}, "ready": False})
    complete, sent = replying(reply, reply)
    for message in ("first", "second"):
        discovery.clarify(
            store=store,
            thread_id=thread_id,
            user_message=message,
            brief=PromptBrief(),
            platform_instruction=PLATFORM,
            complete=complete,  # type: ignore[arg-type]
            model="m",
            settings=ModelSettings(),
        )
    assert [m.content for m in store.history(thread_id)] == [
        "first",
        "noted",
        "second",
        "noted",
    ]
    # The second call saw the first exchange.
    assert any("first" in m["content"] for m in sent[1])


def test_discovery_tells_the_model_which_brief_fields_are_still_blank() -> None:
    store = ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)
    thread_id = store.open("discovery")
    complete, sent = replying(json.dumps({"reply": "ok", "brief": {}, "ready": False}))
    discovery.clarify(
        store=store,
        thread_id=thread_id,
        user_message="hi",
        brief=PromptBrief(purpose="already known"),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
    )
    system = sent[0][0]["content"]
    assert "audience" in system
    assert "already known" in system


def test_a_discovery_reply_that_is_not_json_fails_without_losing_the_brief() -> None:
    store = ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)
    thread_id = store.open("discovery")
    complete, _ = replying("I am afraid I cannot do that.")
    with pytest.raises(GenerationError, match="could not"):
        discovery.clarify(
            store=store,
            thread_id=thread_id,
            user_message="hi",
            brief=PromptBrief(purpose="keep me"),
            platform_instruction=PLATFORM,
            complete=complete,  # type: ignore[arg-type]
            model="m",
            settings=ModelSettings(),
        )
    # The failed exchange left no assistant turn behind.
    assert [m.role for m in store.history(thread_id)] == ["user"]


# --- dataset generation ---------------------------------------------------


def dataset_reply(count: int = 2) -> str:
    return json.dumps(
        {
            "cases": [
                {
                    "test_message": f"message {i}",
                    "required_criteria": ["is specific"],
                    "forbidden_behaviours": ["invents a policy"],
                    "tags": ["t"],
                    "reference_answer": None,
                    "category": ["normal", "edge", "failure"][i % 3],
                }
                for i in range(count)
            ]
        }
    )


def test_dataset_generation_reads_only_the_confirmed_brief_snapshot() -> None:
    complete, sent = replying(dataset_reply())
    dataset_generation.generate(
        brief=a_brief(),
        count=2,
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    prompt = "\n".join(m["content"] for m in sent[0])
    assert "Summarize incoming support tickets" in prompt
    assert "Team leads" in prompt
    # No conversation history is smuggled into a stateless generation.
    assert len(sent[0]) == 2


def test_generated_cases_become_a_dataset_pinned_to_that_brief() -> None:
    complete, _ = replying(dataset_reply(3))
    dataset = dataset_generation.generate(
        brief=a_brief(),
        count=3,
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert len(dataset) == 3
    assert dataset.source_brief == SourceRef(id="brief-1", revision=1)
    assert {case.category for case in dataset} == {
        CaseCategory.NORMAL,
        CaseCategory.EDGE,
        CaseCategory.FAILURE,
    }


def test_the_requested_case_count_reaches_the_model() -> None:
    complete, sent = replying(dataset_reply())
    dataset_generation.generate(
        brief=a_brief(),
        count=9,
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert "9" in sent[0][1]["content"]


def test_a_generated_case_missing_a_criterion_is_reported_not_dropped() -> None:
    complete, _ = replying(json.dumps({"cases": [{"test_message": "hi", "required_criteria": []}]}))
    with pytest.raises(GenerationError, match="required criterion"):
        dataset_generation.generate(
            brief=a_brief(),
            count=1,
            platform_instruction=PLATFORM,
            complete=complete,  # type: ignore[arg-type]
            model="m",
            settings=ModelSettings(),
            new_id=sequential_ids(),
            clock=lambda: WHEN,
        )


def test_a_dataset_reply_with_no_cases_fails_rather_than_returning_an_empty_set() -> None:
    complete, _ = replying(json.dumps({"cases": []}))
    with pytest.raises(GenerationError, match="no test cases"):
        dataset_generation.generate(
            brief=a_brief(),
            count=1,
            platform_instruction=PLATFORM,
            complete=complete,  # type: ignore[arg-type]
            model="m",
            settings=ModelSettings(),
            new_id=sequential_ids(),
            clock=lambda: WHEN,
        )


# --- candidate generation -------------------------------------------------


def test_one_candidate_is_generated_per_requested_technique() -> None:
    complete, _ = replying("You are a helpful summarizer.")
    candidates = candidate_generation.generate(
        brief=a_brief(),
        dataset=a_dataset(),
        techniques=(PromptTechnique.DIRECT, PromptTechnique.ROLE_BASED),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert [c.technique for c in candidates] == [
        PromptTechnique.DIRECT,
        PromptTechnique.ROLE_BASED,
    ]
    assert all(c.source_brief == SourceRef(id="brief-1", revision=1) for c in candidates)


def test_each_technique_is_generated_with_its_own_template() -> None:
    complete, sent = replying("prompt text")
    candidate_generation.generate(
        brief=a_brief(),
        dataset=a_dataset(),
        techniques=(PromptTechnique.DIRECT, PromptTechnique.REASONING_GUIDED),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert "DIRECT" in sent[0][0]["content"]
    assert "REASONING-GUIDED" in sent[1][0]["content"]


def test_a_few_shot_candidate_is_shown_the_visible_dataset() -> None:
    complete, sent = replying("prompt text")
    candidate_generation.generate(
        brief=a_brief(),
        dataset=a_dataset(),
        techniques=(PromptTechnique.FEW_SHOT,),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    prompt = "\n".join(m["content"] for m in sent[0])
    assert "My package is late." in prompt
    assert "Your package is delayed." in prompt


def test_a_candidate_that_needs_no_examples_is_not_given_the_dataset() -> None:
    """A direct candidate shown the test cases would be quietly few-shot, and
    the comparison between the two techniques would measure nothing."""
    complete, sent = replying("prompt text")
    candidate_generation.generate(
        brief=a_brief(),
        dataset=a_dataset(),
        techniques=(PromptTechnique.DIRECT,),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    prompt = "\n".join(m["content"] for m in sent[0])
    assert "My package is late." not in prompt


def test_few_shot_generation_without_a_dataset_is_refused_before_the_call() -> None:
    def explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("few-shot must not be generated without visible examples")

    with pytest.raises(GenerationError, match="test case"):
        candidate_generation.generate(
            brief=a_brief(),
            dataset=None,
            techniques=(PromptTechnique.FEW_SHOT,),
            platform_instruction=PLATFORM,
            complete=explode,  # type: ignore[arg-type]
            model="m",
            settings=ModelSettings(),
            new_id=sequential_ids(),
            clock=lambda: WHEN,
        )


def test_a_candidate_records_the_dataset_revision_it_was_built_from() -> None:
    complete, _ = replying("prompt text")
    candidates = candidate_generation.generate(
        brief=a_brief(),
        dataset=a_dataset(),
        techniques=(PromptTechnique.FEW_SHOT,),
        platform_instruction=PLATFORM,
        complete=complete,  # type: ignore[arg-type]
        model="m",
        settings=ModelSettings(),
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert candidates[0].source_dataset == SourceRef(id="ds-1", revision=1)


def test_an_empty_candidate_reply_is_an_error_not_a_blank_prompt() -> None:
    complete, _ = replying("   ")
    with pytest.raises(GenerationError, match="empty"):
        candidate_generation.generate(
            brief=a_brief(),
            dataset=a_dataset(),
            techniques=(PromptTechnique.DIRECT,),
            platform_instruction=PLATFORM,
            complete=complete,  # type: ignore[arg-type]
            model="m",
            settings=ModelSettings(),
            new_id=sequential_ids(),
            clock=lambda: WHEN,
        )


def test_generation_never_calls_a_judge() -> None:
    """Nothing in this module may reach the evaluator; scoring is a decision."""
    import inspect

    for module in (discovery, dataset_generation, candidate_generation):
        source = inspect.getsource(module)
        assert "judge" not in source.lower(), module.__name__
        assert "metric_adapters" not in source, module.__name__
