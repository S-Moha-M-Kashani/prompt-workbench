"""Running a candidate by hand, and what every response has to record."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.core import testing
from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.models import (
    BriefSnapshot,
    CandidatePrompt,
    GroundTruthCase,
    ModelSettings,
    PromptBrief,
    PromptTechnique,
    SourceRef,
    ThreadConfig,
    sequential_ids,
)

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_brief() -> BriefSnapshot:
    return BriefSnapshot(id="brief-1", revision=1, brief=PromptBrief(purpose="x"), created_at=WHEN)


def a_candidate(revision: int = 1) -> CandidatePrompt:
    return CandidatePrompt(
        id="cand-1",
        technique=PromptTechnique.DIRECT,
        system_prompt="You are a support assistant.",
        revision=revision,
        source_brief=SourceRef(id="brief-1", revision=1),
        created_at=WHEN,
    )


def a_case() -> GroundTruthCase:
    return GroundTruthCase(
        id="case-1", test_message="My order is late.", required_criteria=("acknowledges it",)
    )


def replying(text: str = "I am sorry about that."):  # type: ignore[no-untyped-def]
    sent: list[list[dict[str, str]]] = []

    def complete(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        sent.append(messages)
        return text

    return complete, sent


def a_store() -> ThreadStore:
    return ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)


def config(revision: int = 1, model_id: str = "openai/gpt-4o-mini") -> ThreadConfig:
    return ThreadConfig(
        candidate=SourceRef(id="cand-1", revision=revision),
        model_id=model_id,
        settings=ModelSettings(),
    )


def send(store: ThreadStore, thread_id: str, complete, **overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(
        store=store,
        thread_id=thread_id,
        candidate=a_candidate(),
        brief=a_brief(),
        case=a_case(),
        user_message="My order is late.",
        model="openai/gpt-4o-mini",
        settings=ModelSettings(),
        complete=complete,
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    kwargs.update(overrides)
    return testing.send_test_message(**kwargs)  # type: ignore[arg-type]


def test_the_candidate_prompt_is_sent_as_the_system_message() -> None:
    store = a_store()
    thread_id = store.open("test", config=config())
    complete, sent = replying()
    send(store, thread_id, complete)
    assert sent[0][0] == {"role": "system", "content": "You are a support assistant."}


def test_the_response_is_recorded_with_every_identifier_needed_to_score_it() -> None:
    store = a_store()
    thread_id = store.open("test", config=config())
    complete, _ = replying()
    record = send(store, thread_id, complete)
    assert record.candidate == SourceRef(id="cand-1", revision=1)
    assert record.source_brief == SourceRef(id="brief-1", revision=1)
    assert record.case_id == "case-1"
    assert record.thread_id == thread_id
    assert record.model_id == "openai/gpt-4o-mini"
    assert record.is_attributable


def test_the_thread_keeps_its_full_history_across_turns() -> None:
    store = a_store()
    thread_id = store.open("test", config=config())
    complete, sent = replying()
    send(store, thread_id, complete)
    send(store, thread_id, complete, user_message="Any update?")
    assert len(store.history(thread_id)) == 4
    # The second call replayed the whole conversation under the same system prompt.
    assert [m["role"] for m in sent[1]] == ["system", "user", "assistant", "user"]


def test_free_text_is_allowed_but_produces_an_unattributable_record() -> None:
    store = a_store()
    thread_id = store.open("test", config=config())
    complete, _ = replying()
    record = send(store, thread_id, complete, case=None, user_message="just chatting")
    assert record.case_id is None
    assert not record.is_attributable


def test_an_empty_model_response_is_reported_and_records_nothing_scoreable() -> None:
    store = a_store()
    thread_id = store.open("test", config=config())
    complete, _ = replying("   ")
    with pytest.raises(testing.TestRunFailed, match="empty"):
        send(store, thread_id, complete)


def test_sending_into_a_thread_whose_configuration_moved_on_is_refused() -> None:
    store = a_store()
    thread_id = store.open("test", config=config(revision=1))
    complete, _ = replying()
    with pytest.raises(testing.ConfigurationChanged, match="new thread"):
        send(store, thread_id, complete, candidate=a_candidate(revision=2))


def test_a_matching_configuration_is_accepted() -> None:
    store = a_store()
    thread_id = store.open("test", config=config(revision=2))
    complete, _ = replying()
    assert send(store, thread_id, complete, candidate=a_candidate(revision=2))


def test_a_thread_can_be_opened_for_the_current_configuration() -> None:
    store = a_store()
    thread_id = testing.open_test_thread(
        store, candidate=a_candidate(), model="m", settings=ModelSettings()
    )
    assert store.thread(thread_id).config == ThreadConfig(
        candidate=SourceRef(id="cand-1", revision=1), model_id="m", settings=ModelSettings()
    )
