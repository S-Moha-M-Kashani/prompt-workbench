"""The two chat modes, and the asymmetry between them that does the work."""

import json
from datetime import UTC, datetime

import pytest

from prompt_workbench.core import engineer, one_shot
from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.generation import GenerationError
from prompt_workbench.core.one_shot import RunFailed
from prompt_workbench.models import ModelSettings, sequential_ids
from prompt_workbench.services import use_case_catalog

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_use_case():  # type: ignore[no-untyped-def]
    return use_case_catalog.get("grounded_briefing")


def replying(*replies: str):  # type: ignore[no-untyped-def]
    sent: list[list[dict[str, str]]] = []
    queue = list(replies)

    def complete(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        sent.append(messages)
        return queue.pop(0) if queue else replies[-1]

    return complete, sent


def a_store() -> ThreadStore:
    return ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)


# --- prompt engineer mode -------------------------------------------------


def test_the_engineer_is_shown_the_situation_the_trap_and_the_mocks() -> None:
    store = a_store()
    thread = store.open("engineer")
    complete, sent = replying(json.dumps({"reply": "Add an empty-case rule.", "prompt": ""}))
    engineer.discuss(
        store=store,
        thread_id=thread,
        user_message="why does it invent history?",
        use_case=a_use_case(),
        prompt_under_test="Use {history}",
        complete=complete,
        model="m",
        settings=ModelSettings(),
    )
    system = sent[0][0]["content"]
    assert "A second model is about to answer" in system
    assert "Filling the empty space" in system
    assert "(no matching history)" in system
    assert "Use {history}" in system


def test_the_engineer_can_return_a_complete_replacement_prompt() -> None:
    store = a_store()
    thread = store.open("engineer")
    complete, _ = replying(
        json.dumps({"reply": "Here is the whole thing.", "prompt": "FULL PROMPT\n{history}"})
    )
    result = engineer.discuss(
        store=store,
        thread_id=thread,
        user_message="write it",
        use_case=a_use_case(),
        prompt_under_test="old",
        complete=complete,
        model="m",
        settings=ModelSettings(),
    )
    assert result.has_prompt
    assert result.proposed_prompt == "FULL PROMPT\n{history}"


def test_a_discussion_turn_without_a_prompt_is_normal() -> None:
    store = a_store()
    thread = store.open("engineer")
    complete, _ = replying(json.dumps({"reply": "What does it do today?", "prompt": ""}))
    result = engineer.discuss(
        store=store,
        thread_id=thread,
        user_message="hi",
        use_case=a_use_case(),
        prompt_under_test="old",
        complete=complete,
        model="m",
        settings=ModelSettings(),
    )
    assert not result.has_prompt
    assert result.reply.startswith("What does it")


def test_the_engineer_keeps_the_whole_conversation() -> None:
    store = a_store()
    thread = store.open("engineer")
    reply = json.dumps({"reply": "noted", "prompt": ""})
    complete, sent = replying(reply, reply)
    for message in ("first", "second"):
        engineer.discuss(
            store=store,
            thread_id=thread,
            user_message=message,
            use_case=a_use_case(),
            prompt_under_test="p",
            complete=complete,
            model="m",
            settings=ModelSettings(),
        )
    assert len(store.history(thread)) == 4
    assert any("first" in m["content"] for m in sent[1])


def test_an_unreadable_engineer_reply_leaves_no_half_exchange() -> None:
    store = a_store()
    thread = store.open("engineer")
    complete, _ = replying("not json")
    with pytest.raises(GenerationError):
        engineer.discuss(
            store=store,
            thread_id=thread,
            user_message="hi",
            use_case=a_use_case(),
            prompt_under_test="p",
            complete=complete,
            model="m",
            settings=ModelSettings(),
        )
    assert [m.role for m in store.history(thread)] == ["user"]


# --- end user mode --------------------------------------------------------


def run_once(complete, **overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(
        use_case=a_use_case(),
        prompt_under_test=a_use_case().system_prompt,
        prompt_revision=1,
        user_message="What should I know about this learner?",
        model="openai/gpt-4o-mini",
        settings=ModelSettings(),
        complete=complete,
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    kwargs.update(overrides)
    return one_shot.run_once(**kwargs)  # type: ignore[arg-type]


def test_a_one_shot_run_sends_exactly_one_system_and_one_user_message() -> None:
    complete, sent = replying("There is no previous history on this topic.")
    run_once(complete)
    assert len(sent) == 1
    assert [m["role"] for m in sent[0]] == ["system", "user"]


def test_the_mocks_are_substituted_before_the_call() -> None:
    complete, sent = replying("ok")
    run_once(complete)
    system = sent[0][0]["content"]
    assert "(no matching history)" in system
    assert "{history}" not in system


def test_the_run_records_the_filled_prompt_that_was_actually_sent() -> None:
    complete, _ = replying("ok")
    record = run_once(complete)
    assert "{history}" not in record.system_prompt
    assert record.use_case_key == "grounded_briefing"
    assert record.prompt_revision == 1
    assert record.is_scoreable


def test_two_runs_in_a_row_share_no_history() -> None:
    """The whole point of one-shot: the second call must not see the first."""
    complete, sent = replying("first answer", "second answer")
    run_once(complete, user_message="one")
    run_once(complete, user_message="two")
    assert len(sent[1]) == 2
    assert "first answer" not in str(sent[1])
    assert "one" not in sent[1][1]["content"]


def test_a_prompt_referring_to_a_missing_mock_is_refused_before_the_call() -> None:
    def explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("must not call the model with an unfilled placeholder")

    with pytest.raises(RunFailed, match=r"\{nonexistent\}"):
        run_once(explode, prompt_under_test="Use {history} and {nonexistent}")


def test_an_empty_response_is_reported_and_records_nothing() -> None:
    complete, _ = replying("   ")
    with pytest.raises(RunFailed, match="empty response"):
        run_once(complete)


def test_an_empty_message_is_refused() -> None:
    def explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("must not call the model with an empty message")

    with pytest.raises(RunFailed):
        run_once(explode, user_message="   ")


def test_neither_mode_can_reach_the_evaluator() -> None:
    """Editing, discussing, or running must never start a scoring call."""
    import ast
    import inspect

    forbidden = {"prompt_workbench.core.evaluation", "prompt_workbench.services.metric_adapters"}
    for module in (engineer, one_shot):
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not (imported & forbidden), module.__name__
