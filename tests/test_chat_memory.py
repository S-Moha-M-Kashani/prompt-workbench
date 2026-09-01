"""Thread-scoped short-term memory: the only stateful thing in the workbench."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.core.chat_memory import ContextLimitReached, ThreadStore
from prompt_workbench.models import ModelSettings, SourceRef, ThreadConfig, sequential_ids

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_store() -> ThreadStore:
    return ThreadStore(new_id=sequential_ids(), clock=lambda: WHEN)


def test_each_opened_thread_gets_its_own_id() -> None:
    store = a_store()
    first = store.open("discovery")
    second = store.open("discovery")
    assert first != second


def test_a_thread_returns_its_complete_history_in_order() -> None:
    store = a_store()
    thread_id = store.open("discovery")
    store.append(thread_id, "user", "one")
    store.append(thread_id, "assistant", "two")
    store.append(thread_id, "user", "three")
    assert [m.content for m in store.history(thread_id)] == ["one", "two", "three"]


def test_one_thread_never_sees_another_threads_messages() -> None:
    store = a_store()
    first, second = store.open("discovery"), store.open("test")
    store.append(first, "user", "private to the first")
    assert store.history(second) == ()
    assert len(store.history(first)) == 1


def test_clearing_a_thread_opens_an_empty_replacement() -> None:
    store = a_store()
    original = store.open("discovery")
    store.append(original, "user", "forget me")
    replacement = store.clear(original)
    assert replacement != original
    assert store.history(replacement) == ()
    with pytest.raises(KeyError):
        store.history(original)


def test_a_thread_remembers_the_configuration_it_was_opened_with() -> None:
    store = a_store()
    config = ThreadConfig(
        candidate=SourceRef(id="cand-1", revision=1),
        model_id="openai/gpt-4o-mini",
        settings=ModelSettings(temperature=0.2),
    )
    thread_id = store.open("test", config=config)
    assert store.thread(thread_id).config == config


def test_a_thread_is_stale_when_the_live_configuration_has_moved_on() -> None:
    store = a_store()
    config = ThreadConfig(
        candidate=SourceRef(id="cand-1", revision=1), model_id="m", settings=ModelSettings()
    )
    thread_id = store.open("test", config=config)
    same = ThreadConfig(
        candidate=SourceRef(id="cand-1", revision=1), model_id="m", settings=ModelSettings()
    )
    changed = ThreadConfig(
        candidate=SourceRef(id="cand-1", revision=1), model_id="other", settings=ModelSettings()
    )
    assert not store.config_changed(thread_id, same)
    assert store.config_changed(thread_id, changed)


def test_history_that_would_overflow_the_budget_blocks_instead_of_trimming() -> None:
    store = a_store()
    thread_id = store.open("discovery")
    store.append(thread_id, "user", "x" * 4000)
    with pytest.raises(ContextLimitReached) as raised:
        store.check_budget(thread_id, next_message="y" * 4000, token_budget=100)
    assert "clear" in str(raised.value).lower()
    # Nothing was dropped to make room.
    assert len(store.history(thread_id)) == 1


def test_a_thread_within_budget_reports_its_estimated_usage() -> None:
    store = a_store()
    thread_id = store.open("discovery")
    store.append(thread_id, "user", "hello there")
    used = store.check_budget(thread_id, next_message="and again", token_budget=10_000)
    assert 0 < used < 10_000


def test_asking_for_an_unknown_thread_is_an_error_not_an_empty_list() -> None:
    with pytest.raises(KeyError):
        a_store().history("t-does-not-exist")
