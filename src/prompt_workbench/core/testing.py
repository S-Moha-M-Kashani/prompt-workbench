"""Running a candidate by hand and recording what happened.

The Test area is the one place a candidate meets a real model. Two rules keep
its output usable as evidence:

**Everything is recorded.** A response is stored with the candidate revision,
brief revision, test case, model, settings, and thread that produced it. A
response missing any of those cannot be scored later, and finding that out at
evaluation time is too late.

**A thread is pinned to one configuration.** Change the candidate or the model
halfway through a conversation and the transcript no longer describes a single
setup — earlier turns influenced later ones under different conditions. So this
refuses, and asks for a new thread, rather than recording responses attributed
to a mixture.
"""

from collections.abc import Callable
from datetime import datetime

from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.models.brief import BriefSnapshot
from prompt_workbench.models.candidates import CandidatePrompt
from prompt_workbench.models.chat import ThreadConfig
from prompt_workbench.models.execution import ExecutionRecord
from prompt_workbench.models.ground_truth import GroundTruthCase
from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn


class TestRunFailed(RuntimeError):
    """The model call did not produce a response worth recording."""


class ConfigurationChanged(RuntimeError):
    """The live configuration no longer matches what this thread was opened with."""


def thread_config(
    *, candidate: CandidatePrompt, model: str, settings: ModelSettings
) -> ThreadConfig:
    """What a thread running this candidate on this model is pinned to."""
    return ThreadConfig(candidate=candidate.ref, model_id=model, settings=settings)


def open_test_thread(
    store: ThreadStore, *, candidate: CandidatePrompt, model: str, settings: ModelSettings
) -> str:
    """Start a testing thread pinned to the current configuration."""
    return store.open(
        "test", config=thread_config(candidate=candidate, model=model, settings=settings)
    )


def send_test_message(
    *,
    store: ThreadStore,
    thread_id: str,
    candidate: CandidatePrompt,
    brief: BriefSnapshot,
    case: GroundTruthCase | None,
    user_message: str,
    model: str,
    settings: ModelSettings,
    complete: CompletionFn,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> ExecutionRecord:
    """Send one message under ``candidate`` and record the response.

    ``case`` may be ``None`` for free-text exploration. That is allowed on
    purpose — poking at a candidate is how you find out what to test — but the
    resulting record says it is not attributable, so it cannot drift into an
    evaluation later.
    """
    current = thread_config(candidate=candidate, model=model, settings=settings)
    if store.config_changed(thread_id, current):
        raise ConfigurationChanged(
            "The candidate, model, or settings changed after this conversation "
            "started. Start a new thread so the responses are not attributed to a "
            "mixture of configurations."
        )

    store.append(thread_id, "user", user_message)
    messages = [
        {"role": "system", "content": candidate.system_prompt},
        *store.provider_messages(thread_id),
    ]

    response = complete(messages, model=model, settings=settings).strip()
    if not response:
        raise TestRunFailed(
            "The model returned an empty response. Nothing was recorded, so the "
            "conversation is unchanged — try again or adjust the settings."
        )

    store.append(thread_id, "assistant", response)
    return ExecutionRecord(
        id=new_id("execution"),
        thread_id=thread_id,
        candidate=candidate.ref,
        source_brief=brief.ref,
        case_id=case.id if case is not None else None,
        model_id=model,
        settings=settings,
        user_message=user_message,
        response=response,
        created_at=clock(),
    )
