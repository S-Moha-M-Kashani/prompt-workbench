"""The end-user side of the chat: one message in, one response out.

No history, by construction. That is what makes the response evidence about the
prompt rather than about the conversation: the model sees the prompt under test
with its mocks filled in, one user message, and nothing else. Send the same
message twice and the only thing that changed is the prompt.

It is the counterpart to the engineer mode, which keeps a full conversation.
Mixing the two — a prompt tested inside a chat that has been going for ten
turns — is how a prompt appears to work because of something said earlier.
"""

from collections.abc import Callable
from datetime import datetime

from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn
from prompt_workbench.models.runs import PromptRun
from prompt_workbench.models.use_case import UseCase


class RunFailed(RuntimeError):
    """The one-shot call produced nothing worth recording."""


def run_once(
    *,
    use_case: UseCase,
    prompt_under_test: str,
    prompt_revision: int,
    user_message: str,
    model: str,
    settings: ModelSettings | None = None,
    complete: CompletionFn,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> PromptRun:
    """Send one message under the prompt under test and record what came back.

    The mocks are substituted here, at the last moment, so the panel keeps
    showing the prompt with its placeholders — the form the user is editing —
    while the model receives the filled version.
    """
    if not user_message.strip():
        raise RunFailed("Nothing was sent: the message is empty")

    missing = use_case.unfilled_placeholders(prompt_under_test)
    if missing:
        named = ", ".join(f"{{{name}}}" for name in missing)
        raise RunFailed(
            f"The prompt refers to {named}, which this use case supplies no mock "
            "for. It would reach the model as literal text — remove it, or use "
            "one of the placeholders listed in the situation."
        )

    system_prompt = use_case.filled_prompt(prompt_under_test)
    response = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        settings=settings,
    ).strip()

    if not response:
        raise RunFailed(
            "The model returned an empty response. Nothing was recorded, so the "
            "prompt and the run history are unchanged."
        )

    return PromptRun(
        id=new_id("run"),
        use_case_key=use_case.key,
        prompt_revision=prompt_revision,
        system_prompt=system_prompt,
        model_id=model,
        user_message=user_message,
        response=response,
        created_at=clock(),
    )
