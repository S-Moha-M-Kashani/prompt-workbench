"""One prompt, one message, one response — and no history.

That absence is the point. A sweep compares variants and models by running the
same inputs through each, so anything carried between calls would show up as a
difference between configurations that is not actually a difference between
them. The model sees a system prompt, one user message, and nothing else.
"""


from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionWithUsageFn
from prompt_workbench.models.usage import TokenUsage


class RunFailed(RuntimeError):
    """The one-shot call produced nothing worth recording."""


def run_plain(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    settings: ModelSettings | None,
    complete: CompletionWithUsageFn,
) -> tuple[str, TokenUsage]:
    """One call, returning the text and what it cost. No record, no use case.

    What a sweep needs: it runs the same prompt over many cases on many models
    and keeps its own tally, so building a full ``PromptRun`` for each of a
    hundred and twenty calls would be bookkeeping nobody reads.
    """
    if not user_message.strip():
        raise RunFailed("Nothing was sent: the message is empty")

    raw, usage = complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        settings=settings,
    )
    response = raw.strip()
    if not response:
        raise RunFailed(f"{model} returned an empty response")
    return response, usage
