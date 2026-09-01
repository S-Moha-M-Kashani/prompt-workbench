"""Choosing which model does the grading.

Two backends behind one callable. The local CLI is the default because it needs
no workspace API key and, being a different model from the one usually under
test, it avoids the most obvious way an evaluation flatters itself — a model
grading its own output.

Both are built here rather than in the UI so that "which judge produced this
grade?" is recorded from one place and shown on the run.
"""

from collections.abc import Mapping

from openai import OpenAI

from prompt_workbench.models.protocols import Message
from prompt_workbench.services import codex_cli_client, model_catalog, openrouter_client
from prompt_workbench.services.metric_adapters import JudgeFn

CLI_BACKEND = "codex"
PROVIDER_BACKEND = "provider"

BACKEND_LABELS: Mapping[str, str] = {
    CLI_BACKEND: "Codex CLI (local, no API key)",
    PROVIDER_BACKEND: "Provider model (uses your API key)",
}


def available_backends() -> tuple[str, ...]:
    """The judges this machine can actually run, best first."""
    backends: list[str] = []
    if codex_cli_client.is_available():
        backends.append(CLI_BACKEND)
    backends.append(PROVIDER_BACKEND)
    return tuple(backends)


def models_for(backend: str) -> tuple[str, ...]:
    """The judge models selectable for ``backend``."""
    if backend == CLI_BACKEND:
        return codex_cli_client.KNOWN_MODELS
    return tuple(model.id for model in model_catalog.all_models())


def default_model_for(backend: str) -> str:
    return (
        codex_cli_client.DEFAULT_MODEL
        if backend == CLI_BACKEND
        else model_catalog.DEFAULT_MODEL_ID
    )


def build_judge(
    backend: str, *, model: str, client: OpenAI | None = None, effort: str = "low"
) -> JudgeFn:
    """A judge callable for ``backend``.

    The provider backend needs a client; the CLI backend must not be given one,
    which is the point of it. Asking for a provider judge without a key is a
    ``ValueError`` here rather than a confusing failure mid-run.
    """
    if backend == CLI_BACKEND:

        def cli_judge(messages: list[Message]) -> str:
            return codex_cli_client.complete(messages, model=model, effort=effort)

        return cli_judge

    if client is None:
        raise ValueError(
            "The provider judge needs an API key. Add one in the sidebar, or "
            "switch the judge backend to the local CLI."
        )

    def provider_judge(messages: list[Message]) -> str:
        return openrouter_client.chat_completion(messages, client=client, model=model)

    return provider_judge
