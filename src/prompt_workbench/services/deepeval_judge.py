"""The models that run deepeval's metrics.

Two, for two different moments. The local `codex` CLI costs no API key, so it is
what you iterate against — a sweep re-scoring forty responses is a lot of judge
calls to pay for while still changing your mind. A provider model gives the
numbers your real test suite will actually produce, so it is what you confirm
with before writing thresholds into another project.

The CLI is not free of trade-offs, and the honest one is stated rather than
buried: GEval scores by reading token log-probabilities when the judge can
supply them, and a subprocess cannot. It still works — deepeval falls back to
asking for a number directly — but the scores are coarser and move in bigger
steps. Confirm on a provider judge before trusting a threshold to two decimals.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from prompt_workbench.services import codex_cli_client, deepeval_metrics, openrouter_client

CLI_BACKEND = "codex"
PROVIDER_BACKEND = "provider"

BACKEND_LABELS: dict[str, str] = {
    CLI_BACKEND: "Codex CLI (local, no API key)",
    PROVIDER_BACKEND: "Provider model (uses your API key)",
}

_CLI_FIDELITY_NOTE = (
    "GEval prefers token logprobs for fine-grained scores and a subprocess cannot "
    "provide them, so scores from this judge are coarser. Confirm a threshold on a "
    "provider judge before writing it into another project's test suite."
)


def available_backends() -> tuple[str, ...]:
    """The judges this machine can run, best-for-iteration first."""
    backends: list[str] = []
    if codex_cli_client.is_available():
        backends.append(CLI_BACKEND)
    backends.append(PROVIDER_BACKEND)
    return tuple(backends)


def models_for(backend: str) -> tuple[str, ...]:
    if backend == CLI_BACKEND:
        return codex_cli_client.KNOWN_MODELS
    return ("openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-sonnet-4.5")


def fidelity_note(backend: str) -> str:
    """What this judge gives up, if anything."""
    return _CLI_FIDELITY_NOTE if backend == CLI_BACKEND else ""


def build(backend: str, *, model: str, config: Any = None) -> Any:
    """A deepeval judge model for ``backend``.

    Returns something deepeval accepts as its ``model=`` argument. Requires the
    optional dependency, since the base class comes from it.
    """
    deepeval_metrics.require()
    from deepeval.models.base_model import DeepEvalBaseLLM

    class CodexJudge(DeepEvalBaseLLM):
        """deepeval's model interface over the local CLI.

        ``load_model`` returns nothing to load: the model lives in another
        process and is reached per call. That is the whole adaptation.
        """

        def __init__(self, model_name: str) -> None:
            self._model_name = model_name

        def load_model(self) -> "CodexJudge":
            return self

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            return codex_cli_client.complete(
                [{"role": "user", "content": prompt}], model=self._model_name
            )

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            # Sync only: the thing being parallelised would be a process, and
            # deepeval's own async path already runs metrics concurrently.
            return self.generate(prompt, *args, **kwargs)

        def get_model_name(self) -> str:
            return f"codex/{self._model_name}"

        def supports_log_probs(self) -> bool:
            # False, and deliberately so: claiming support would make GEval ask
            # for logprobs the CLI cannot return, and fail rather than fall back.
            return False

    class ProviderJudge(DeepEvalBaseLLM):
        """deepeval's model interface over the session's provider client."""

        def __init__(self, model_name: str, client: OpenAI) -> None:
            self._model_name = model_name
            self._client = client

        def load_model(self) -> "ProviderJudge":
            return self

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            return openrouter_client.chat_completion(
                [{"role": "user", "content": prompt}],
                client=self._client,
                model=self._model_name,
            )

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            return self.generate(prompt, *args, **kwargs)

        def get_model_name(self) -> str:
            return self._model_name

    if backend == CLI_BACKEND:
        return CodexJudge(model)

    if config is None or not config.has_api_key:
        raise ValueError(
            "The provider judge needs an API key. Add one in the sidebar, or switch "
            "the judge to the local CLI."
        )
    return ProviderJudge(model, openrouter_client.build_client(config))
