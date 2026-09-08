"""Driving the whole page headlessly, with nothing real behind it.

``test_app`` asserts on single renders; the end-to-end tests walk the page from
a described case to a ranked sweep. Both need the same three things switched
off — the live model list, the missing credential, and every call to a model —
so they live here rather than being written twice and drifting apart.

Nothing in this module reaches the network. The registry is pinned to the
committed snapshot, the credential is a string, and the framework adapter is a
script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from prompt_workbench.models.call import CallRequest, CallResult, ToolInvocation
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.services import model_registry, openrouter_client
from prompt_workbench.services.openrouter_client import ProviderConfig

APP = Path(__file__).resolve().parents[1] / "src" / "prompt_workbench" / "app.py"

FAKE_API_KEY = "test-fake-key"


def go_offline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pin the catalogue to the snapshot and refuse any live fetch."""
    from prompt_workbench.ui import session as ui_session

    def fail() -> dict:
        raise RuntimeError("tests never fetch the live model list")

    monkeypatch.setattr(model_registry, "fetch_live", fail)
    # Already a plain function when a test starts the app twice; the cache
    # it would have cleared is the one replaced on the first start.
    getattr(ui_session._cached_model_list, "clear", lambda: None)()
    monkeypatch.setattr(ui_session, "_cached_model_list", fail)


def with_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        openrouter_client, "config_from_env", lambda **_: ProviderConfig(api_key=FAKE_API_KEY)
    )


def rendered(at: Any) -> str:
    """Every string the page put on screen."""
    texts = [
        element.value
        for group in (at.title, at.caption, at.markdown, at.subheader, at.info,
                      at.warning, at.success, at.error)
        for element in group
        if isinstance(element.value, str)
    ]
    for tile in at.metric:
        texts.extend(
            part for part in (tile.label, tile.value, tile.delta) if isinstance(part, str)
        )
    return " ".join(texts)


class ScriptedRunner:
    """A framework adapter that answers from a script and remembers every ask.

    One instance per framework, so a sweep over two frameworks can be told
    apart afterwards — which is the whole point of the framework being part of
    a result rather than a detail of how it ran.
    """

    def __init__(
        self,
        framework: str,
        *,
        answer: str = "billing",
        answers: dict[str, str] | None = None,
        fails_on: tuple[str, ...] = (),
        tool_calls: tuple[ToolInvocation, ...] = (),
        usage: TokenUsage = TokenUsage(tokens_in=120, tokens_out=8),
    ) -> None:
        self.framework = framework
        self.requests: list[CallRequest] = []
        self._answer = answer
        self._answers = answers or {}
        self._fails_on = fails_on
        self._tool_calls = tool_calls
        self._usage = usage

    def fail_on(self, *model_ids: str) -> None:
        """Make these models refuse, once the sweep has said which they are."""
        self._fails_on = self._fails_on + model_ids

    def run(self, request: CallRequest) -> CallResult:
        self.requests.append(request)
        if request.model_id in self._fails_on:
            return CallResult(
                answer="", framework=self.framework, model_id=request.model_id,
                latency_ms=None, error="the provider refused",
            )
        return CallResult(
            answer=self._answers.get(request.model_id, self._answer),
            framework=self.framework,
            model_id=request.model_id,
            usage=self._usage,
            latency_ms=410.0,
            tool_calls=self._tool_calls,
            model_calls=2 if self._tool_calls else 1,
        )

    def describe(self) -> str:
        return self.framework


def use_runners(monkeypatch, **by_framework: ScriptedRunner) -> dict[str, ScriptedRunner]:
    """Serve one scripted adapter per framework, and no judge.

    The judge is stubbed out too: a sweep asks for one before it runs a cell,
    and building the real one would reach for a credential these tests do not
    have.
    """
    from prompt_workbench.ui import session as ui_session

    runners = dict(by_framework)

    def runner_for(key: str) -> ScriptedRunner:
        return runners.setdefault(key, ScriptedRunner(key))

    monkeypatch.setattr(ui_session, "call_runner", runner_for)
    monkeypatch.setattr(ui_session, "judge", lambda: None)
    return runners


def start(monkeypatch, **runners: ScriptedRunner) -> Any:
    """A running app with a key, a pinned catalogue and no real calls."""
    go_offline(monkeypatch)
    with_credentials(monkeypatch)
    use_runners(monkeypatch, **runners)
    return AppTest.from_file(APP).run()
