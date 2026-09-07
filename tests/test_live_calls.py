"""Real calls to real providers — opt-in, and never part of a normal test run.

Everything else in this suite injects a fake, and that rule is not softened
here. Every test in this file skips unless ``PROMPT_WORKBENCH_LIVE`` says you
meant it — set it in the shell for one run, or in ``.env`` to stay opted in and
be able to run a single test from an editor's run button:

    PROMPT_WORKBENCH_LIVE=1 uv run pytest -m live -v

Note what putting it in ``.env`` means: a plain ``uv run pytest`` will then run
these too, and spend money. In the shell it opts in for exactly one command.

Skipped rather than deselected on purpose. A deselected test is invisible, and
an editor's test list is where these are meant to be found — every name starts
with ``test_live_`` so they sort together and can be run one at a time by hand.
Nothing here is imported by the rest of the suite.

Why have them at all. Every adapter and every service in this project is
verified against a *recorded* shape — a fake usage object, a fake tool-use
block, a saved price list — and a recorded shape is a claim about somebody
else's API that keeps moving. These are the only tests that can catch the day
LangChain stops reporting ``usage_metadata``, Anthropic renames a field, the
provider retires a model on the curated shortlist, or a model stops returning
parseable JSON to the case writer. They are a smoke test on the round-trip, not
a measurement of prompt quality.

What they cost. Prompts are one sentence and answers are capped, so a full run
is small change — well under a dollar on the default models. Two of them cost
nothing at all: the model-list checks call no model, and the Codex CLI judge
uses this machine's own login.

What they need. ``PROVIDER_API_KEY`` (from the environment or ``.env``) for
everything except the Anthropic tests, which need ``ANTHROPIC_API_KEY``, and
the Codex judge, which needs the ``codex`` CLI on the path. Each test skips by
name when its own requirement is missing, so a partial setup still runs the
part it can. Every credential is read through the same session-scoped config
the app uses and passed explicitly into the thing that needs it; nothing here
writes a key into the environment.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import pytest
from dotenv import dotenv_values

from prompt_workbench.core import case_intake, scoring, variants
from prompt_workbench.llm_call import registry as frameworks
from prompt_workbench.models import CallRequest, ModelSettings
from prompt_workbench.models.call import OutputStructure, ToolSpec
from prompt_workbench.models.case import CaseBrief, EvalCase
from prompt_workbench.services import (
    codex_cli_client,
    deepeval_judge,
    deepeval_metrics,
    model_registry,
    openrouter_client,
    task_catalog,
)

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


@lru_cache(maxsize=1)
def _env_file_values() -> dict[str, str]:
    """``.env`` as a plain mapping, read once.

    ``dotenv_values`` returns a dict instead of mutating ``os.environ``, which
    is the same choice ``openrouter_client`` makes and for the same reason: a
    credential read here must not become process-global state that some other
    library can pick up by accident.

    The path is absolute, so these tests behave the same whether pytest was
    started from the repository root or from an editor with its own working
    directory.
    """
    return {k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None}


def setting(name: str, default: str = "") -> str:
    """A real environment variable, else the same name in ``.env``, else default.

    The real environment wins, so a one-off ``FOO=1 uv run pytest`` still
    overrides the file — and an editor that runs a single test with the
    project's own working directory still sees what is in ``.env``, which is
    where a switch you set once belongs.
    """
    return (os.environ.get(name) or _env_file_values().get(name) or default).strip()


# Deliberately trivial and deliberately short: what is under test is the
# round-trip, so the answer has to be one nobody can argue with and the bill has
# to be a rounding error.
SYSTEM_PROMPT = "Answer with one word and nothing else."
USER_PROMPT = "What is the capital of France?"
EXPECTED_IN_ANSWER = "paris"

# Overridable, because the cheapest sensible model is a moving target and this
# file should not be the reason someone edits a test to run it.
PROVIDER_MODEL = setting("LIVE_MODEL", "openai/gpt-4o-mini")
ANTHROPIC_MODEL = setting("LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5")

ALL_FRAMEWORKS = ("openai", "langchain", "langgraph", "anthropic")

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


# --- the two gates --------------------------------------------------------


@pytest.fixture(autouse=True)
def _explicit_opt_in() -> None:
    """The gate. Set in the shell for one run, or in ``.env`` to stay opted in.

    Every spelling of "no" is off, case ignored — and so is anything unset.
    Only a deliberate yes opens the gate, because the failure direction here
    costs money: someone writing ``False`` and getting a live run would be
    right to be annoyed.
    """
    if setting("PROMPT_WORKBENCH_LIVE").lower() in ("", "0", "no", "false", "off", "n"):
        pytest.skip(
            "live calls are off. Set PROMPT_WORKBENCH_LIVE=1 in the environment "
            f"or in {ENV_FILE.name} to spend money on a real call."
        )


def _provider_config() -> openrouter_client.ProviderConfig:
    config = openrouter_client.config_from_env(env_file=str(ENV_FILE))
    if not config.has_api_key:
        pytest.skip(f"no {openrouter_client.API_KEY_ENV} in the environment or .env")
    return config


def _adapter_for(framework: str) -> tuple[object, str]:
    """A live adapter for ``framework`` and the model id to use with it.

    Skips rather than fails when the extra is absent or the credential is
    missing: a partial setup should still run the part it can.
    """
    entry = frameworks.get(framework)
    if not entry.is_available():
        pytest.skip(entry.unavailable_reason())
    if entry.reaches_own_provider:
        key = setting("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("no ANTHROPIC_API_KEY in the environment or .env")
        return entry.build(api_key=key), ANTHROPIC_MODEL
    config = _provider_config()
    return entry.build(api_key=config.api_key, base_url=config.base_url), PROVIDER_MODEL


def _completion():  # type: ignore[no-untyped-def]
    """The authoring call, bound to a real key the way the app binds it."""
    config = _provider_config()
    client = openrouter_client.build_client(config)

    def complete(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        return openrouter_client.chat_completion(
            messages, client=client, model=model, settings=settings,
            response_format=response_format,
        )

    return complete


# --- one real round, through every framework ------------------------------


def _request(model_id: str, **overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "model_id": model_id,
        "settings": ModelSettings(temperature=0.0, max_tokens=64),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def _assert_a_real_round_happened(result, framework: str, model_id: str) -> None:
    """What every framework must report, whatever it did to get there."""
    assert not result.failed, result.error
    assert result.framework == framework
    assert result.model_id == model_id
    # A round that reported no latency did not happen; one that reported no
    # tokens cannot be priced, which is the whole point of measuring it.
    assert result.latency_ms is not None and result.latency_ms > 0
    assert result.usage.tokens_in > 0, "the provider reported no input tokens"
    assert result.usage.tokens_out > 0, "the provider reported no output tokens"
    assert result.model_calls >= 1


@pytest.mark.parametrize("framework", ALL_FRAMEWORKS)
def test_live_round_through_each_framework(framework: str) -> None:
    """The smoke test: one question, one answer, and honest numbers back."""
    call, model_id = _adapter_for(framework)
    result = call.run(_request(model_id))  # type: ignore[attr-defined]

    _assert_a_real_round_happened(result, framework, model_id)
    assert EXPECTED_IN_ANSWER in result.answer.lower(), result.answer


@pytest.mark.parametrize("framework", ALL_FRAMEWORKS)
def test_live_tool_call_is_recorded_by_each_framework(framework: str) -> None:
    """The most framework-specific thing there is.

    Each one describes tools differently, reports a request for one
    differently, and returns the result differently — and each one is checked
    against a hand-written fake everywhere else in this suite. Only a real model
    deciding to call a real tool exercises the whole shape at once.
    """
    call, model_id = _adapter_for(framework)
    result = call.run(  # type: ignore[attr-defined]
        _request(
            model_id,
            system_prompt="Use the tools you are given. Never guess a customer's details.",
            user_prompt="Look up customer 7 and tell me what you find.",
            tools=(
                ToolSpec(
                    name="lookup_customer",
                    description="Look a customer up by their numeric id.",
                    input_schema={
                        "type": "object",
                        "properties": {"customer_id": {"type": "integer"}},
                        "required": ["customer_id"],
                    },
                ),
            ),
            settings=ModelSettings(temperature=0.0, max_tokens=300),
        )
    )

    _assert_a_real_round_happened(result, framework, model_id)
    assert [c.name for c in result.tool_calls] == ["lookup_customer"], result.answer
    # A tool round-trip is a second call to the model, and the sweep's cost
    # estimate says so. If this ever reported one, the estimate would be half.
    assert result.model_calls >= 2


def test_live_answer_shape_is_enforced_by_the_provider() -> None:
    """Enforced means the provider guarantees it, not that the model was asked.

    Only the bare SDK forwards the shape as a provider parameter, so this is the
    one framework where a valid-JSON assertion is a statement about the API
    rather than a lucky roll of the model.
    """
    call, model_id = _adapter_for("openai")
    result = call.run(  # type: ignore[attr-defined]
        _request(
            model_id,
            system_prompt="Answer the question.",
            output_structure=OutputStructure(
                name="capital",
                schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            ),
            structure_is_enforced=True,
        )
    )

    assert not result.failed, result.error
    assert result.structure_was_enforced is True
    assert EXPECTED_IN_ANSWER in json.loads(result.answer)["city"].lower()


def test_live_unenforceable_shape_is_reported_as_unenforced() -> None:
    """The honest half of the same feature.

    LangChain forwards tools and parameters but not the shape, so the shape is
    asked for in the prompt. The claim under test is that the result *says* so —
    not that the model complied, which is exactly what nobody can promise.
    """
    call, model_id = _adapter_for("langchain")
    result = call.run(  # type: ignore[attr-defined]
        _request(
            model_id,
            system_prompt="Answer the question.",
            output_structure=OutputStructure(
                name="capital",
                schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
            structure_is_enforced=False,
            settings=ModelSettings(temperature=0.0, max_tokens=120),
        )
    )

    assert not result.failed, result.error
    assert result.structure_was_enforced is False
    assert result.answer.strip()


def test_live_unknown_model_fails_the_round_instead_of_raising() -> None:
    """Costs nothing, and covers the path the sweep depends on: one broken cell
    must come back as a recorded failure, not an exception that ends the run."""
    call, _ = _adapter_for("openai")
    result = call.run(_request("definitely/not-a-real-model"))  # type: ignore[attr-defined]

    assert result.failed
    assert result.error
    # A round that never happened reports no latency, never zero.
    assert result.latency_ms is None


# --- the catalogue: no model called, and still worth running --------------


def test_live_model_list_still_parses_into_priced_entries() -> None:
    """The prices on screen come from this. A changed field name would show as
    every model being free, which is worse than an error."""
    _provider_config()
    registry = model_registry.ModelRegistry(fetch=model_registry.fetch_live)
    priced = [entry for entry in registry.all_models() if entry.has_price]

    assert len(priced) > 50, "the provider returned almost nothing priced"
    assert all(entry.price_in_per_million > 0 for entry in priced)


def test_live_curated_models_still_exist_upstream() -> None:
    """The shortlist is committed to this repository, so it goes stale quietly.
    A retired model would only be discovered by a user's sweep failing."""
    _provider_config()
    registry = model_registry.ModelRegistry(fetch=model_registry.fetch_live)
    live_ids = {entry.id for entry in registry.all_models()}

    missing = [mid for mid in registry.curated_ids() if mid not in live_ids]
    assert not missing, f"retired since the snapshot was taken: {missing}"


# --- the workbench's own prompts, against a real model --------------------


def _a_brief() -> CaseBrief:
    return CaseBrief(
        description="Sort incoming support tickets into one of six queues.",
        task_type_key="classification",
        created_at=WHEN,
        cases=(EvalCase(id="c1", input="I was charged twice.", expected_output="billing"),),
    )


def test_live_generated_test_cases_come_back_and_parse() -> None:
    """The case writer asks for JSON and parses it by hand. A model that starts
    fencing its output, or renaming a key, breaks this and nothing else."""
    cases = case_intake.generate_cases(
        description="Sort incoming support tickets into one of six queues.",
        task=task_catalog.get("classification"),
        count=2,
        complete=_completion(),
        model=PROVIDER_MODEL,
        settings=None,
        new_id=lambda prefix: f"{prefix}-live",
    )

    assert len(cases) == 2
    assert all(case.input.strip() for case in cases)


def test_live_variant_is_written_for_one_approach() -> None:
    """The variant writer returns prose, not JSON, and an empty reply is the
    failure it guards against. Only a real model can produce either."""
    task = task_catalog.get("classification")
    written = variants.generate(
        case=_a_brief(),
        task=task,
        approaches=task.variants[:1],
        complete=_completion(),
        model=PROVIDER_MODEL,
        settings=None,
        new_id=lambda prefix: f"{prefix}-live",
        clock=lambda: WHEN,
    )

    assert len(written) == 1
    assert len(written[0].system_prompt.strip()) > 40
    assert written[0].approach_key == task.variants[0].key


# --- the judges, scoring a real response ----------------------------------


def _a_judged_outcome(backend: str, model: str):  # type: ignore[no-untyped-def]
    choice = deepeval_metrics.suggested_for(("answer_relevancy",))[0]
    return scoring.score_one(
        case=EvalCase(id="c1", input="What is the capital of France?"),
        response="Paris is the capital of France.",
        choice=choice,
        judge=deepeval_judge.build(
            backend, model=model,
            config=openrouter_client.config_from_env(env_file=str(ENV_FILE)),
        ),
    )


def test_live_provider_judge_scores_a_real_response() -> None:
    """The numbers a shipped test suite will actually produce come from here."""
    if not deepeval_metrics.is_available():
        pytest.skip(deepeval_metrics.INSTALL_HINT)
    _provider_config()

    outcome = _a_judged_outcome(deepeval_judge.PROVIDER_BACKEND, PROVIDER_MODEL)

    assert not outcome.failed, outcome.failure
    assert outcome.score is not None and outcome.score > 0.5


def test_live_codex_judge_scores_a_real_response() -> None:
    """Free, and the judge people will actually iterate against. It reaches a
    model through a subprocess, which is a shape no fake can stand in for."""
    if not deepeval_metrics.is_available():
        pytest.skip(deepeval_metrics.INSTALL_HINT)
    if not codex_cli_client.is_available():
        pytest.skip("the codex CLI is not on this machine's PATH")

    outcome = _a_judged_outcome(
        deepeval_judge.CLI_BACKEND, codex_cli_client.KNOWN_MODELS[0]
    )

    assert not outcome.failed, outcome.failure
    assert outcome.score is not None
