"""The metric layer and its judges, without requiring the optional dependency."""

import pytest

from prompt_workbench.services import deepeval_judge, deepeval_metrics, task_catalog


def _fake_judge():  # type: ignore[no-untyped-def]
    """A judge deepeval will accept.

    It has to genuinely subclass DeepEvalBaseLLM — deepeval type-checks the
    argument rather than duck-typing it, which is also why the real adapters
    subclass it. Built lazily so this module still imports without the extra.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    class FakeJudge(DeepEvalBaseLLM):
        def load_model(self):  # type: ignore[no-untyped-def]
            return self

        def generate(self, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
            return "1.0"

        async def a_generate(self, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
            return "1.0"

        def get_model_name(self):  # type: ignore[no-untyped-def]
            return "fake-judge"

    return FakeJudge()


# --- optionality ----------------------------------------------------------


def test_the_install_command_is_named_rather_than_an_import_error(monkeypatch) -> None:
    monkeypatch.setattr(deepeval_metrics, "is_available", lambda: False)
    with pytest.raises(deepeval_metrics.DeepEvalMissing) as raised:
        deepeval_metrics.require()
    assert deepeval_metrics.INSTALL_HINT in str(raised.value)


def test_telemetry_is_opted_out_before_deepeval_could_be_imported() -> None:
    import os

    assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"


def test_the_catalogue_is_readable_without_the_dependency_installed() -> None:
    """Choosing metrics must work before installing anything, or the user cannot
    tell what the extra would give them."""
    assert deepeval_metrics.METRIC_SPECS
    assert deepeval_metrics.get("faithfulness").deepeval_class == "FaithfulnessMetric"


# --- specs ----------------------------------------------------------------


def test_metrics_scored_in_code_are_marked_as_such() -> None:
    """Their score cannot drift between runs, which is why they are worth
    preferring where the job allows one."""
    for key in ("json_correctness", "exact_match", "tool_correctness"):
        assert deepeval_metrics.get(key).is_deterministic, key
    assert not deepeval_metrics.get("faithfulness").is_deterministic


def test_only_exact_match_calls_no_model_at_all() -> None:
    """Two metrics score in code but still ask a model to write the reason.
    Conflating those with genuinely free ones would under-estimate a sweep."""
    free = {k for k, s in deepeval_metrics.METRIC_SPECS.items() if not s.uses_judge}
    assert free == {"exact_match"}


def test_declared_judge_use_matches_what_deepeval_actually_requires() -> None:
    """The check that caught this: a metric declared judge-free that in fact
    builds a model would reach for an ambient key at evaluate time."""
    pytest.importorskip("deepeval")
    import inspect

    import deepeval.metrics as real_metrics

    for key, spec in deepeval_metrics.METRIC_SPECS.items():
        cls = getattr(real_metrics, spec.deepeval_class)
        takes_model = "model" in inspect.signature(cls.__init__).parameters
        assert takes_model == spec.uses_judge, (
            f"{key}: deepeval {'takes' if takes_model else 'does not take'} a model, "
            f"but the spec says uses_judge={spec.uses_judge}"
        )


def test_directions_match_deepeval(monkeypatch) -> None:
    """The guard for a bug that was already shipped once.

    deepeval 4.2 flipped bias, misuse and hallucination to score in the same
    direction as everything else. A spec still calling them lower-is-better
    inverts the verdict — and worse, a threshold of 0.2 meant as "at most 20%
    violations" becomes a floor that almost any output clears, so the safety
    metrics silently pass everything.

    This reads the pass rule out of the installed library rather than trusting
    the spec, so the next flip fails a test instead of a user."""
    pytest.importorskip("deepeval")
    import inspect

    import deepeval.metrics as real_metrics

    for key, spec in deepeval_metrics.METRIC_SPECS.items():
        source = inspect.getsource(getattr(real_metrics, spec.deepeval_class).is_successful)
        passes_when_high = "self.score >= self.threshold" in source
        assert passes_when_high == spec.higher_is_better, (
            f"{key}: deepeval passes on "
            f"{'score >= threshold' if passes_when_high else 'score <= threshold'}, "
            f"but the spec says higher_is_better={spec.higher_is_better}"
        )


def test_safety_thresholds_are_demanding_rather_than_permissive() -> None:
    """With the direction corrected, a safety threshold must be a high bar. The
    old 0.1-0.2 values now mean 'at least 10% good', which passes anything."""
    for key in ("hallucination", "bias", "pii_leakage", "misuse", "non_advice",
                "role_violation"):
        assert deepeval_metrics.get(key).default_threshold >= 0.8, key


def test_a_metric_declares_the_test_case_fields_it_reads() -> None:
    assert "retrieval_context" in deepeval_metrics.get("faithfulness").needs
    assert "expected_output" in deepeval_metrics.get("exact_match").needs


def test_a_metric_reports_which_fields_the_cases_do_not_supply() -> None:
    """Checked before spending, not after: a faithfulness metric on cases with
    no retrieval context can never score."""
    choice = deepeval_metrics.suggested_for(("faithfulness",))[0]
    assert choice.missing_fields({"input", "actual_output"}) == ("retrieval_context",)
    assert choice.missing_fields({"input", "actual_output", "retrieval_context"}) == ()


def test_a_task_type_suggests_its_metrics_at_starting_thresholds() -> None:
    chosen = deepeval_metrics.suggested_for(task_catalog.get("grounded_qa").metric_keys)
    keys = {choice.key for choice in chosen}
    assert "faithfulness" in keys
    assert all(0.0 <= choice.threshold <= 1.0 for choice in chosen)


# --- the code the user copies out ----------------------------------------


def test_a_metric_renders_as_the_deepeval_call_that_builds_it() -> None:
    """The reason for using deepeval at all: what is tuned here is pasted there."""
    choice = deepeval_metrics.suggested_for(("faithfulness",))[0]
    code = choice.as_code()
    assert code.startswith("FaithfulnessMetric(")
    assert "threshold=0.9" in code
    assert "model=judge" in code


def test_the_one_model_free_metric_renders_without_a_judge() -> None:
    code = deepeval_metrics.suggested_for(("exact_match",))[0].as_code()
    assert "model=" not in code


def test_custom_criteria_render_as_evaluation_steps_when_given() -> None:
    base = deepeval_metrics.suggested_for(("g_eval_criteria",))[0]
    from dataclasses import replace

    choice = replace(base, evaluation_steps=("Check the label is in the set.", "Check nothing else is said."))
    code = choice.as_code()
    assert "evaluation_steps=[" in code
    assert "SingleTurnParams.ACTUAL_OUTPUT" in code


# --- judges ---------------------------------------------------------------


def test_the_cli_judge_is_offered_without_an_api_key() -> None:
    backends = deepeval_judge.available_backends()
    assert deepeval_judge.PROVIDER_BACKEND in backends


def test_building_a_judge_requires_the_dependency(monkeypatch) -> None:
    monkeypatch.setattr(deepeval_metrics, "is_available", lambda: False)
    with pytest.raises(deepeval_metrics.DeepEvalMissing):
        deepeval_judge.build(deepeval_judge.CLI_BACKEND, model="gpt-5.6-luna")


def test_the_cli_judge_reports_reduced_fidelity() -> None:
    """GEval prefers logprobs for fine-grained scores; a subprocess cannot give
    them, and saying so beats silently coarser numbers."""
    note = deepeval_judge.fidelity_note(deepeval_judge.CLI_BACKEND)
    assert "logprob" in note.lower()
    assert deepeval_judge.fidelity_note(deepeval_judge.PROVIDER_BACKEND) == ""


def test_the_judge_backends_have_readable_labels() -> None:
    for backend in deepeval_judge.available_backends():
        assert deepeval_judge.BACKEND_LABELS[backend]


# --- metrics that need configuring before they can be built ---------------


def test_metrics_with_required_arguments_declare_what_they_need() -> None:
    """Discovered by building them for real: four of the sixteen take required
    constructor arguments, and a metric that cannot be built is worse than one
    that is not offered."""
    expected = {
        "json_correctness": "expected_schema",
        "prompt_alignment": "prompt_instructions",
        "misuse": "domain",
        "non_advice": "advice_types",
        "role_violation": "role",
    }
    for key, field_name in expected.items():
        names = {field.name for field in deepeval_metrics.get(key).config_fields}
        assert field_name in names, key


def test_a_metric_needing_no_configuration_declares_none() -> None:
    assert deepeval_metrics.get("faithfulness").config_fields == ()


def test_an_unconfigured_metric_reports_what_is_missing() -> None:
    choice = deepeval_metrics.suggested_for(("misuse",))[0]
    assert choice.missing_config() == ("domain",)
    assert not choice.is_ready


def test_a_configured_metric_is_ready() -> None:
    from dataclasses import replace

    choice = replace(
        deepeval_metrics.suggested_for(("misuse",))[0], config={"domain": "cooking"}
    )
    assert choice.missing_config() == ()
    assert choice.is_ready


def test_configuration_appears_in_the_code_the_user_copies_out() -> None:
    from dataclasses import replace

    choice = replace(
        deepeval_metrics.suggested_for(("non_advice",))[0],
        config={"advice_types": "medical\nlegal"},
    )
    code = choice.as_code()
    assert '"medical"' in code and '"legal"' in code


def test_a_json_example_becomes_a_schema_model() -> None:
    """The user pastes an example of the output they expect; a pydantic model
    is derived from it, since that is what the metric actually wants."""
    pytest.importorskip("deepeval")
    model = deepeval_metrics.schema_from_example('{"label": "billing", "confidence": 0.9}')
    assert model is not None
    assert set(model.model_fields) == {"label", "confidence"}


def test_a_json_example_that_is_not_an_object_is_rejected_clearly() -> None:
    pytest.importorskip("deepeval")
    with pytest.raises(ValueError, match="JSON object"):
        deepeval_metrics.schema_from_example("[1, 2, 3]")


def test_building_an_unconfigured_metric_says_what_is_missing() -> None:
    pytest.importorskip("deepeval")
    choice = deepeval_metrics.suggested_for(("misuse",))[0]
    with pytest.raises(deepeval_metrics.MetricNotConfigured, match="domain"):
        deepeval_metrics.build(choice, judge=_fake_judge())


def test_a_judged_metric_refuses_to_build_without_a_judge() -> None:
    """deepeval would otherwise fall back to an ambient OPENAI_API_KEY, and this
    project never lets a credential arrive by accident."""
    pytest.importorskip("deepeval")
    choice = deepeval_metrics.suggested_for(("faithfulness",))[0]
    with pytest.raises(deepeval_metrics.JudgeRequired, match="OPENAI_API_KEY"):
        deepeval_metrics.build(choice)


def test_the_model_free_metric_builds_with_no_judge() -> None:
    pytest.importorskip("deepeval")
    assert deepeval_metrics.build(deepeval_metrics.suggested_for(("exact_match",))[0]) is not None


def test_every_shipped_metric_can_actually_be_built_once_configured() -> None:
    """The test that would have caught the missing arguments in the first place."""
    pytest.importorskip("deepeval")
    from dataclasses import replace

    samples = {
        "json_correctness": {"expected_schema": '{"label": "x"}'},
        "prompt_alignment": {"prompt_instructions": "Answer in one sentence."},
        "misuse": {"domain": "cooking"},
        "non_advice": {"advice_types": "medical"},
        "role_violation": {"role": "customer service agent"},
        "g_eval_criteria": {},
    }
    for key in deepeval_metrics.METRIC_SPECS:
        base = deepeval_metrics.suggested_for((key,))[0]
        choice = replace(base, config=samples.get(key, {}))
        if key == "g_eval_criteria":
            choice = replace(choice, evaluation_steps=("Check the answer is correct.",))
        judge = _fake_judge() if deepeval_metrics.get(key).uses_judge else None
        metric = deepeval_metrics.build(choice, judge=judge)
        assert metric is not None, key


# --- the tool trace is scoreable evidence ---------------------------------


def _case_expecting(*tools: str):
    from prompt_workbench.models.case import EvalCase

    return EvalCase(
        id="c1",
        input="Who is customer 7?",
        expected_tools=tuple(tools),
    )


def _trace(*names: str):
    from prompt_workbench.models.call import ToolInvocation

    return tuple(
        ToolInvocation(name=name, arguments={}, order=index)
        for index, name in enumerate(names)
    )


def test_a_case_can_name_the_tools_it_expects() -> None:
    case = _case_expecting("lookup")
    assert case.expected_tools == ("lookup",)
    assert {"expected_tools"} <= case.supplied_fields()


def test_a_round_with_a_trace_supplies_the_field_the_metric_needs() -> None:
    from prompt_workbench.core import scoring

    available = scoring.available_fields(_case_expecting("lookup"), _trace("lookup"))
    assert {"tools_called", "expected_tools"} <= available


def test_a_round_that_called_nothing_still_supplies_the_field() -> None:
    """An empty trace is evidence, not a missing input — otherwise the metric
    would be skipped exactly when it has something to say."""
    from prompt_workbench.core import scoring

    available = scoring.available_fields(_case_expecting("lookup"), ())
    assert "tools_called" in available


def test_the_test_case_carries_the_recorded_trace_not_the_answer_text() -> None:
    pytest.importorskip("deepeval")
    from prompt_workbench.core import scoring

    built = scoring.as_llm_test_case(
        _case_expecting("lookup"),
        "I looked up customer 7.",
        tool_calls=_trace("notify"),
    )
    called = [tool.name for tool in (built.tools_called or [])]
    assert called == ["notify"], "the trace, never the assistant's prose"
    assert [tool.name for tool in (built.expected_tools or [])] == ["lookup"]


def test_an_expected_tool_that_was_not_called_fails_visibly() -> None:
    pytest.importorskip("deepeval")
    from prompt_workbench.core import scoring

    outcome = scoring.score_one(
        case=_case_expecting("lookup"),
        response="Ada Lovelace.",
        choice=deepeval_metrics.suggested_for(("tool_correctness",))[0],
        judge=_fake_judge(),
        tool_calls=(),
    )
    assert outcome.failed is False, "a missed tool is a score of 0, not an error"
    assert outcome.score == 0.0
    assert outcome.passed is False


def test_a_matched_trace_scores_full_marks() -> None:
    pytest.importorskip("deepeval")
    from prompt_workbench.core import scoring

    outcome = scoring.score_one(
        case=_case_expecting("lookup"),
        response="Ada Lovelace.",
        choice=deepeval_metrics.suggested_for(("tool_correctness",))[0],
        judge=_fake_judge(),
        tool_calls=_trace("lookup"),
    )
    assert outcome.score == 1.0
    assert outcome.passed is True
