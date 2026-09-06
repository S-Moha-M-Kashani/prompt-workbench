"""The LangChain adapter, exercised against a fake chat model.

Skipped whole when the extra is absent: an uninstalled optional framework is a
supported state, not a failure.
"""

import pytest

pytest.importorskip("langchain", reason="needs the langchain extra")

from prompt_workbench.llm_call.langchain_call import LangChainCall  # noqa: E402
from prompt_workbench.models import CallRequest, ModelSettings, ToolSpec  # noqa: E402
from tests_support_lc import FakeChatModel, ai, ai_asks_for  # noqa: E402

LOOKUP = ToolSpec(name="lookup", description="Look a customer up.")


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "Be brief.",
        "user_prompt": "Who is customer 7?",
        "model_id": "openai/gpt-4o-mini",
        "settings": ModelSettings(temperature=0.2),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def test_a_round_without_tools_answers_and_reports_its_usage() -> None:
    model = FakeChatModel([ai("Ada Lovelace.", (10, 4))])
    result = LangChainCall(chat_model=model).run(_request())

    assert result.answer == "Ada Lovelace."
    assert result.framework == "langchain"
    assert result.usage.tokens_in == 10 and result.usage.tokens_out == 4
    assert result.model_calls == 1
    assert result.tool_calls == ()


def test_a_tool_round_trip_is_recorded_in_order_and_summed() -> None:
    model = FakeChatModel(
        [
            ai_asks_for([("lookup", {"id": 7})], (20, 6)),
            ai("Ada Lovelace.", (30, 5)),
        ]
    )
    result = LangChainCall(chat_model=model).run(_request(tools=(LOOKUP,)))

    assert [c.name for c in result.tool_calls] == ["lookup"]
    assert result.tool_calls[0].arguments == {"id": 7}
    assert result.usage.tokens_in == 50 and result.usage.tokens_out == 11
    assert result.model_calls == 2


def test_the_tools_reach_the_model_bound_by_name() -> None:
    model = FakeChatModel([ai("done")])
    LangChainCall(chat_model=model).run(_request(tools=(LOOKUP,)))
    assert model.bound_tool_names == ["lookup"]


def test_a_round_with_no_tools_binds_none() -> None:
    model = FakeChatModel([ai("done")])
    LangChainCall(chat_model=model).run(_request())
    assert model.bound_tool_names == []


def test_the_system_prompt_reaches_the_model() -> None:
    model = FakeChatModel([ai("done")])
    LangChainCall(chat_model=model).run(_request())
    kinds = [type(message).__name__ for message in model.seen[0]]
    assert kinds[0] == "SystemMessage"
    assert model.seen[0][0].content == "Be brief."


def test_a_model_reporting_no_usage_is_unknown_not_zero_cost() -> None:
    model = FakeChatModel([ai("done", None)])
    result = LangChainCall(chat_model=model).run(_request())
    assert result.usage.is_reported is False
