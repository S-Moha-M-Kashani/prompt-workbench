"""The LangGraph adapter: one model node, one tool node, one round."""

import pytest

pytest.importorskip("langgraph", reason="needs the langgraph extra")

from prompt_workbench.llm_call.langgraph_call import LangGraphCall  # noqa: E402
from prompt_workbench.models import (  # noqa: E402
    CallRequest,
    ModelSettings,
    TokenUsage,
    ToolSpec,
)
from tests_support_lc import FakeChatModel, ai, ai_asks_for  # noqa: E402

LOOKUP = ToolSpec(name="lookup", description="Look a customer up.")
NOTIFY = ToolSpec(name="notify", description="Send a notice.")


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "Be brief.",
        "user_prompt": "Who is customer 7?",
        "model_id": "openai/gpt-4o-mini",
        "settings": ModelSettings(temperature=0.2),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def test_a_round_without_tools_runs_the_model_node_alone() -> None:
    model = FakeChatModel([ai("Ada Lovelace.", (10, 4))])
    result = LangGraphCall(chat_model=model).run(_request())

    assert result.answer == "Ada Lovelace."
    assert result.framework == "langgraph"
    assert result.usage == TokenUsage(tokens_in=10, tokens_out=4)
    assert result.model_calls == 1
    assert result.tool_calls == ()


def test_the_tool_node_runs_and_the_trace_keeps_its_order() -> None:
    model = FakeChatModel(
        [
            ai_asks_for([("lookup", {"id": 7}), ("notify", {"to": "ops"})], (20, 6)),
            ai("Ada Lovelace.", (30, 5)),
        ]
    )
    result = LangGraphCall(chat_model=model).run(_request(tools=(LOOKUP, NOTIFY)))

    assert [c.name for c in result.tool_calls] == ["lookup", "notify"]
    assert [c.order for c in result.tool_calls] == [0, 1]


def test_usage_is_summed_across_every_step() -> None:
    model = FakeChatModel(
        [ai_asks_for([("lookup", {"id": 7})], (20, 6)), ai("Ada Lovelace.", (30, 5))]
    )
    result = LangGraphCall(chat_model=model).run(_request(tools=(LOOKUP,)))
    assert result.usage.tokens_in == 50 and result.usage.tokens_out == 11
    assert result.model_calls == 2


def test_the_system_prompt_leads_the_conversation() -> None:
    model = FakeChatModel([ai("done")])
    LangGraphCall(chat_model=model).run(_request())
    first = model.seen[0][0]
    assert type(first).__name__ == "SystemMessage"
    assert first.content == "Be brief."
