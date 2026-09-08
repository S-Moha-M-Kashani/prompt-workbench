"""Mocked tools: one behaviour, recorded per call, shared by every framework."""

from prompt_workbench.llm_call import mock_tools
from prompt_workbench.models import ToolSpec

LOOKUP = ToolSpec(name="lookup", description="Look a customer up.")
NOTIFY = ToolSpec(name="notify", description="Send a notice.")


def test_a_mocked_tool_announces_itself_and_returns_that_text(capsys) -> None:
    recorder = mock_tools.TraceRecorder()
    call = mock_tools.build(LOOKUP, recorder)
    assert call({"id": 7}) == "tool lookup was called"
    assert "tool lookup was called" in capsys.readouterr().out


def test_the_recorder_keeps_name_arguments_and_order() -> None:
    recorder = mock_tools.TraceRecorder()
    mock_tools.build(LOOKUP, recorder)({"id": 7})
    mock_tools.build(NOTIFY, recorder)({"to": "ops"})

    trace = recorder.trace()
    assert [(c.name, c.order) for c in trace] == [("lookup", 0), ("notify", 1)]
    assert trace[0].arguments == {"id": 7}


def test_two_recorders_never_share_state() -> None:
    first, second = mock_tools.TraceRecorder(), mock_tools.TraceRecorder()
    mock_tools.build(LOOKUP, first)({})
    assert len(first.trace()) == 1
    assert second.trace() == ()
