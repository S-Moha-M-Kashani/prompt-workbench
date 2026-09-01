"""The local CLI judge: argv, environment, stream parsing, and every failure.

No test spawns a process. The runner is injected, so these assert on exactly
what the workbench *would* run and how it reads what comes back.
"""

import json

import pytest

from prompt_workbench.services import codex_cli_client as cli
from prompt_workbench.services.codex_cli_client import CliJudgeError, CliResult


def stream(*events: dict[str, object]) -> str:
    return "\n".join(json.dumps(event) for event in events)


AGENT_MESSAGE = {
    "type": "item.completed",
    "item": {"id": "item_1", "type": "agent_message", "text": "0.75"},
}
TURN_DONE = {
    "type": "turn.completed",
    "usage": {"input_tokens": 100, "output_tokens": 4},
}


def runner_returning(stdout: str, *, returncode: int = 0, stderr: str = "") -> cli.CliRunner:
    def run(argv: list[str], *, stdin: str, timeout: int, env: dict[str, str], cwd: str) -> CliResult:
        return CliResult(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


# --- argv -----------------------------------------------------------------


def test_the_command_pins_the_model_and_the_reasoning_effort() -> None:
    argv = cli.build_argv(model="gpt-5.6-luna", effort="low")
    assert argv[:2] == ["codex", "exec"]
    assert "-m" in argv and argv[argv.index("-m") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="low"' in argv


def test_the_command_sandboxes_the_judge_and_ignores_local_configuration() -> None:
    argv = cli.build_argv(model="m", effort="low")
    for flag in ("--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config"):
        assert flag in argv, f"{flag} missing: a judge must not read the user's project"
    assert argv[argv.index("-s") + 1] == "read-only"
    assert argv[-1] == "-", "the prompt is delivered on stdin, not in the argv"


def test_an_effort_the_cli_does_not_accept_is_refused_before_the_call() -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        cli.checked_effort("enormous")
    assert cli.checked_effort("high") == "high"


# --- environment ----------------------------------------------------------


def test_the_child_environment_is_an_allowlist_not_a_denylist() -> None:
    env = cli.child_env(
        {
            "PATH": "/usr/bin",
            "HOME": "/Users/someone",
            "CODEX_HOME": "/Users/someone/.codex",
            "OPENAI_BASE_URL": "https://attacker.example",
            "ANYTHING_ELSE": "leaked",
        }
    )
    assert set(env) == {"PATH", "HOME", "CODEX_HOME"}


def test_the_workspace_provider_key_never_reaches_the_child() -> None:
    env = cli.child_env({"PATH": "/usr/bin", "PROVIDER_API_KEY": "sk-secret"})
    assert "PROVIDER_API_KEY" not in env
    assert "sk-secret" not in "".join(env.values())


def test_linux_desktop_variables_are_kept_by_prefix() -> None:
    env = cli.child_env({"XDG_CONFIG_HOME": "/home/a/.config", "XDG_DATA_HOME": "/home/a/.local"})
    assert set(env) == {"XDG_CONFIG_HOME", "XDG_DATA_HOME"}


# --- reading the stream ---------------------------------------------------


def test_the_last_agent_message_is_the_answer_not_the_first() -> None:
    text, usage = cli.read_stream(
        stream(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "thinking"}},
            AGENT_MESSAGE,
            TURN_DONE,
        )
    )
    assert text == "0.75"
    assert usage["output_tokens"] == 4


def test_a_line_the_workbench_does_not_understand_is_skipped_not_fatal() -> None:
    text, _ = cli.read_stream("not json at all\n" + stream(AGENT_MESSAGE, TURN_DONE))
    assert text == "0.75"


def test_an_error_item_in_the_stream_does_not_hide_the_answer() -> None:
    text, _ = cli.read_stream(
        stream(
            {"type": "item.completed", "item": {"type": "error", "message": "a warning"}},
            AGENT_MESSAGE,
            TURN_DONE,
        )
    )
    assert text == "0.75"


def test_a_reply_that_is_entirely_one_code_fence_is_unwrapped() -> None:
    assert cli.unfence('```json\n{"score": 1}\n```') == '{"score": 1}'
    assert cli.unfence("```\nplain\n```") == "plain"


def test_a_reply_that_merely_contains_a_fence_is_left_byte_exact() -> None:
    text = "Here you go:\n```json\n{}\n```\nhope that helps"
    assert cli.unfence(text) == text


# --- calling --------------------------------------------------------------


def test_a_successful_call_returns_the_agent_message() -> None:
    text = cli.complete(
        [{"role": "user", "content": "score this"}],
        runner=runner_returning(stream(AGENT_MESSAGE, TURN_DONE)),
    )
    assert text == "0.75"


def test_the_system_message_and_the_completion_instruction_go_to_stdin() -> None:
    captured: dict[str, str] = {}

    def run(argv: list[str], *, stdin: str, timeout: int, env: dict[str, str], cwd: str) -> CliResult:
        captured["stdin"] = stdin
        return CliResult(returncode=0, stdout=stream(AGENT_MESSAGE, TURN_DONE), stderr="")

    cli.complete(
        [{"role": "system", "content": "You are a judge."}, {"role": "user", "content": "score"}],
        runner=run,
    )
    assert "You are a judge." in captured["stdin"]
    assert cli.COMPLETION_INSTRUCTION in captured["stdin"]
    assert "score" in captured["stdin"]


def test_a_non_zero_exit_raises_and_reports_what_the_cli_said() -> None:
    with pytest.raises(CliJudgeError, match="exited 1"):
        cli.complete(
            [{"role": "user", "content": "x"}],
            runner=runner_returning("", returncode=1, stderr="not logged in"),
        )


def test_an_empty_reply_raises_rather_than_being_read_as_no_opinion() -> None:
    with pytest.raises(CliJudgeError, match="no text"):
        cli.complete(
            [{"role": "user", "content": "x"}], runner=runner_returning(stream(TURN_DONE))
        )


def test_a_missing_binary_names_the_alternative_instead_of_a_traceback() -> None:
    def run(argv: list[str], **kwargs: object) -> CliResult:
        raise FileNotFoundError(argv[0])

    with pytest.raises(CliJudgeError, match="not installed"):
        cli.complete([{"role": "user", "content": "x"}], runner=run)  # type: ignore[arg-type]


def test_a_timeout_says_how_long_it_waited() -> None:
    def run(argv: list[str], **kwargs: object) -> CliResult:
        raise TimeoutError

    with pytest.raises(CliJudgeError, match="did not answer within"):
        cli.complete([{"role": "user", "content": "x"}], runner=run, timeout=30)  # type: ignore[arg-type]


def test_availability_is_a_lookup_of_the_binary_on_this_machine() -> None:
    assert cli.is_available(which=lambda name: "/usr/local/bin/codex")
    assert not cli.is_available(which=lambda name: None)
