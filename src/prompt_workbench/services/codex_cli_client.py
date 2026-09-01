"""The judge that runs on this machine: the `codex` CLI, as a subprocess.

The second of two evaluator backends, and the default one. It reaches a strong
model through the machine's own CLI login, which means evaluation costs the
workspace no API key at all — at the price of a process spawn per call.

It is driven as a *completion endpoint*, not as an agent. An agent CLI's default
habits — explaining itself, reading the repository, asking a question back — are
all wrong for something whose entire output is a number and a reason, so every
flag below closes one specific way a score could otherwise be wrong.

Nothing here falls back. A missing binary, a non-zero exit, a timeout, an
unreadable stream, or an empty reply each raise ``CliJudgeError``. That is the
whole design: a judge that quietly returns nothing, read tolerantly as "no
opinion" and softened to a neutral score, moves a grade with nothing on screen
to say it happened.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from prompt_workbench.models.protocols import Message

BINARY = "codex"

# The CLI publishes no model list to verify against. This is the cost-sensitive
# default for high-volume judging; a run scores every case against every metric,
# so the judge is by far the busiest caller in the workbench.
DEFAULT_MODEL = "gpt-5.6-luna"
KNOWN_MODELS: tuple[str, ...] = ("gpt-5.6-luna", "gpt-5.6-terra")

EFFORTS: tuple[str, ...] = ("low", "medium", "high", "none")
DEFAULT_EFFORT = "low"
DEFAULT_TIMEOUT_SECONDS = 600

# Appended to whatever system text the caller wrote. The CLI is an assistant by
# default and every caller here parses the reply, so it is told plainly that it
# is not talking to a person.
COMPLETION_INSTRUCTION = (
    "You are being called as a text completion endpoint, not an assistant. "
    "Follow the instruction exactly and emit only the requested output: no "
    "preamble, no explanation outside the requested format, no code fences, no "
    "tool use, and no questions back."
)

# The only variables a call is allowed to see. An **allowlist**, not a denylist.
# The CLI reads its configuration from the environment, and a variable such as
# a base-url override could redirect the call or remap which model actually
# answers — so a grade would be labelled with a model that never saw the text. A
# denylist would need revising every time the CLI ships a new variable, and
# would fail silently in between. Each entry below earns its place by being
# needed to *reach* the model, and none of them can change which model answers.
_KEEP: tuple[str, ...] = (
    "PATH",  # find the binary, and whatever it execs
    "HOME",  # where the CLI login lives: ~/.codex
    "CODEX_HOME",  # ...unless it was moved. Credential location, not configuration
    "TMPDIR",  # the per-user temp directory on macOS
    "USER",
    "LOGNAME",
    "SHELL",  # identity, which the CLI reads for its own logs
    "LANG",
    "LC_ALL",
    "LC_CTYPE",  # the child's idea of text; see `encoding` in `_subprocess_runner`
)
_KEEP_PREFIX: tuple[str, ...] = ("XDG_",)

# Deliberately absent: proxy and certificate variables, and above all the
# workspace's own PROVIDER_API_KEY. The judge uses the machine's CLI login; a
# workspace credential has no business in this process.


class CliJudgeError(RuntimeError):
    """The CLI did not produce a reply this workbench can use, and why."""


@dataclass(frozen=True)
class CliResult:
    """What one subprocess call came back with."""

    returncode: int
    stdout: str
    stderr: str


class CliRunner(Protocol):
    """The seam that makes this module testable without spawning anything."""

    def __call__(
        self,
        argv: list[str],
        *,
        stdin: str,
        timeout: int,
        env: dict[str, str],
        cwd: str,
    ) -> CliResult: ...


def build_argv(*, model: str, effort: str) -> list[str]:
    """The exact command one judge call runs.

    ``-s read-only`` and ``--ephemeral`` are the sandbox: a judge that could
    write, or that kept state between calls, would make two runs of the same
    evaluation incomparable. ``--ignore-user-config`` and ``--ignore-rules``
    stop the machine's own CLI configuration from editing the judge's behaviour
    behind the workbench's back.
    """
    return [
        BINARY,
        "exec",
        "--json",
        "-s",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-rules",
        "--ignore-user-config",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-",  # read the prompt from stdin
    ]


def checked_effort(effort: str) -> str:
    """The effort value, or a refusal naming what the CLI accepts.

    Refused before the call because an unaccepted value exits successfully and
    answers nothing — which is exactly the shape a tolerant caller would read as
    "no opinion".
    """
    if effort not in EFFORTS:
        raise ValueError(
            f"{effort!r} is not a reasoning effort {BINARY} accepts; expected one of "
            + ", ".join(repr(value) for value in EFFORTS)
        )
    return effort


def child_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment one call runs with: ``_KEEP``, and nothing else."""
    source: Mapping[str, str] = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if name in _KEEP or name.startswith(_KEEP_PREFIX)
    }


def read_stream(stdout: str) -> tuple[str, dict[str, Any]]:
    """The last agent message and the turn's token usage.

    *Last*, not first: the CLI may emit a preliminary message before its real
    answer, and scoring the preamble would be worse than not scoring at all.
    Unparseable lines are skipped rather than fatal — the stream is a log, and a
    line a future version adds must not end an evaluation.
    """
    text = ""
    usage: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text") or ""
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    return text, usage


# A whole reply that is one fenced block and nothing else. Anchored at both ends
# on purpose — see `unfence`.
_WHOLE_FENCE = re.compile(r"\A```[A-Za-z0-9_+-]*\n(?P<body>.*?)\n?```\Z", re.S)


def unfence(text: str) -> str:
    """Unwrap a reply that is *entirely* one code fence.

    That wrapping is a habit of the transport, not part of the judgement. A
    reply that merely *contains* a fence is returned byte-exact: tidying those
    too is how a score measured through the CLI stops being comparable with the
    same score measured through the provider.
    """
    match = _WHOLE_FENCE.match(text.strip())
    if not match or "```" in match.group("body"):
        return text
    return match.group("body").strip()


def _split_messages(messages: list[Message]) -> tuple[str, str]:
    """System text and everything else; a non-user turn keeps its role label."""
    system: list[str] = []
    body: list[str] = []
    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "user")
        if role == "system":
            system.append(content)
        elif role == "user":
            body.append(content)
        else:
            body.append(f"{role}: {content}")
    return "\n\n".join(system).strip(), "\n\n".join(body).strip()


def _subprocess_runner(
    argv: list[str], *, stdin: str, timeout: int, env: dict[str, str], cwd: str
) -> CliResult:
    """The real runner. Everything about it is deliberate; see below.

    ``encoding="utf-8"`` rather than ``text=True``: the latter follows the
    locale, and under a C/POSIX locale that is ASCII — so a correct judgement
    about non-English text would fail to decode, or under latin-1 would decode
    as silent mojibake. ``errors="strict"`` for the same reason nothing else
    here falls back: a replaced byte changes the text that gets scored, with
    nothing on the record saying so.
    """
    done = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    return CliResult(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)


def is_available(*, which: Callable[[str], str | None] = shutil.which) -> bool:
    """Whether this machine has the binary — the whole check that is possible."""
    return bool(which(BINARY))


def complete(
    messages: list[Message],
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: CliRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Send ``messages`` to the local CLI and return its reply text.

    Matches the shape of ``openrouter_client.chat_completion`` so the two judge
    backends are interchangeable at the port.
    """
    run = runner if runner is not None else _subprocess_runner
    system, body = _split_messages(messages)
    instructions = f"{system}\n\n{COMPLETION_INSTRUCTION}".strip()
    prompt = f"{instructions}\n\n{body}".strip()
    argv = build_argv(model=model, effort=checked_effort(effort))

    # A fresh directory per call, not one per process, so nothing one call
    # leaves behind can be read by the next.
    with tempfile.TemporaryDirectory() as empty:
        try:
            result = run(
                argv, stdin=prompt, timeout=timeout, env=child_env(environ), cwd=empty
            )
        except FileNotFoundError as error:
            raise CliJudgeError(
                f"{BINARY} is not installed on this machine, so the local judge "
                "cannot run. Install it, or switch the judge backend to the "
                "provider in the Evaluate area."
            ) from error
        except (TimeoutError, subprocess.TimeoutExpired) as error:
            raise CliJudgeError(f"{BINARY} did not answer within {timeout}s") from error
        except UnicodeDecodeError as error:
            raise CliJudgeError(
                f"{BINARY} wrote bytes that are not UTF-8, and guessing an encoding "
                f"would change the text being scored: {error}"
            ) from error

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:400]
        raise CliJudgeError(f"{BINARY} exited {result.returncode}: {detail}")

    try:
        text, _usage = read_stream(result.stdout)
    except Exception as error:  # noqa: BLE001 - any parse failure is one failure
        raise CliJudgeError(
            f"{BINARY} wrote a reply this workbench cannot read: {result.stdout[:200]!r}"
        ) from error

    text = unfence(text.strip())
    if not text:
        raise CliJudgeError(
            f"{BINARY} exited successfully but answered with no text at all. This is "
            "reported as a metric failure rather than a neutral score, because a "
            "silent half-mark would move the grade with nothing on screen to explain it."
        )
    return text
