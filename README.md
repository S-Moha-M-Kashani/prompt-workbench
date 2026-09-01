# Prompt Workbench

A [Streamlit](https://streamlit.io/) workspace for finding the prompt that makes
a job come out right. Pick one of ten real prompt-engineering situations — its
knowledge base, candidate lists and tool schemas already mocked — then work on
the prompt with an engineer, run it as the end user, and examine what comes
back.

Built by [Moha Kashani](mailto:s.moha.m.kashani@gmail.com) as a demonstration of
production-minded LLM application engineering.

> **Status:** early. The ten use cases, both chat modes, and evaluation work
> and are covered by an offline test suite. Nothing persists beyond the browser
> session.

## The idea

A prompt is only judgeable inside a situation. "What should this classifier's
prompt say?" has no answer until you know what the candidate list looks like,
how near the distractors are, and what happens when the list comes back empty.
Those surroundings normally come from a whole application, which is why prompts
usually get written blind.

So each use case here brings its own world, already mocked — a knowledge base, a
candidate list, tool schemas, prior state — and states the failure it is known
to produce. Then you can run the prompt and read what comes back.

## The workflow

1. **Pick a use case.** The dropdown holds ten real situations. Picking one
   writes an explanation into the chat: what the job is, what usually goes
   wrong, what has been mocked for you, and what a good response must do.
2. **Read the prompt it starts from.** One collapsed panel above the chat. Every
   use case starts from a prompt that is plausible and imperfect — usually the
   one with the failure still in it.
3. **Work on it in the chat**, in either of two modes:
   - **Prompt engineer** — discusses the problem across turns and hands back a
     *complete* replacement prompt you apply in one click. Never a diff:
     reassembling a prompt by hand is how a placeholder goes missing.
   - **End user (one-shot)** — one message in, one response out, no history. The
     response reflects the prompt and the mocks and nothing else, so sending the
     same message twice tells you what your edit actually changed.
4. **Evaluate, when you ask.** The use case's own criteria are the ground truth,
   so there is no dataset to write. Nothing is scored until you press the button.

## The ten use cases

| Situation | What is mocked | The failure being hunted |
| --- | --- | --- |
| Judging retrieved chunks | a result set with near-miss distractors | waves everything through; drops an id |
| Grounded briefing | a history block that is **empty** | invents a history that was never there |
| Answering from several contexts | four blocks, one stale and contradicting | states the stale claim as current |
| Classify against a closed set | candidate ids incl. a cluster head | invents ids; over-tags |
| Gatekeeping a near-duplicate | a catalogue already holding it | accepts, and the catalogue proliferates |
| Routing to one bucket | buckets, one fitting but worded differently | creates new instead of reusing |
| Rewriting a running summary | a prior summary holding a detail | logs events; drops the earlier detail |
| One step of a tool-calling agent | tool schemas + a scratchpad of dead ends | repeats the call that just failed |
| Holding a scope guardrail | a message inviting clinical advice | helps anyway, because refusing feels unkind |
| Prompt injection in user data | a store containing the injection | obeys the injected instruction |

Use cases are YAML files in `src/prompt_workbench/use_cases/`. Adding one needs
no code change.

## Design principles

- **One-shot means one-shot.** The end-user mode keeps no history at all, which
  is what makes a response evidence about the prompt rather than about the
  conversation that preceded it.
- **Complete prompts, never diffs.** The engineer returns the whole prompt, so
  applying its advice cannot silently drop a placeholder.
- **Nothing hidden.** Prompts, mocks, criteria, rubrics, and grades are visible
  and editable rather than buried in code.
- **Evaluation is a decision, not a side effect.** No edit, generation, or
  manual run triggers a judge call.
- **Layered architecture.** A thin Streamlit UI over `core/` (orchestration),
  `services/` (the only place that talks to a provider), and dependency-free
  typed `models/`; every layer testable without Streamlit.
- **Offline tests.** The suite injects fake provider ports and makes no live
  model calls — including the CLI judge, whose subprocess runner is injected.
- **Nothing fails quietly.** A judge that times out is recorded as a visible
  failure and excluded from the grade, never softened into a neutral score.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and an API key for an
OpenAI-compatible model provider (defaults to
[OpenRouter](https://openrouter.ai/)).

```console
uv sync                          # install dependencies into .venv
cp .env.example .env             # then set PROVIDER_API_KEY in .env
uv run streamlit run src/prompt_workbench/app.py
```

You can also paste the API key into the app's sidebar instead of using `.env`;
it stays in that browser session.

### Docker

```console
docker build -t prompt-workbench .
docker run --rm -p 8501:8501 -e PROVIDER_API_KEY=your_key prompt-workbench
```

Then open <http://localhost:8501>.

## Development

```console
uv run pytest    # test suite (no live model calls)
uv run mypy      # strict type check of src/
```

The optional local judge needs the [`codex`](https://developers.openai.com/codex/cli/)
CLI on your `PATH` and signed in. The sidebar says whether it was found; without
it, pick the provider judge in the Evaluate area.

## License

Copyright © 2026 Moha Kashani. **All rights reserved.**

This code is published for portfolio and demonstration purposes only. Viewing
it and running it locally to evaluate the author's work is permitted; any
other use, copying, modification, or distribution requires prior written
permission — see [`LICENSE`](LICENSE). For services or collaboration,
[get in touch](mailto:s.moha.m.kashani@gmail.com).
