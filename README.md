# Prompt Workbench

A [Streamlit](https://streamlit.io/) workspace for engineering system prompts
for any use case. Describe what you need a prompt to do, answer a few
clarifying questions, and the workbench turns that brief into artifacts you can
actually test: candidate system prompts written with different prompt
techniques, a hybrid ground-truth dataset, manual model runs, and an evaluation
you start yourself.

Built by [Moha Kashani](mailto:s.moha.m.kashani@gmail.com) as a demonstration of
production-minded LLM application engineering.

> **Status:** the five workspace areas below are implemented and covered by an
> offline test suite. Workspace artifacts live in the browser session; durable
> save/load is deliberately out of scope for this release.

## The workflow

1. **Define** — write a free-text use-case brief. A clarification chat asks
   about purpose, audience, inputs, desired behaviour, constraints, output
   format, examples, and failure cases, then you edit and confirm a structured
   brief. Confirming records an immutable snapshot.
2. **Ground Truth** — generate a hybrid dataset of test cases from the
   confirmed brief. Each case carries a test message, required criteria,
   forbidden behaviours, and tags, plus an optional reference answer for tasks
   that have one right shape. Every case is editable.
3. **Candidates** — generate one complete candidate system prompt per prompt
   technique (direct/zero-shot, role-based, few-shot, structured-output,
   reasoning-guided). Each candidate is labelled with its technique and its
   provenance, and stays editable.
4. **Test** — pick a candidate, a model, and sampling settings, then run a test
   message yourself in an isolated chat thread that keeps its full short-term
   history. Every response records the candidate, model, settings, brief, test
   case, and thread it came from.
5. **Evaluate** — choose built-in or custom weighted metrics and press
   Evaluate. Nothing is scored until you do. Results show per-metric evidence,
   reasons, failures, and a grade you can reproduce by hand from the displayed
   scores and weights.

## Metrics and judges

Four metrics ship enabled, each answering something the others cannot:

| Metric | Scored by | What it asks |
| --- | --- | --- |
| Criteria coverage | Judge | Did the reply do what this case requires? |
| Forbidden behaviour | Judge | Did it avoid what this case rules out? |
| Format compliance | Code | Is it the right shape — valid JSON, within a stated limit? |
| Reference similarity | Judge | Where an ideal answer exists, how close is this? |

Add your own with a name, a rubric, and a weight; custom metrics score through
exactly the same path as the built-ins.

Judging runs on one of two backends:

- **Codex CLI (default)** — runs the local `codex` binary as a subprocess,
  reaching a strong model through this machine's own login. It needs no
  workspace API key, and it never receives one: the child process gets a strict
  environment allowlist, a read-only sandbox, and an empty working directory.
- **Provider model** — uses your API key, for machines without the CLI.

Judging with a different model from the one under test is the point of the
separate selector: a model grading its own output grades its own habits as
correct.

**Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before trusting a grade.**
Judge noise, generated ground truth, and what a failed metric does to the
arithmetic are all covered there.

## Design principles

- **Only chat is stateful.** Conversations keep thread-scoped history;
  generation and evaluation take explicit input snapshots, so their results are
  reproducible and testable.
- **Nothing hidden.** Prompts, datasets, metrics, rubrics, and grades are
  visible and editable rather than buried in code.
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
