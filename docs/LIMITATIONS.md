# What a score here does and does not mean

The workbench produces numbers, and ranks configurations by them. This page is
about how much weight those numbers can carry, because a score with a tick next
to it looks more authoritative than it is.

## What the numbers on screen support

These statements are rendered on the page beside the numbers they qualify, and
they are quoted here from `core/considerations.py` so the document and the
screen cannot drift. A test fails if any of them stops matching.

### A single run's latency is an observation, not a benchmark

A latency here is one run: it carries network variance, provider queueing and
this machine's own overhead, and none of that is the framework's doing.
Comparing two frameworks on one run each tells you almost nothing. Repeat runs
and their spread are what turn this into a claim.

### A judged grade is one model's opinion

Most metrics here are a model reading a rubric, and they move between runs and
prefer longer, hedged answers. Exact match, JSON correctness and tool
correctness decide in code; everything else is judgement. Compare variants
against each other, and do not read a score as a percentage correct.

### Mocked tools measure the asking, not the handling

Every tool here is mocked: it announces itself, returns that text, and records
the call. So a tool trace measures whether the model asked for the right tool
with the right arguments. It measures nothing about whether real tool output is
handled, which is a different question needing a different bench.

### One round is not a multi-round agent

The unit measured here is a single round: prompts in, one answer out, with
whatever tool round-trips that answer needed. An agent's real behaviour lives
in the state it carries between rounds, and this bench deliberately keeps none.
Tune a workflow by bringing each of its rounds here separately.

### An unknown cost is shown as unknown, never as zero

Where the provider published no price, or the framework reported no token
counts, the cost is reported as unknown. It is never shown as $0.00, because a
free call and a call nobody measured must not read the same.

## Most metrics are scored by a language model

deepeval's metrics divide into three kinds, and the difference matters more than
the score does:

- **`ExactMatchMetric`** calls no model. It is string comparison, and it is as
  reliable as your expected outputs are.
- **`JsonCorrectnessMetric`** and **`ToolCorrectnessMetric`** decide in code —
  does it parse against the schema, were the right tools called — but still ask
  a model to *write the reason*. The verdict is deterministic; the explanation
  is not.
- **Everything else** — GEval, faithfulness, answer relevancy, summarization,
  bias, misuse and the rest — is a model reading a rubric.

That last group is inconsistent in the ordinary way models are: run the same
evaluation twice and scores move, usually by a little and occasionally by a lot.
It also brings its own preferences — for longer answers, for hedged ones, for
the style it was trained to write.

What follows:

- **Compare, do not certify.** "Variant B scored 0.12 above variant A on the
  same cases with the same judge" is a useful sentence. "Variant B is 87%
  correct" is not.
- **Do not judge with the model under test.** A model scoring its own output
  grades its own habits as correct. The judge is chosen separately in the
  sidebar for exactly this reason.
- **A single run is not a measurement.** One sweep gives one sample per cell.
  A variant that would score 0.9, 0.7 and 0.7 on three runs is indistinguishable
  here from one that scores 0.9 every time. Repeat runs and their spread are
  planned for 0.4.0; until then, treat a narrow margin as no margin.

## The local CLI judge is the cheaper, weaker option

The `codex` backend runs the judge as a local subprocess with no API key, which
makes iterating free. It is not equivalent to a provider judge: GEval prefers
token logprobs to produce a continuous score, and a subprocess cannot supply
them, so its scores are coarser. Confirm a threshold on a provider judge before
you commit to it — that is the number your real test suite will see.

## The test cases may be generated

Cases can be written by a model from your description. When they are, they
encode that model's assumptions about your domain, and the sweep then measures
how well a variant satisfies those assumptions rather than your real
requirements.

Every case is editable for this reason. Reading and correcting the generated
cases is not optional polish — it is the step that makes the rest mean
something. A case you would not have written is a measurement you should not
trust.

## A metric measures its rubric, nothing more

A metric that cannot see what it needs is refused before the sweep rather than
scored against nothing: exact match needs an expected output on every case,
faithfulness needs retrieval context, tool correctness needs the tools called.
The warnings under **4 · Metrics** name the missing field. Removing a metric
because it complains is a decision about what you are no longer measuring.

## Failures stay visible

When a judge errors, times out, or answers with nothing, that score is recorded
as a **failure** and excluded from the mean. It is never replaced with a zero
(which would punish a variant for a network problem) or with a neutral half
(which would move the score with nothing on screen to explain it). A cell whose
metric failed is not reported as having cleared every threshold.

The same applies to a whole combination: one variant-and-model cell that errors
is shown as failed, and the rest of the sweep still runs.

## The arithmetic is deliberately simple

- A cell's **score** is the plain mean of the metric scores obtained for it.
  Metrics are not weighted against each other — a metric that matters more is
  one whose threshold you set higher, not one you weight up.
- **Passing** is separate from scoring: a cell clears only if *every* enabled
  metric met its own threshold, in its own direction. A high mean with one
  metric below its threshold is a failure, and is shown as one.
- **Ranking** is by score, with cost breaking ties. The configuration named at
  the bottom of the page is not the top of that ranking — it is the *cheapest*
  one that cleared everything, which is usually the answer and usually not the
  highest score.

Everything in that calculation is on screen next to the result, so any number
can be reproduced by hand. If it cannot, that is a bug.

## Cost is an estimate until it is spent

The figure shown before a sweep uses assumed token shapes — a fixed guess at
input and output length, and a fixed guess at what a judge call costs. It is
approximate, but it is approximate in dollars, which is the unit the decision is
made in. The cost-per-thousand shown *after* a sweep is different: it comes from
the tokens the provider actually reported. Where a model publishes no price, or
the provider reported no usage, the workbench says the cost is unknown rather
than showing zero.

A sweep costs roughly *frameworks × variants × models × cases* model calls,
plus *that many × judged metrics* judge calls. Two frameworks, two variants,
two models, eight cases and three judged metrics is 64 model calls and 192
judge calls. A round that sends tools costs more than one model call per
case, because a tool round-trip is a second call — the preview says which
assumption it used rather than quietly counting one.

## Scoring only happens when you ask

No edit, generation, or metric change ever starts a scoring call. This is
enforced in the import graph and covered by a test that fails if it stops being
true. The sweep is the only thing that spends, and it states its call count and
estimated cost before it runs.

## Not in this version

Everything lives in the browser session. Closing the tab loses it. Save/load,
import/export, and any durable project store are deliberately out of scope for
this release — see `PLAN.md` for what is scheduled and what is deferred.
