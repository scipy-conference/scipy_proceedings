---
# Ensure that this title is the same as the one in `myst.yml`
title: "Solving the AI Eval Gap: Domain-Aware Evaluation for Production AI Agents"
abstract: |
  AI agents are moving into enterprise production faster than the tooling used to evaluate
  them. Generic evaluation frameworks measure hallucination rate, latency, and token cost,
  but not the question that actually blocks a launch: did the agent do the *right thing* for
  this schema, these documents, this domain? Today every team hand-labels golden sets that
  go stale the moment data, prompts, or use cases change, producing non-reproducible
  pipelines that do not transfer across teams.

  We present **AI Eval Engine**, an open-source Python framework that treats evaluation as a
  pipeline to be *generated* rather than an artifact to be *authored*. A team points the
  framework at where its domain lives; the framework (1) ingests a pluggable domain context
  from a lightweight configuration, (2) generates a versioned, domain-grounded golden set,
  (3) emits a runnable evaluation script that scores correctness, evidence grounding, and
  output format, (4) accumulates a living **Safety Runbook** that clusters observed failures
  into domain-specific constraints, and (5) tracks the result as quarter-over-quarter
  objectives and key results. A central methodological concern for any pipeline that uses a
  language model both to generate tests and to judge them is *self-bias*; we address it by
  anchoring demonstrations on tasks whose correctness can be verified by execution or by
  grounding in source documents, and where the model demonstrably fails without the ingested
  domain context. We illustrate the framework on two contrasting agent shapes — a structured,
  execution-scored scientific-coding agent (ScienceAgentBench) and an open-ended,
  grounding-scored financial document question-answering agent (FinanceBench) — showing that
  one reusable pipeline spans both without per-domain scratch tooling.
---

## Introduction

AI agents are moving from prototypes into enterprise production at a pace that has outrun the
tooling used to evaluate them. Teams shipping agents over financial filings, scientific
datasets, internal knowledge bases, and customer-support transcripts repeatedly hit the same
wall: existing evaluation tooling measures generic {abbr}`LLM (large language model)`
properties — hallucination rate, latency, token cost, refusal frequency — but does not answer
the question that actually blocks a launch: *did the agent do the right thing in this domain?*

Domain correctness is not a generic property. A data agent over a financial schema and a
{abbr}`RAG (retrieval-augmented generation)` agent over scientific documents require
fundamentally different notions of "correct," and those notions only exist in the **domain
context**. A scientific-analysis agent must load the right columns from a specific dataset and
compute the metric the paper actually reported; a financial agent must ground every figure it
states in the filing it was given and decline when the filing does not support an answer.
Neither constraint is visible to a generic judge that scores token-level fluency.

The current state of practice for closing this gap is the **manually labeled golden set**: a
domain expert writes representative queries and expected answers in a spreadsheet and hands
the file to the engineering team. This approach has three well-known failure modes:

1. **Staleness.** The moment the underlying data changes, prompts are tuned, or a new use case
   is added, the golden set drifts out of alignment with what the agent actually does.
2. **Non-reproducibility.** The finance team writes its own; the science team does the same;
   none of the pipelines are comparable, and none can be reused across teams.
3. **No clarity on what to fix.** A pass/fail score against a hand-written golden set tells the
   team *that* something is broken, not *why*, nor which domain constraint was violated.

These are not bugs in the evaluation tools; they are consequences of treating evaluation as an
artifact to be authored rather than a pipeline to be generated.

This paper presents **AI Eval Engine**, an open-source Python framework that reframes
evaluation as a domain-driven generator. The team points at where the domain lives (a dataset,
a document corpus, a database); the framework produces a versioned golden set, a runnable
evaluation script, and a living Safety Runbook that captures emergent failure modes as they
appear. The human role shifts from *author* to *reviewer*. The pipeline becomes reusable
across domains and rerunnable as data evolves, which in turn lets teams track agent behavior as
a first-class quarter-over-quarter {abbr}`OKR (objective and key result)` rather than a
one-time launch check.

The framework leans on a language model — Anthropic's Claude [@anthropic2025claude], invoked
through the Claude {abbr}`API (application programming interface)` and the Claude Code
development workflow — for context extraction, golden set generation, evaluation-script
generation, and the {abbr}`LLM (large language model)`-as-judge step. Using a model to both
*generate* and *judge* tests raises a real methodological objection — self-bias — which we
confront directly in [](#methodology) and which drives our choice of demonstration datasets.

## Background and Related Work

Open-source evaluation frameworks tackle adjacent slices of the problem. **RAGAS** scores
retrieval-augmented systems on faithfulness, answer relevancy, and context recall using
LLM-as-judge primitives over user-supplied datasets [@es2024ragas]. **ARES** trains lightweight
judges for RAG evaluation [@saadfalcon2024ares]. **G-Eval** formalizes the LLM-as-judge pattern
with chain-of-thought scoring [@liu2023geval], and the broader practice of using strong models
as judges was characterized by @zheng2023llmjudge. Holistic suites such as HELM standardize
*which* metrics are reported across models [@liang2023helm]. **DeepEval** [@deepeval] provides a
pytest-style harness for LLM outputs, while **Arize Phoenix** [@phoenix] and **Comet Opik**
[@opik] focus on observability of production traffic. In every case the *test cases themselves*
are assumed to already exist; the framework scores them.

The closest neighbors generate tests rather than assume them. @guinet2024examgen generate
task-specific exams from a corpus to evaluate RAG systems; EvalGen aligns LLM-assisted
evaluators with human preferences [@shankar2024evalgen]; SPADE synthesizes data-quality
assertions for LLM pipelines [@shankar2024spade]. Recent agent-centric systems — the
Agent-Testing Agent [@ata2025], TestAgent for vertical domains [@testagent2024], and
$\tau$-bench for tool-agent-user interaction under policy [@yao2024taubench] — push toward
automated, domain-specific benchmarking. Surveys of agent evaluation
[@yehudai2025survey; @mohammadi2025survey] and domain safety benchmarks such as TRIDENT
[@hui2025trident] and DecodingTrust [@wang2023decodingtrust] map the surrounding landscape.

AI Eval Engine differs in two ways. First, it treats the **automated generation of the test
set itself**, driven by a pluggable domain context, as the primary contribution — not a
preprocessing step. Second, it accumulates a **living Safety Runbook** of domain-specific
constraints surfaced from observed failure patterns, closing a loop from evaluation back into
test generation. It is designed to be used *alongside* the frameworks above: RAGAS or DeepEval
metrics can be registered as additional scorers inside Step 3. Self-improving agent methods
such as Agentic Context Engineering [@zhang2026ace] are candidates for evaluation *by* this
framework, not components of it.

## Methodology: Domain-Aware Evaluation Without Self-Bias

(methodology)=

A pipeline that uses an LLM to *generate* the golden set and the same family of models to
*judge* the agent invites an obvious objection: the evaluation may simply reward outputs that
look like what the generator would itself produce. This **self-bias** is the central
methodological risk of LLM-as-benchmark-generator plus LLM-as-judge pipelines
[@silencer2025], and it is the first thing a careful reviewer will probe.

We adopt a single guiding principle that defuses self-bias, benchmark saturation, and training
contamination at once: **choose tasks where the model fails without the ingested domain
context, and where correctness can be checked independently of the judge.** Two consequences
follow.

- **Prefer verifiable correctness over judge opinion.** Where a task's output can be *executed*
  (does the generated program run and produce the expected artifact?) or *grounded* (is the
  stated figure supported by the cited evidence?), correctness is decided by the world, not by
  a model's preference. The LLM-as-judge is reserved for the genuinely open-ended residue,
  never used as the sole arbiter on a task that execution or grounding can settle.
- **Prefer domains the model could not have memorized.** Generic code generation
  (e.g., HumanEval-style problems) is saturated and largely present in pretraining, so a high
  score measures recall, not domain-aware evaluation. The contribution only becomes meaningful
  on tasks that require context — a specific dataset, a particular filing — that the model was
  never trained on.

This principle is what makes the demonstration datasets in [](#use-cases) load-bearing rather
than decorative: each was selected because the agent cannot succeed on it without the domain
context the framework ingests, and each admits an execution- or grounding-based check that does
not depend on the judge agreeing with the generator.

## The Five-Step Framework

The framework decomposes evaluation into five steps, each implemented as an independent Python
module with a stable interface so teams can replace, extend, or skip a step without forking the
pipeline. @fig:pipeline shows the flow and the feedback loop from the Safety Runbook back into
golden set generation.

<!-- TODO: replace figure1.png with the final five-step pipeline diagram (source:
abstract_flowchart.png in the ai-eval-engine repo). Placeholder kept so the paper builds. -->

:::{figure} figure1.png
:label: fig:pipeline
The AI Eval Engine pipeline. A pluggable domain context (Step 1) drives golden set generation
(Step 2) and evaluation-script generation and scoring (Step 3). Failure clusters feed a living
Safety Runbook (Step 4) whose new constraints flow back into Step 2, and a monitoring dashboard
(Step 5) tracks results over time.
:::

### Step 1 — Pluggable Domain Context Ingestion

The user provides a lightweight YAML configuration that points to one or more domain sources:

```yaml
project: financebench-qa-agent
domain_sources:
  - type: csv
    path: data/financebench/financebench.csv
    description: >-
      open-book QA over corporate 10-K filings; each row is a question
      grounded in a filing with an evidence-linked answer

stratify_by: company
sample_per_stratum: 2
sample_seed: 42

# how each row is read as an evaluation task (Steps 2-5)
task:
  kind: grounded_qa
  id_field: financebench_id
  input_field: question
  gold_field: answer
  grounding_field: evidence
  category_field: company
```

The framework reads a representative, stratified sample from each source and issues a Claude API
call that extracts the *implicit* domain structure into a typed `DomainContext` object — not a
prose summary. The extraction captures the categories and taxonomies present in the data, the
safety and compliance constraints the domain implies (e.g., "every stated figure must trace to
a cited passage"), and quality signals (distributions, ambiguity clusters, missing-value
patterns). Users are **not** asked to hand-author a taxonomy or compliance document; they point
to where the data lives, and the framework surfaces what matters. Stratified sampling ensures
the extraction sees the breadth of the domain rather than only its most common cases.

### Step 2 — Automated Golden Set Generation

From the `DomainContext`, the framework constructs a versioned, domain-grounded test set of
`GoldenCase`s. Each case carries the input query, the expected outcome (an executable check, an
expected artifact, or an evidence-grounded reference answer), a *kind*
(`normal`, `ambiguous`, `out_of_scope`, or `safety_boundary`), a *difficulty*
(`easy`, `medium`, `hard`), and the constraint it probes. Cases are constructed two ways. In
**generative** mode, Claude authors fresh cases grounded in real domain evidence, spanning
happy-path lookups through hard multi-step computations, definitionally ambiguous queries, and
out-of-scope questions the agent must decline — for the FinanceBench domain this produces cases
such as *"What is Amazon's FY2017 gross margin?"*, whose correct response is to flag that the
income statement carries no gross-profit line and state the assumption, and out-of-scope traps
such as requesting free cash flow from a balance sheet alone (expected: refuse). In
**normalize** mode, the framework adopts a public benchmark's own verified labels as cases,
anchoring the evaluation to externally trusted ground truth. `safety_boundary` cases are
synthesized from the `DomainContext`'s constraints in either mode, so every domain rule is
always probed.

Both modes enforce one rule: every figure in an expected answer must be traceable to cited
evidence, or the case becomes an explicit refusal — the self-bias guard from
[](#methodology), letting a reviewer (or, where a public split exists, the benchmark's own gold
answer) verify each case without trusting the generator. The runs in [](#sec:results) use
normalize mode on FinanceBench and ScienceAgentBench so our numbers sit directly beside each
benchmark's published baseline; generative mode is the framework's more ambitious capability and
is demonstrated separately. Golden sets are content-addressed (`goldensets/<version>/`) so reruns
can be diffed, and a human reviewer may accept, edit, or reject any case before it is promoted —
the human role is **reviewer**, not author.

### Step 3 — Eval Script Generation and Scoring

The framework emits a runnable Python evaluation script that runs the agent against the golden
set and produces three orthogonal scores per case:

- **Correctness** — for executable cases, whether the produced program runs and yields the
  expected result (decided by execution in a subprocess); for grounded cases, a normalized /
  numeric match against the gold answer.
- **Grounding** — for grounded cases, the fraction of the predicted answer that is supported by
  the cited evidence, computed deterministically (token and numeric overlap against the evidence
  string) so a fluent but unsupported figure is penalized.
- **Format validation** — structural integrity of the output (parses, required fields present,
  output bounds), evaluated deterministically without a model call.

All three scorers shipped for the runs in [](#sec:results) are **deterministic** — no model
grades another model's output. The {abbr}`LLM (large language model)`-as-judge remains the
designed fallback for the genuinely open-ended residue (tasks that can be neither executed nor
grounded), but it was *not* exercised in these runs. This is the operational expression of the
anti-self-bias principle in [](#methodology): the judge is never the sole arbiter on a task that
can be checked by execution or grounding.

### Step 4 — Agentic Safety Runbook

The framework clusters failures across a run by question type, domain area, and likely root
cause, and writes them to a living, incremental **Safety Runbook** — a domain-specific Markdown
document that accumulates emergent constraints, failure modes, and insights over time. Safety is
not declared upfront; it is *discovered* from how the agent behaves in its domain. A failure
cluster surfaced in run $N{+}1$ becomes a new Runbook section, and the Step 2 generator picks it
up as a constraint to probe in the next golden set — closing the loop between observation and
test generation.

### Step 5 — Post-Launch Monitoring Dashboard

Once the pipeline has run, the framework becomes a live OKR tracker. As underlying data changes,
golden sets and the Runbook regenerate on rerun, enabling quarter-over-quarter tracking of:

- **Agent Safety Score** — aggregate pass rate on safety-critical cases (refusals, out-of-scope
  handling, grounding compliance).
- **Domain Accuracy Score** — correctness on in-domain queries, tracking whether the agent still
  performs as the data evolves.
- **Drift Indicator** — share of *new* failure patterns since the last run; high drift signals a
  shifting domain.
- **Coverage Score** — share of the discovered domain context represented in the current golden
  set, catching blind spots before production.

## Demonstration Use Cases

(use-cases)=

We demonstrate the framework on two contrasting, domain-dependent agent shapes that bracket the
spectrum the framework targets: a **structured, execution-scored** agent and an **open-ended,
grounding-scored** agent. Both were chosen under the principle of [](#methodology) — the agent
cannot succeed without the ingested domain context, and correctness is checkable independently
of the judge. @tbl:usecases summarizes the contrast.

```{list-table} The two demonstration agents bracket the agent-shape spectrum: structured and execution-verifiable versus open-ended and grounding-verifiable. One pipeline serves both.
:label: tbl:usecases
:header-rows: 1
* - Dimension
  - Use Case 1 — ScienceAgentBench
  - Use Case 2 — FinanceBench
* - Agent shape
  - Scientific data-analysis coder
  - Open-book document QA (RAG)
* - Output
  - Self-contained Python program
  - Free-text answer with evidence
* - Correctness check
  - Execution of the program
  - Grounding in the cited filing
* - "Unsafe" means
  - Wrong/unsafe analysis, silent failure
  - Hallucinated figure, wrong refusal
* - Why context is required
  - Specific dataset and task knowledge
  - Specific filing the model never saw
```

### Use Case 1 — Scientific-Coding Agent on ScienceAgentBench (Primary Demo)

The agent under test is a data-analysis coder: given a scientific task and a dataset, it
produces a self-contained Python program. We use **ScienceAgentBench** [@chen2024scienceagentbench],
102 tasks drawn from 44 peer-reviewed papers across four disciplines, where each task's output
is a program **scored by execution**. The best agents in the original study solve roughly a
third of tasks, so the benchmark is far from saturated. Its structure — a dataset plus optional
expert-provided knowledge as input — maps directly onto Step 1's domain-context ingestion, and
its execution-based scoring makes it self-bias-proof: a program either reproduces the expected
artifact or it does not. The benchmark is also SciPy-native, exercising exactly the data-driven
scientific computing the audience builds.

What the framework showcases here: Step 1 ingests the task's dataset and expert knowledge into a
`DomainContext`; Step 2 generates probes spanning normal tasks, ambiguous specifications, and
safety-boundary cases (e.g., silently producing a plausible-but-wrong figure); Step 3 scores by
executing the produced program and validating the artifact, reserving the judge for open-ended
explanation quality; Step 4 surfaces clusters such as "loads the wrong columns" or "ignores the
requested metric." The public ScienceAgentBench split runs alongside as a sanity baseline.

### Use Case 2 — Financial Document QA on FinanceBench (Contrast Demo)

The contrasting agent answers open-ended questions over real corporate filings. We use
**FinanceBench** [@islam2023financebench], open-book question answering over 10-K filings with
10,231 evidence-linked questions; in the original study a strong retrieval-augmented model
answered incorrectly or refused on roughly four-fifths of a sampled set, so the task is
genuinely hard and not memorized. The agent **must** ground its answer in the supplied
documents, and the safety dimension is intrinsic: a hallucinated financial figure or an
incorrect refusal on a regulated domain is the failure mode that matters. This is the opposite
shape from Use Case 1 — prose rather than code, grounding rather than execution — yet the same
pipeline applies.

What the framework showcases here: Step 1 extracts the filing taxonomy and grounding constraints;
Step 2 generates queries including evidence-required cases and out-of-scope cases where the
correct behavior is refusal; Step 3 scores answer correctness against cited evidence (drawing on
hallucination-labeling resources such as RAGTruth [@wu2024ragtruth] for the grounding
dimension) and output format; Step 4 surfaces clusters such as "states an unsupported
figure" or "refuses despite available evidence."

Together, the two use cases demonstrate that one pipeline operates across a structured,
execution-verified coding agent and an open-ended, grounding-verified QA agent — fundamentally
different domains with fundamentally different correctness criteria — without per-domain scratch
tooling.

## Results

(sec:results)=

We report a real end-to-end run of the framework on FinanceBench, scored by the offline
grounding-based scorer, and a structural ablation on ScienceAgentBench anchored to its
published expert-knowledge gap. Every number below is reproducible from the committed
configuration and the saved `DomainContext`; predictions were produced through the Claude Code
workflow [@anthropic2025claude]. Throughout, the experimental variable is **Step 1**: the
*same* agent predictions are scored by an evaluation generated *with* the ingested domain
context and *without* it, isolating what the domain context contributes.

### FinanceBench: a real, grounding-scored run

The agent under test answered a stratified sample of 61 FinanceBench questions (drawn across
32 companies) using only each question's supplied filing evidence; the framework scored each
answer for **correctness** (normalized / numeric match to the reference) and **grounding**
(whether the figures and salient terms in the answer are supported by the cited evidence),
with a pass threshold of 0.6. @tbl:fb-ablation reports the ablation.

```{list-table} FinanceBench ablation. The same 61 agent predictions are scored by the generated evaluation with Step-1 domain context off versus on. Normal-case correctness and grounding are identical by construction (the predictions are unchanged); turning the domain context on *adds* a five-test safety/compliance dimension and a safety-clustered runbook that the generic evaluation structurally cannot produce.
:label: tbl:fb-ablation
:header-rows: 1
* - Metric
  - Without domain context
  - With domain context
* - Normal questions scored
  - 61
  - 61
* - Pass rate
  - 0.557
  - 0.591
* - Domain accuracy
  - 0.595
  - 0.595
* - Grounding rate
  - 0.780
  - 0.780
* - Safety-boundary cases
  - 0
  - 5 (all handled)
* - Categories surfaced
  - 32 (by company)
  - 33 (+ safety)
```

The agent answered 34 of 61 questions correctly (55.7% pass rate) at 78.0% mean grounding,
with domain accuracy of 0.595 — consistent with FinanceBench being a genuinely hard,
non-saturated task. The decisive observation is *not* an accuracy swing between the two arms:
because the predictions are identical, normal-case correctness and grounding are unchanged.
What the domain context adds is an evaluation **dimension**. With Step 1 on, the framework
auto-generates five domain-specific safety-boundary tests — ground every figure in cited
evidence, refuse when the filing does not support an answer, respect the stated statement
scope, decline personalized investment advice, and show the line items behind a computed
metric — none of which exist in the generic arm. The agent handled all five, and Step 4
clustered the run's failures into 23 itemized runbook entries by company and failure type
(27 wrong-value, 1 ungrounded). A generic evaluation of the *same* outputs reports a single
flat pass rate with no safety dimension and no actionable clustering (@fig:fb-ablation).

:::{figure} fb_ablation.pdf
:label: fig:fb-ablation
:width: 100%
Our ablation, baseline (Step-1 domain context off) versus domain-aware (on); all numbers are
from our own runs. **(a)** FinanceBench scored metrics: domain accuracy and grounding are
identical by construction (the predictions are unchanged), pass rate ticks up only because the
five added safety cases all pass, and the baseline has **no safety dimension to score at all**.
**(b)** What domain context unlocks in the *generated evaluation* on both datasets — extra
golden cases (FinanceBench 61→66, ScienceAgentBench 12→17), five safety-boundary tests where
the baseline has none, per-category accuracy, grounding/refusal checks, failure-clustered
runbook, and a safety OKR score. The baseline columns are the same agent outputs scored without
domain context.
:::

Manual inspection of the 27 wrong-value cases yields a methodologically useful finding: a
substantial share are **definitional or lexical mismatches rather than substantive errors**.
For example, the agent computed Corning's working capital as total current assets minus
current liabilities ($2,278\,\mathrm{M}$) where the reference used a narrower operating
definition ($831\,\mathrm{M}$); answered that gross margin "is not meaningful for a bank" —
semantically equivalent to the reference's "not a relevant metric" but scored low by lexical
matching; and reported a cash decline as an absolute ($\$781\,\mathrm{M}$) where the reference
gave a percentage ($\sim 42\%$). These are exactly the cases the grounding score rates highly
even as lexical correctness fails, and they argue for two of the framework's design choices:
reporting grounding alongside correctness, and letting the ingested domain context pin
down domain-specific definitions (here, which "working capital" the domain means) so the
generated golden set encodes them. We report the unadjusted pass rate and flag this caveat
rather than hand-tune the metric. @fig:grounding-scatter makes the pattern visible: 18 of the
scored cases sit in the high-grounding, low-correctness region.

:::{figure} grounding_scatter.pdf
:label: fig:grounding-scatter
:width: 85%
Per-case correctness versus grounding on the FinanceBench run. Each point is one question;
the highlighted cluster (lower-right) is the 18 cases whose answers are well supported by the
cited evidence (grounding ≥ 0.75) yet score low on lexical/numeric correctness — definitional
or lexical mismatches (e.g. the Corning working-capital definition, the JPM "gross margin"
phrasing, the Best Buy absolute-vs-percentage case) rather than substantive errors. This is
why the framework reports grounding alongside correctness instead of collapsing to a single
pass/fail.
:::

:::{figure} per_company.pdf
:label: fig:per-company
:width: 70%
The same run broken out by company (Step 5 coverage view). A generic evaluation reports one
flat number (the dashed line, 0.59); the domain-aware evaluation turns it into a per-company
map that says *where* to look — bars above the mean in blue, below in grey. This per-category
breakdown exists only because Step 1 surfaced the company taxonomy.
:::

### ScienceAgentBench: structural ablation and the published expert-knowledge gap

For the execution-scored coding agent we report the **structural** effect of domain context
on the generated evaluation, holding the correctness side to the benchmark's own published
numbers. Running Step 1 on the ScienceAgentBench tasks and generating the golden set with and
without the resulting `DomainContext` changes the evaluation as shown in @tbl:sab-structural.

```{list-table} ScienceAgentBench structural ablation. Domain context adds five domain-specific safety-boundary tests and a safety dimension to the generated evaluation; the generic evaluation has neither.
:label: tbl:sab-structural
:header-rows: 1
* - Generated evaluation contains
  - Without domain context
  - With domain context
* - Golden cases
  - 12 (normal only)
  - 17 (12 normal + 5 safety-boundary)
* - Safety dimension
  - none
  - safety score + safety category
* - Categories surfaced
  - 4 disciplines
  - 4 disciplines + safety
```

The five generated safety-boundary tests are domain failure modes the extractor surfaced from
the tasks themselves — compute the requested metric rather than a proxy (flagged critical),
load the specified files and columns, write the artifact to the exact output path, fail loudly
on missing dependencies or data, and preserve domain-method semantics. They exist only because
Step 1 ingested the domain; a generic eval over the same tasks tests none of them.

For the correctness dimension we anchor to ScienceAgentBench's own published result: the best
agents solve roughly 32% of tasks, rising to about 42% when supplied with **hand-authored
expert knowledge** [@chen2024scienceagentbench]. That published 32%→42% gap is itself a
with-versus-without-domain-knowledge ablation, validated by the benchmark's authors. Our Step-1
`DomainContext` is precisely an *automated* replacement for that hand-authored expert
knowledge, so we frame our contribution against a baseline the source benchmark already
established rather than one we defend from scratch. A full execution-scored run of generated
programs is left to future work (see [](#sec:limitations)); the benchmark's
heavyweight per-task environment is orthogonal to the domain-aware evaluation claim.

### Summary

:::{figure} okr_radar.pdf
:label: fig:okr-radar
:width: 70%
The FinanceBench run as a safety/quality OKR profile (Step 5), our pipeline with domain
context off (baseline) versus on. Domain accuracy, grounding, and pass rate are near-identical
by construction — the predictions are unchanged — so the two profiles overlap on three axes.
The baseline **collapses to zero on the safety axis**: with Step 1 off, the generated
evaluation has no safety dimension to score at all. Tracking that axis quarter over quarter is
the proposal's "safety as a first-class OKR."
:::

Across both demonstrations the result is consistent: scoring identical agent behavior, the
domain-aware evaluation surfaces a safety/compliance dimension and actionable failure
clustering that a generic evaluation of the same outputs cannot — on FinanceBench as five
grounding-and-refusal tests with a 23-entry runbook from a real run, and on ScienceAgentBench
as five execution-safety tests (@fig:fb-ablation), anchored in the prose above to the
benchmark's published expert-knowledge gap. The delta is the evaluation content that exists
*only because Step 1 ran* (@fig:okr-radar).

## Implementation Notes

The framework is implemented in Python (3.11+). All five steps are implemented and exercised in
the runs of [](#sec:results): a Pydantic-typed `DomainContext` extractor with a stratified sampler
and cached system prompt (Step 1), versioned content-addressed golden-set generation (Step 2),
the `grounded_qa` and `code_execution` scorers and a generated standalone eval runner (Step 3),
an accumulating itemized Safety Runbook (Step 4), and a self-contained HTML dashboard (Step 5).
Only Step 1's context extraction requires a model call; Steps 2–5 run fully offline and
deterministically, so the golden set, scoring, runbook, and dashboard reproduce without an API
key. Key choices:

- **Model:** Claude via the Anthropic API [@anthropic2025claude], with prompt caching on the
  `DomainContext` and Safety Runbook prompts to amortize cost across reruns.
- **Datasets:** both demos use public data with independent correctness checks — execution for
  ScienceAgentBench, document grounding for FinanceBench.
- **Configuration:** YAML; no code is required to onboard a new domain.
- **Artifacts:** all generated files (golden set, eval script, Runbook) are plain text and
  Git-friendly, so versioning and review use normal code-review tooling.
- **Retrieval (Use Case 2):** Chroma [@chroma] as a local vector store with a contextual-retrieval
  chunking strategy.

The repository is available under an MIT license; a live demonstration will accompany the SciPy
2026 talk.

## Discussion

**Reviewer, not author.** The framework keeps a human in the loop at every generation step. This
is not an admission that the model cannot do the job; a generated artifact a domain expert
*accepted* carries different organizational weight than one a model produced alone. The framework
optimizes for *acceptance latency* — making artifacts easy to skim and edit — rather than full
autonomy.

**Discovered, not declared.** The Safety Runbook grows from observed failures rather than from an
upfront enumeration of constraints. The wager is that for most enterprise domains, teams do not
know the full constraint list until they have watched the agent fail in a few characteristic
ways; the Runbook captures that learning and feeds it back into test generation.

**Verifiability as a design constraint.** Selecting demonstration datasets by *checkability*
rather than convenience is what lets the paper claim domain-aware evaluation without conceding to
the self-bias objection. The same constraint guides adoption: the framework is most trustworthy
on domains where some independent check — execution, grounding, schema validation — exists, and
its judged scores should be read as weaker evidence than its verified ones.

(sec:limitations)=
## Limitations and Future Work

The current scope targets text-input/text-output and code-producing agents reached through an
HTTP endpoint; multimodal and long-horizon tool-using agents are future work, though Step 1's
extractor is designed to generalize. The Safety Runbook is a single Markdown file today and will
need per-sub-domain splitting for very large domains. The Step 5 dashboard is read-only;
trend visualization is planned. Finally, an execution-verifiable text-to-SQL slice (e.g., a small
BIRD subset) is a natural third demonstration left for future work.

## Conclusion

AI Eval Engine treats evaluation as a *pipeline to be generated*, not an artifact to be authored.
By driving the pipeline from a pluggable domain context, generating the golden set and evaluation
script automatically, scoring with verifiable checks wherever possible, and accumulating a living
Safety Runbook across runs, the framework gives teams a reusable, rerunnable, domain-aware path
to tracking agent behavior as a first-class OKR. Demonstrations on an execution-scored
scientific-coding agent and a grounding-scored financial-QA agent show one pipeline spanning two
fundamentally different domains — with self-bias addressed by construction rather than by
assertion.

## Availability

Source code is released under the MIT license. The SciPy 2026 talk page and the public
repository are linked from the proceedings entry.

## Generative AI Disclosure

In accordance with the SciPy generative AI policy, the author discloses that generative AI tools
(Anthropic Claude, via the Claude API and Claude Code) were used in two capacities: (1) as the
*subject* of this work — the framework described here invokes Claude for context extraction,
golden set generation, evaluation-script generation, and LLM-as-judge scoring; and (2) as a
*writing aid* — Claude assisted with drafting and refining prose and code examples in this
manuscript. All outputs were reviewed, verified, and revised by the author, who takes full
responsibility for the accuracy and integrity of the final content.

## Acknowledgments

This work is an independent open-source contribution to the SciPy 2026 Proceedings, separate from
the author's employer affiliation.
