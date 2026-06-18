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

AI agents are moving into enterprise production faster than the tooling used to evaluate them.
Teams shipping agents over financial filings, scientific datasets, knowledge bases, and support
transcripts hit the same wall: existing tooling measures generic {abbr}`LLM (large language
model)` properties — hallucination rate, latency, token cost, refusal frequency — but not the
question that blocks a launch: *did the agent do the right thing in this domain?*

Domain correctness is not a generic property. A data agent over a financial schema and a
{abbr}`RAG (retrieval-augmented generation)` agent over scientific documents require different
notions of "correct," and those notions exist only in the **domain context**: a scientific agent
must load the right columns and compute the metric the paper reported; a financial agent must
ground every figure in its filing and decline when the filing does not support an answer. Neither
constraint is visible to a generic judge scoring token-level fluency.

The state of practice is the **manually labeled golden set**, with three well-known failure modes:
it goes *stale* the moment data, prompts, or use cases change; it is *non-reproducible*, each team
building its own incomparable set; and it gives *no clarity on what to fix*, reporting only that
something broke, not which domain constraint was violated. These follow from treating evaluation
as an artifact to be *authored* rather than a pipeline to be *generated*.

This paper presents **AI Eval Engine**, an open-source Python framework that reframes evaluation
as a domain-driven generator. The team points at where the domain lives; the framework produces a
versioned golden set, a runnable evaluation script, and a living Safety Runbook that captures
emergent failure modes — shifting the human role from *author* to *reviewer* and letting teams
track agent behavior as a first-class quarter-over-quarter {abbr}`OKR (objective and key result)`.
It leans on a language model — Anthropic's Claude [@anthropic2025claude], via the Claude
{abbr}`API (application programming interface)` and Claude Code — for context extraction, golden
set generation, eval-script generation, and the {abbr}`LLM (large language model)`-as-judge step.
Using one model to both *generate* and *judge* tests raises a real objection — self-bias — which
we confront in [](#methodology) and which drives our dataset choice.

## Background and Related Work

Open-source evaluation frameworks tackle adjacent slices. **RAGAS** scores RAG systems on
faithfulness, answer relevancy, and context recall [@es2024ragas]; **ARES** trains lightweight RAG
judges [@saadfalcon2024ares]; **G-Eval** formalizes LLM-as-judge with chain-of-thought scoring
[@liu2023geval], a practice characterized by @zheng2023llmjudge; HELM standardizes *which* metrics
are reported [@liang2023helm]; **DeepEval** [@deepeval] is a pytest-style harness, and **Arize
Phoenix** [@phoenix] and **Comet Opik** [@opik] focus on production observability. In every case
the *test cases* are assumed to exist. The closest neighbors instead generate them:
@guinet2024examgen build task-specific exams from a corpus, EvalGen aligns LLM evaluators with
human preferences [@shankar2024evalgen], and SPADE synthesizes data-quality assertions
[@shankar2024spade]; agent-centric systems — the Agent-Testing Agent [@ata2025], TestAgent
[@testagent2024], and $\tau$-bench [@yao2024taubench] — push toward domain-specific benchmarking,
with surveys [@yehudai2025survey; @mohammadi2025survey] and safety benchmarks TRIDENT
[@hui2025trident] and DecodingTrust [@wang2023decodingtrust] mapping the landscape.

AI Eval Engine differs in two ways: it treats **automated generation of the test set itself**,
driven by a pluggable domain context, as the primary contribution; and it accumulates a **living
Safety Runbook** of domain constraints from observed failures, closing a loop back into
generation. It is meant to be used *alongside* these tools — RAGAS or DeepEval metrics register as
extra Step-3 scorers — and self-improving methods such as Agentic Context Engineering
[@zhang2026ace] are candidates for evaluation *by* it, not components of it.

## Methodology: Domain-Aware Evaluation Without Self-Bias

(methodology)=

A pipeline that uses an LLM to *generate* the golden set and the same model family to *judge* the
agent invites an obvious objection: the evaluation may simply reward outputs that look like what
the generator would produce. This **self-bias** is the central risk of LLM-as-benchmark-generator
plus LLM-as-judge pipelines [@silencer2025], and the first thing a reviewer will probe.

We adopt a single guiding principle that defuses self-bias, benchmark saturation, and training
contamination at once: **choose tasks where the model fails without the ingested domain
context, and where correctness can be checked independently of the judge.** Two consequences
follow.

- **Prefer verifiable correctness over judge opinion.** Where output can be *executed* (does the
  program produce the expected artifact?) or *grounded* (is the figure supported by cited
  evidence?), correctness is decided by the world, not the model; the judge is reserved for the
  open-ended residue.
- **Prefer domains the model could not have memorized.** Generic code generation (HumanEval-style)
  is saturated and largely in pretraining, so a high score measures recall, not domain-aware
  evaluation; the contribution is meaningful only on tasks needing context — a specific dataset or
  filing — the model never saw.

This principle makes the datasets in [](#use-cases) load-bearing: each was chosen because the
agent cannot succeed without the ingested domain context, and each admits an execution- or
grounding-based check independent of the judge.

## The Five-Step Framework

The framework decomposes evaluation into five steps, each an independent Python module with a
stable interface so teams can replace, extend, or skip a step without forking the pipeline. A
pluggable domain context (Step 1) drives golden-set generation (Step 2) and eval-script generation
and scoring (Step 3); failure clusters feed a living Safety Runbook (Step 4) whose new constraints
flow back into Step 2, and a monitoring dashboard (Step 5) tracks results over time.

### Step 1 — Pluggable Domain Context Ingestion

The user provides a lightweight YAML configuration that points to one or more domain sources:

```yaml
project: financebench-qa-agent
domain_sources:
  - type: csv
    path: data/financebench/financebench.csv
    description: open-book QA over corporate 10-K filings
stratify_by: company
sample_per_stratum: 2
task:                      # how each row becomes an eval task
  kind: grounded_qa
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

From the `DomainContext`, the framework constructs a versioned, domain-grounded set of
`GoldenCase`s, each carrying the input, the expected outcome (an executable check, an artifact, or
an evidence-grounded answer), a *kind* (`normal`, `ambiguous`, `out_of_scope`, `safety_boundary`),
a *difficulty*, and the constraint it probes. In **generative** mode Claude authors fresh cases
grounded in real evidence — from happy-path lookups to hard multi-step computations, ambiguous
queries, and out-of-scope traps the agent must decline (e.g. asking for free cash flow from a
balance sheet alone). In **normalize** mode it adopts a benchmark's own verified labels. In both,
`safety_boundary` cases are synthesized from the `DomainContext`'s constraints, so every domain
rule is always probed.

Both modes enforce one rule: every figure in an expected answer must be traceable to cited
evidence, or the case becomes an explicit refusal — the self-bias guard from
[](#methodology), letting a reviewer (or, where a public split exists, the benchmark's own gold
answer) verify each case without trusting the generator. The FinanceBench result in
[](#sec:results) uses **generative** mode — the framework authors its own golden set, which is
exactly what we evaluate there — while the ScienceAgentBench comparison anchors to that
benchmark's published expert-knowledge baseline. Golden sets are content-addressed (`goldensets/<version>/`) so reruns
can be diffed, and a human reviewer may accept, edit, or reject any case before it is promoted —
the human role is **reviewer**, not author.

### Step 3 — Eval Script Generation and Scoring

The framework emits a runnable Python evaluation script producing three orthogonal scores per
case: **correctness** (execution of the produced program for executable cases, or a normalized /
numeric match for grounded ones), **grounding** (the fraction of the answer supported by the cited
evidence, by deterministic token/numeric overlap, so a fluent but unsupported figure is
penalized), and **format validation** (structural integrity, checked without a model call). All
three scorers in [](#sec:results) are **deterministic** — no model grades another model's output.
The {abbr}`LLM (large language model)`-as-judge is the designed fallback for the genuinely
open-ended residue but was not exercised here: the judge is never the sole arbiter on a task that
execution or grounding can settle.

### Step 4 — Agentic Safety Runbook

The framework clusters failures across a run by question type, domain area, and likely root
cause, and writes them to a living, incremental **Safety Runbook** — a domain-specific Markdown
document that accumulates emergent constraints, failure modes, and insights over time. Safety is
not declared upfront; it is *discovered* from how the agent behaves in its domain. A failure
cluster surfaced in run $N{+}1$ becomes a new Runbook section, and the Step 2 generator picks it
up as a constraint to probe in the next golden set — closing the loop between observation and
test generation.

### Step 5 — Post-Launch Monitoring Dashboard

Once the pipeline has run, the framework becomes a live OKR tracker: as data changes, golden sets
and the Runbook regenerate on rerun, enabling quarter-over-quarter tracking of an **Agent Safety
Score** (pass rate on safety-critical cases), a **Domain Accuracy Score** (correctness on
in-domain queries), a **Drift Indicator** (share of new failure patterns since the last run), and
a **Coverage Score** (share of the discovered domain context represented in the golden set).

## Demonstration Use Cases

(use-cases)=

We demonstrate the framework on two contrasting, domain-dependent agent shapes that bracket the
spectrum it targets: a **structured, execution-scored** scientific coder (ScienceAgentBench) and
an **open-ended, grounding-scored** document-QA agent (FinanceBench) — Python program versus
free-text-with-evidence, execution versus grounding, both failing without the ingested domain
context. Both were chosen under the principle of [](#methodology): correctness is checkable
independently of the judge, and one pipeline serves both.

### Use Case 1 — Scientific-Coding Agent on ScienceAgentBench (Primary Demo)

Given a scientific task and a dataset, the agent produces a self-contained Python program. We use
**ScienceAgentBench** [@chen2024scienceagentbench] — 102 tasks from 44 peer-reviewed papers across
four disciplines, each **scored by execution** (the best agents solve roughly a third, so it is
far from saturated). Its *dataset + optional expert knowledge* input maps directly onto Step 1,
its execution scoring is self-bias-proof, and it is SciPy-native. Step 1 ingests the task and
expert knowledge; Step 2 generates normal, ambiguous, and safety-boundary cases; Step 3 scores by
executing the program; Step 4 clusters failures such as "loads the wrong columns."

### Use Case 2 — Financial Document QA on FinanceBench (Contrast Demo)

The contrasting agent answers open-ended questions over real 10-K filings. We use **FinanceBench**
[@islam2023financebench] — open-book QA with 10,231 evidence-linked questions, on which a strong
retrieval-augmented model was wrong or refused on roughly four-fifths of a sampled set, so the
task is genuinely hard. The agent **must** ground each answer in the supplied documents, and the
safety dimension is intrinsic: a hallucinated figure or a wrong refusal on a regulated domain is
the failure that matters. The pipeline is identical — Step 1 extracts the filing taxonomy and
grounding constraints; Step 2 generates evidence-required and refusal-expected cases; Step 3 scores
against cited evidence (drawing on hallucination labels such as RAGTruth [@wu2024ragtruth]); Step 4
clusters failures such as "states an unsupported figure." One pipeline thus spans a structured,
execution-verified coder and an open-ended, grounding-verified QA agent without per-domain tooling.

## Results

(sec:results)=

The experimental variable throughout is the **domain context supplied to the Step-2
generator**. On each dataset the generator runs twice over the *same* source documents with an
*identical* prompt — once **with** the ingested `DomainContext` (and, on FinanceBench, the
domain-compliance runbook) and once **without** either — and we compare the two golden sets it
produces. The evaluation is of the **generated questions, not the agent's answers**: what a
framework can *test for* is decided at generation time, so that is what we measure. Every number
below is reproducible from the committed configuration and the two saved golden sets; generation
ran through the Claude Code workflow [@anthropic2025claude]. One metric uses an LLM-as-judge; we
corroborate it with two judge-free metrics so that no result rests on a model grading its own
output.

### FinanceBench: what the generated golden set can test for

Both arms generated **50** questions over the same FinanceBench 10-K filings. Given the domain
context, the generator wrote questions that name specific issuers, periods, and line items and
that probe the domain's compliance rules; given neither the data nor the runbook, it could only
write generic template disclosure questions. We quantify the gap three ways — a judge-scored
accuracy metric and two judge-free ones (lexical relatedness and unsupervised topic structure).

**Domain accuracy of the questions.** An LLM-as-judge holding the raw filing data marks a
generated question *domain-accurate* when it targets a real, specific financial fact verifiable
against that data — scored as the mean of a relevance and a groundedness axis, with
compliance/refusal probes excluded as non-factual. The with-context questions score **93%**
(n = 36) against **51%** (n = 50) without — a **41-point** gap (@fig:qa-grounded). The
without-context questions are valid finance questions, so they earn relevance credit, but they
are company-less and verify against no specific filing.

**A judge-free corroboration.** Because that metric uses a judge, we repeat the comparison with a
deterministic, offline one: each question's **domain relatedness** is its maximum TF-IDF cosine
similarity to the closest real FinanceBench record (over all question, answer, and evidence
text). With-context questions sit closer to real filings — median **0.263** versus **0.176**,
roughly a 49% higher median (@fig:qa-grounded). Both sets are genuinely finance questions, so
the distributions overlap; the shift is the vocabulary of real filings — named companies,
specific line items, periods — that only the with-context generator could draw on.

:::{figure} combo_grounded.png
:label: fig:qa-grounded
:width: 100%
The generated questions are more domain-grounded *with* vs *without* domain context, on both a
judge-scored metric and a judge-free one. **(a, left)** Domain accuracy — mean of an LLM-judged
relevance and groundedness axis (we test the questions, not the answers): 93% with context vs 51%
without. **(b, right)** Per-question domain relatedness (max TF-IDF cosine to the closest real
filing), a judge-free offline metric; the with-context distribution shifts up while overlapping
the baseline.
:::

**Topic structure and the compliance gap.** Running NMF topic modeling over the raw question
text of each set — a fully unsupervised, judge-free view of what the generator actually produced
— recovers two very different golden sets (@fig:qa-topics, @fig:qa-flow). With context, the 50
questions split into **36 grounded capability questions** across six finance topics, including
domain-aware probes the blind generator never produces (*segment & revenue mix*,
*metric-applicability / when-to-decline* — e.g. "is inventory turnover meaningful for a bank?"),
plus **14 agent-compliance probes** (28% of the budget) spanning the four behavioral safety
anchors: off-domain refusal, MNPI / confidential data, no-advice, and escalation (the fifth
anchor, evidence-grounding, is enforced on every capability question).

:::{figure} qa_topic_bubbles.png
:label: fig:qa-topics
:width: 72%
NMF topics over the question text, placed by mean domain relatedness (x) and mean specificity
(y); bubble area is the number of questions. With-context topics (blue) — including a cluster of
agent-compliance probes the baseline never generates — sit higher and to the right; without-context
topics (grey) collapse into a generic, less-grounded region.
:::

Without context, the same generator produces **only generic disclosure topics** — 25 of the 50
are template "summarize Item 1A / MD&A / the auditor's opinion" prompts — and the
**agent-compliance category is empty (0 probes)**, confirmed by reading every question. This is
the decisive result, and it is a *coverage* gap rather than a score swing: the generic golden set
cannot test a single compliance rule, so an agent can breach all of them and the evaluation never
knows. @fig:qa-flow traces both sets from root to NMF topic, and @tbl:fb-coverage summarizes what
each golden set can — and cannot — test for.

:::{figure} qa_topic_flow.png
:label: fig:qa-flow
:width: 52%
Each 50-question golden set traced root → sub-category → NMF topic. *With* domain context (top):
36 grounded capability questions across six finance topics plus 14 agent-compliance probes across
four safety topics. *Without* (bottom): 50 generic disclosure questions and an **empty
agent-compliance sub-category (0)** — the blind generator writes no safety probes at all.
:::

```{list-table} What each FinanceBench golden set can test for. The two baseline columns are the no-framework reality (not a measured run); the framework column is the measured with-context golden set, whose 50 questions partition into 36 domain-capability questions (5 of them edge-case/decline) and 14 compliance/safety probes.
:label: tbl:fb-coverage
:header-rows: 1
* - Capability
  - No golden set
  - Hand-authored
  - AI Eval Engine (domain info + runbook)
* - A golden set exists to test against
  - —
  - yes
  - yes — 50 questions
* - Domain-capability / metric questions (income statement, balance sheet, cash flow, ratios, segments)
  - no
  - partial — common ones only
  - 36 tests across 6 topic areas
* - ↳ of which edge-case / decline (metric not meaningful, missing line item)
  - no
  - partial
  - 5 tests (subset of the 36)
* - Compliance / safety probes (off-domain, MNPI, advice, escalation)
  - no
  - partial — only if remembered
  - 14 tests (off-domain 4, MNPI 4, advice 3, escalation 3)
* - Systematic coverage of all 5 compliance anchors
  - no
  - ad hoc — no guarantee
  - yes — all 5 probed by construction
* - Versioned & reproducible
  - —
  - usually not
  - yes — content-addressed
* - Auto-regenerates when data / prompts change
  - —
  - no — goes stale
  - yes
* - Per-team authoring effort
  - none
  - high — from scratch, every team
  - ~15-line YAML
```

### ScienceAgentBench: structural comparison and the published expert-knowledge gap

For the execution-scored coding agent we report the **structural** effect of domain context on the
generated evaluation, holding the correctness side to the benchmark's own published numbers.
Running Step 1 on the ScienceAgentBench tasks and generating the golden set with and without the
resulting `DomainContext` changes its structure: from **12** normal cases to **17** (12 normal + 5
safety-boundary), adding a safety dimension — a safety score and a safety category, across the four
disciplines — that the generic evaluation lacks entirely.

The five generated safety-boundary tests are domain failure modes the extractor surfaced from the
tasks themselves — compute the requested metric not a proxy (critical), use the specified
files/columns, write the exact output path, fail loudly on missing data, and preserve
domain-method semantics. They exist only because Step 1 ingested the domain.

For correctness we anchor to ScienceAgentBench's published result: the best agents solve ~32% of
tasks, rising to ~42% with **hand-authored expert knowledge** [@chen2024scienceagentbench] — itself
a with-versus-without-domain-knowledge comparison validated by the benchmark's authors. Our Step-1
`DomainContext` is an *automated* replacement for that expert knowledge, so we frame the
contribution against a baseline the benchmark already established. A full execution-scored run is
left to future work (see [](#sec:limitations)).

### Summary

Across both demonstrations the result is the same, and it is a *generation-time coverage* result
rather than an accuracy swing: giving the Step-2 generator the ingested domain context changes
what the golden set can test for. On FinanceBench it turns 50 generic, weakly-grounded disclosure
questions with **zero** compliance probes into 36 grounded, domain-specific questions plus 14
probes spanning all four behavioral safety anchors, with the questions measurably closer to real
filings on both a judge and a judge-free metric. On ScienceAgentBench it adds five execution-safety
tests and a safety dimension the generic evaluation structurally lacks. In both cases the added
coverage exists *only because Step 1 ran*; a generic golden set, scoring the very same agent, has
no way to surface it.

## Implementation Notes

The framework is implemented in Python (3.11+) and all five steps are exercised in
[](#sec:results). Only Step 1 needs a model call (Claude via the Anthropic API
[@anthropic2025claude], with prompt caching); Steps 2–5 run offline and deterministically, so the
golden set, scoring, runbook, and dashboard reproduce without an API key. Both demos use public
data with independent checks (execution for ScienceAgentBench, document grounding for FinanceBench),
and Use Case 2 retrieves with Chroma [@chroma]. Configuration is YAML — no code to onboard a domain
— and artifacts are plain-text and Git-friendly. The repository is MIT-licensed; a live
demonstration will accompany the SciPy 2026 talk.

## Discussion

**Reviewer, not author.** The framework keeps a human in the loop at every generation step — not
because the model cannot do the job, but because an artifact a domain expert *accepted* carries
different organizational weight; it optimizes for *acceptance latency*, not full autonomy.

**Discovered, not declared.** The Safety Runbook grows from observed failures rather than an
upfront constraint list, on the wager that teams do not know the full list until they have watched
the agent fail — and that learning feeds back into test generation.

**Verifiability as a design constraint.** Choosing datasets by *checkability* is what lets us claim
domain-aware evaluation without conceding to self-bias; the framework is most trustworthy where an
independent check exists, and judged scores should be read as weaker than verified ones.

(sec:limitations)=
## Limitations and Future Work

The current scope targets text- and code-producing agents reached over HTTP; multimodal and
long-horizon tool-using agents are future work, though Step 1's extractor is designed to
generalize. The Safety Runbook is a single Markdown file that will need per-sub-domain splitting at
scale, the Step 5 dashboard is read-only (trend visualization is planned), and an
execution-verifiable text-to-SQL slice (e.g., a small BIRD subset) is a natural third demonstration
left for future work.

## Conclusion

AI Eval Engine treats evaluation as a *pipeline to be generated*, not an artifact to be authored:
from a pluggable domain context it generates the golden set and eval script, scores with
verifiable checks wherever possible, and accumulates a living Safety Runbook — a reusable,
rerunnable, domain-aware path to tracking agent behavior as a first-class OKR. Demonstrations on an
execution-scored coder and a grounding-scored QA agent show one pipeline across two very different
domains, with self-bias addressed by construction.

## Availability

Source code is released under the MIT license. The SciPy 2026 talk page and the public
repository are linked from the proceedings entry.

## Generative AI Disclosure

In accordance with the SciPy generative AI policy, the author discloses that generative AI
(Anthropic Claude, via the Claude API and Claude Code) was used in two capacities: as the
*subject* of this work — the framework invokes Claude for context extraction, golden-set and
eval-script generation, and LLM-as-judge scoring — and as a *writing aid* for drafting and
refining prose and code examples. All outputs were reviewed, verified, and revised by the author,
who takes full responsibility for the final content.

## Acknowledgments

This work is an independent open-source contribution to the SciPy 2026 Proceedings, separate from
the author's employer affiliation.
