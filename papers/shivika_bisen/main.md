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
  from a lightweight configuration, (2) builds a living **Domain Compliance Runbook** of domain
  facts and compliance criteria, (3) generates a versioned, domain-grounded golden set from those
  criteria, (4) emits a runnable evaluation script that scores correctness, evidence grounding,
  and output format — feeding failures back into the runbook, and (5) tracks the result as
  quarter-over-quarter objectives and key results. A central methodological concern for any pipeline that uses a
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
Teams shipping agents over financial filings, scientific datasets, or support transcripts hit
the same wall: existing tooling measures generic {abbr}`LLM (large language model)` properties —
hallucination rate, latency, token cost — but not the question that blocks a launch: *did the
agent do the right thing in this domain?*

Domain correctness is not a generic property. A data agent over a financial schema and a
{abbr}`RAG (retrieval-augmented generation)` agent over scientific documents require different
notions of "correct," and those notions exist only in the **domain context**: a scientific agent
must load the right columns and compute the metric the paper reported; a financial agent must
ground every figure in its filing and decline when the filing does not support an answer. Neither
constraint is visible to a generic judge scoring token-level fluency.

The state of practice is the **manually labeled golden set**, with three well-known failure
modes: it goes *stale* when data or prompts change; it is *non-reproducible* across teams; and it
gives *no clarity on what to fix*, reporting only that something broke. These follow from
treating evaluation as an artifact to be *authored* rather than a pipeline to be *generated*.

This paper presents **AI Eval Engine**, an open-source Python framework that reframes evaluation
as a domain-driven generator. The team points at where the domain lives; the framework produces a
versioned golden set, a runnable evaluation script, and a living Domain Compliance Runbook that captures
emergent failure modes — shifting the human role from *author* to *reviewer* and letting teams
track agent behavior as a first-class quarter-over-quarter {abbr}`OKR (objective and key result)`.
It leans on a language model — Anthropic's Claude [@anthropic2025claude], via the Claude
{abbr}`API (application programming interface)` and Claude Code — for context extraction, golden-set
and eval-script generation, and the {abbr}`LLM (large language model)`-as-judge step.
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
Domain Compliance Runbook** of domain constraints from observed failures, closing a loop back into
generation. It is meant to be used *alongside* these tools — RAGAS or DeepEval metrics register as
extra Step-4 scorers — and self-improving methods such as Agentic Context Engineering
[@zhang2026ace] are candidates for evaluation *by* it, not components of it.

(methodology)=
## Methodology: Domain-Aware Evaluation Without Self-Bias

A pipeline that uses an LLM to *generate* the golden set and the same model family to *judge* the
agent invites an obvious objection: the evaluation may simply reward outputs that look like what
the generator would produce. This **self-bias** is the central risk of LLM-as-benchmark-generator
plus LLM-as-judge pipelines [@silencer2025], and the first thing a reviewer will probe.

We adopt a single guiding principle that defuses self-bias, benchmark saturation, and training
contamination at once: **choose tasks where the model fails without the ingested domain
context, and where correctness can be checked independently of the judge.** Two consequences
follow.

- **Prefer verifiable correctness over judge opinion.** Where output can be *executed* (does
  the program produce the expected artifact?) or *grounded* (is the answer supported by cited
  evidence? — a claim of "capital expenditure was \$1,577 million" either matches the cited
  cash-flow statement or it does not), correctness is decided by the world, not the model.
- **Prefer domains the model could not have memorized.** Generic code generation (HumanEval-style)
  is saturated and largely in pretraining, so a high score measures recall, not domain-aware
  evaluation; the contribution is meaningful only on tasks needing context — a specific dataset or
  filing — the model never saw.

This principle makes the datasets in [](#use-cases) load-bearing: each was chosen because the
agent cannot succeed without the ingested domain context, and each admits an execution- or
grounding-based check independent of the judge. One scoping note: this principle governs Step 4 —
scoring an agent's *answers*. [](#sec:results) does not run Step 4; it evaluates Step 3 — the
*questions* the generator produces — and its headline metric is itself judge-scored, so the
protection there is corroboration: every judged number is paired with judge-free measurements.

## The Five-Step Framework

The framework decomposes evaluation into five steps (@fig:pipeline), each an independent Python
module with a stable interface so teams can replace, extend, or skip a step without forking the
pipeline.

:::{figure} pipeline.png
:label: fig:pipeline
:width: 100%
The five-step framework. A pluggable domain context (Step 1) builds the living **Domain Compliance
Runbook** (Step 2), whose criteria drive golden-set generation (Step 3); eval-script generation and
scoring (Step 4) feed failures back into the runbook, and an actionable dashboard (Step 5) tracks
the OKRs over time.
:::

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
the extraction sees the domain's breadth, not only its most common cases.

### Step 2 — Domain Compliance Runbook (Living)

From the `DomainContext`, the framework assembles a single content-addressed
`domain_compliance_runbook.json` — the **Domain Compliance Runbook**. The name borrows from
operations, where a *runbook* is the reference document an on-call engineer follows during an
incident; here it is the domain reference the golden-set generator consults on every run. It has
three sections:
**domain facts** (definitions, conventions, units), **compliance criteria** (five *fixed,
universal* agentic-compliance anchors — domain scope, evidence grounding,
privacy/confidentiality, no-advice, human escalation — instantiated per domain with severity:
finance's {abbr}`MNPI (material non-public information)` rule — MNPI being undisclosed
information that could move a company's stock price, which securities law restricts trading on or
selectively revealing — is the privacy anchor's finance instance), and **common failure modes**. The first two are seeded now,
at ingestion; the third is *living* — it accumulates as the eval (Step 4) surfaces failures, which
the framework clusters deterministically by `(category, failure_type)` and turns into a
`recommended_check` (the golden-set addition that would catch each one). The anchor taxonomy is therefore *declared* — fixed by design, so all
five rules are probed in every domain — while its instantiations and failure modes are
*discovered*: a cluster surfaced in run $N$ becomes a runbook entry the golden-set
generator (Step 3) picks up as a priority for run $N{+}1$, closing the loop between observation and
test generation. A concern fitting none of the five is the signal to promote a new universal
anchor, not to bolt on a one-off rule. The clustering and `recommended_check` mapping are **deterministic rules, not a
learned model** — "living" means an accumulated, human-reviewable record, not gradient training.

### Step 3 — Automated Golden Set Generation

From the Domain Compliance Runbook's criteria, the framework constructs a versioned,
domain-grounded set of `GoldenCase`s, each carrying the input, the expected outcome (an executable
check, an artifact, or an evidence-grounded answer), and the compliance criterion or capability it
probes. Two paths exist: **normalize** (`build_golden_set`, fully offline) adopts a public
benchmark's own verified labels; **generative** (`generate_golden_set`) has Claude author fresh
cases grounded in real evidence — happy-path lookups, hard multi-step computations, definitionally
ambiguous queries, and out-of-policy requests the agent must decline (e.g. asking for free cash
flow from a balance sheet alone). In both, **compliance probes** are synthesized from the runbook's
five anchors, so every domain rule is always probed. Generating this *compliance coverage* from
extracted constraints, rather than plain corpus-to-question pairs, is the contribution; recurring
failures recorded in the runbook (Step 2) feed back here as priorities for the next set.

Both modes enforce one rule: every figure in an expected answer must be traceable to cited
evidence, or the case becomes an explicit refusal — the self-bias guard from [](#methodology),
letting a reviewer (or, where a public split exists, the benchmark's own gold answer) verify each
case without trusting the generator. The FinanceBench result in [](#sec:results) uses
**generative** mode — the framework authors its own golden set, which is exactly what we evaluate
there. Golden sets are content-addressed (`goldensets/<version>/`) so reruns can be diffed, and a
human reviewer may accept, edit, or reject any case before it is promoted — the human role is
**reviewer**, not author.

### Step 4 — Eval Script Generation and Scoring

The framework emits a runnable Python evaluation script producing three orthogonal scores per
case: **correctness** (execution of the produced program for executable cases, or a normalized /
numeric match for grounded ones), **grounding** (the fraction of the answer supported by the cited
evidence, by deterministic token/numeric overlap, so a fluent but unsupported figure is
penalized), and **format validation** (structural integrity, checked without a model call). All
three scorers in [](#sec:results) are **deterministic** — no model grades another model's output.
Scorers are pluggable: the shipped `grounded_qa` and `code_execution` cover the two demos, and
metrics from tools such as RAGAS or DeepEval can register as additional scorers. The
{abbr}`LLM (large language model)`-as-judge is the designed fallback for the genuinely open-ended
residue but was not exercised here: the judge is never the sole arbiter on a task that execution or
grounding can settle. Failures from this step feed back into the Domain Compliance Runbook (Step 2).

### Step 5 — Actionable Dashboard UI

Once the pipeline has run, the framework becomes a live OKR tracker: as data changes, golden sets
and the Runbook regenerate on rerun, enabling quarter-over-quarter tracking of an **Agent
Compliance Score** (pass rate on compliance-critical cases), a **Domain Accuracy Score** (correctness on
in-domain queries), a **Drift Indicator** (share of new failure patterns since the last run), and
a **Coverage Score** (share of the discovered domain context represented in the golden set).
These four metrics are computed from Step-4 runs and are defined here but not measured in this
paper.

(use-cases)=
## Demonstration Use Cases

We demonstrate the framework on two contrasting, domain-dependent agent shapes that bracket the
spectrum it targets: a **structured, execution-scored** scientific coder (ScienceAgentBench) and
an **open-ended, grounding-scored** document-QA agent (FinanceBench) — Python program versus
free-text-with-evidence, both failing without the ingested domain context, both chosen under the
principle of [](#methodology).

### Use Case 1 — Scientific-Coding Agent on ScienceAgentBench

Given a scientific task and a dataset, the agent produces a self-contained Python program. We use
**ScienceAgentBench** [@chen2024scienceagentbench] — 102 tasks from 44 peer-reviewed papers across
four disciplines, each **scored by execution** (the best agents solve roughly a third, so it is
far from saturated). Its *dataset + optional expert knowledge* input maps directly onto Step 1,
its execution scoring is self-bias-proof, and it is SciPy-native. Step 1 ingests the task and
expert knowledge; Step 4 scores by executing the program, and failures (e.g. "loads the wrong
columns") cluster back into the runbook.

### Use Case 2 — Financial Document QA on FinanceBench

The contrasting agent answers open-ended questions over real 10-K filings. We use **FinanceBench**
[@islam2023financebench] — open-book QA with 10,231 evidence-linked questions, on which a strong
retrieval-augmented model was wrong or refused on roughly four-fifths of a sampled set, so the
task is genuinely hard. The agent **must** ground each answer in the supplied documents, and the
safety dimension is intrinsic: a hallucinated figure or a wrong refusal on a regulated domain is
the failure that matters. The pipeline is identical — Step 1 extracts the filing taxonomy and
grounding constraints; Step 3 generates evidence-required and refusal-expected cases; Step 4
scores against cited evidence (cf. RAGTruth [@wu2024ragtruth]). One pipeline thus spans a
structured, execution-verified coder and an open-ended, grounding-verified QA agent.

(sec:results)=
## Results

The experimental variable is the **domain context supplied to the golden-set generator
(Step 3)**, and the A/B comparison is run on FinanceBench (Use Case 2): the generator runs
twice — once **with** the ingested domain data, `DomainContext`, and the domain-compliance
runbook, and once **blind**, given neither the data nor the runbook — and we compare the two
golden sets it produces. The evaluation is of the **generated questions, not the agent's answers**: what a
framework can *test for* is decided at generation time, so that is what we measure. Every number
below is reproducible from the committed configuration and the two saved golden sets. Generation
and the LLM-as-judge both ran on Claude Opus 4.8 (`claude-opus-4-8`) via Claude Code
[@anthropic2025claude], June 2026 (flat-rate subscription, default settings, no fixed seed;
[](#sec:limitations)). Because generator and judge are the same model, the judged metric is
corroborated with two judge-free ones, so no conclusion rests on a model grading its own output.

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
deterministic, offline one: each question's **domain relatedness** is its maximum
{abbr}`TF-IDF (term frequency–inverse document frequency)` cosine similarity to the closest real
FinanceBench record (over all question, answer, and evidence text; scikit-learn
`TfidfVectorizer` [@pedregosa2011sklearn] with English stop words, unigrams, and sublinear TF). With-context questions sit closer to real filings — median **0.263** versus **0.176**,
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

**Topic structure and the compliance gap.** Running {abbr}`NMF (non-negative matrix
factorization)` topic modeling [@pedregosa2011sklearn] over the raw question text of each set
(TF-IDF features, 1–2-grams, finance-boilerplate stop words; topic assignments hand-verified by
reading every question) — a judge-free view of what the generator actually produced — recovers two very different golden sets (@fig:qa-topics, @fig:qa-flow). With context, the 50
questions split into **36 grounded capability questions** across six finance topics, including
domain-aware probes the blind generator never produces (*segment & revenue mix*,
*metric-applicability / when-to-decline* — e.g. "is inventory turnover meaningful for a bank?"),
plus **14 agent-compliance probes** (28% of the budget) spanning the four behavioral safety
anchors: off-domain refusal, MNPI / confidential data, no-advice, and escalation (the fifth
anchor, evidence-grounding, is enforced on every capability question).

:::{figure} qa_topic_bubbles.png
:label: fig:qa-topics
:width: 62%
NMF topics over the question text. x = mean domain relatedness (the TF-IDF metric of
@fig:qa-grounded); y = mean **domain specificity** — the fraction of five concreteness markers
pinned down (named company, fiscal period, statement, line item, figure verifiable in a real
filing). Bubble area = questions. With-context *capability* topics (blue, 1–6) sit higher and to
the right; the with-context *compliance* probes (7–10) sit low and left by design — naming an
entity but using little filing vocabulary — and have no baseline counterpart; without-context
topics (grey) collapse into a generic, less-grounded region.
:::

Without context, the same generator produces **only generic disclosure topics** — 25 of the 50
are template "summarize Item 1A / {abbr}`MD&A (Management's Discussion and Analysis)` / the
auditor's opinion" prompts. (Item 1A is the 10-K's risk-factor section and the MD&A the section
where management narrates results and risks — boilerplate every filing contains, so such
questions verify against no *specific* filing.) The
**agent-compliance category is empty (0 probes)**, confirmed by reading every question. To be
precise about why: the shared generation prompt asks for varied normal, ambiguous, and
out-of-scope cases but does not itself request compliance probes — those are synthesized from the
runbook's anchors ([](#sec:availability) links both arms' exact prompts). The blind arm has no
anchors to draw on, so 0/50 is *coverage by construction*, an architectural property of running
the pipeline without extraction rather than a discovered behavior of the generator. The design is
symmetric — both arms run under the identical instruction, so the baseline is not selectively
restricted — and the headline domain-accuracy result rests only on the questions both arms *did*
generate, so it does not depend on this design choice. That is
still the operative point, and it is a *coverage* gap rather than a score swing: an evaluation
built without the extracted anchors cannot test a single compliance rule, so an agent can breach
all of them and the evaluation never knows. @fig:qa-flow traces both sets from root to NMF topic, and @tbl:fb-coverage summarizes what
each golden set can — and cannot — test for.

:::{figure} qa_topic_flow.png
:label: fig:qa-flow
:width: 45%
Each 50-question golden set traced from root to sub-category to NMF topic. *With* domain context (top):
36 grounded capability questions across six finance topics plus 14 agent-compliance probes across
four compliance topics. *Without* (bottom): 50 generic disclosure questions and an **empty
agent-compliance sub-category (0)** — the blind generator writes no compliance probes at all.
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
* - of which: edge-case / decline (metric not meaningful, missing line item)
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

### Summary

The result is a *generation-time coverage* finding rather than an accuracy swing: giving the
golden-set generator the ingested domain context — through the Domain Compliance Runbook — changes
what the golden set can test for. On FinanceBench it turns 50 generic, weakly-grounded disclosure
questions with **zero** compliance probes into 36 grounded, domain-specific questions plus 14 probes
spanning all four behavioral compliance anchors, measurably closer to real filings on both
metrics. The added coverage exists *only because Step 1 ran*; a generic golden set, scoring the
very same agent, has no way to surface it.

## Implementation Notes

The framework is implemented in Python (3.11+); [](#sec:results) exercises Steps 1–3, while the
Step-4 scorers and Step-5 metrics are implemented and unit-tested but not evaluated there.
Only Step 1 needs a model call; Steps 2–5 run offline and deterministically, so the artifacts
reproduce without an API key. Both demos use public data with independent checks, and Use Case 2 retrieves with Chroma
[@chroma]. Configuration is YAML; artifacts are plain-text and Git-friendly. The repository is
MIT-licensed; a live demonstration will accompany the SciPy 2026 talk.

## Discussion

**Reviewer, not author.** A human stays in the loop at every generation step — an *accepted*
artifact carries different organizational weight — so the framework optimizes for *acceptance
latency*, not full autonomy. **Fixed anchors, discovered instances.** The five compliance
anchors are a fixed taxonomy — guaranteeing coverage by construction — while their domain
instantiations and the runbook's failure modes grow from observed behavior — teams learn each
constraint's *content* by watching the agent fail. **Verifiability as a design constraint.** Choosing
datasets by *checkability* underwrites the self-bias defense; judged scores read weaker than
verified ones.

(sec:limitations)=
## Limitations and Future Work

The current scope targets text- and code-producing agents; multimodal, long-horizon tool-using
agents are future work, though Step 1's extractor is designed to generalize. The Domain
Compliance Runbook is a single JSON artifact (`domain_compliance_runbook.json`) that will need
splitting at scale, the Step 5 dashboard is read-only, and an execution-verifiable text-to-SQL
slice (e.g., a small BIRD subset) is a natural third demonstration. Two experimental limits: the blind arm lacks both the
raw data and the extracted context, so the measured gap bounds their *combined* contribution (an
intermediate raw-data-only arm isolating Step 1 is planned), and each arm is a single unseeded
generation run, so the numbers carry no variance estimate.

## Conclusion

AI Eval Engine treats evaluation as a *pipeline to be generated*, not an artifact to be authored:
from a pluggable domain context it generates the golden set and eval script, scores with
verifiable checks where possible, and accumulates a living Domain Compliance Runbook — a
reusable, domain-aware path to tracking agent behavior as a first-class OKR.

(sec:availability)=
## Availability

Source code is MIT-licensed at <https://github.com/sbisen/ai-eval-engine>; the golden sets, judge
outputs, analysis scripts, and both arms' generation prompts behind [](#sec:results) are under
`results/ab_experiment/` there (every model-facing prompt is also printable offline via
`ai-eval-engine generate --show-prompt`). The public API mirrors the paper's steps:

```python
from ai_eval_engine import extract_domain_context, generate_golden_set
ctx = extract_domain_context("configs/financebench.yaml")  # Step 1 -> DomainContext
gs = generate_golden_set("configs/financebench.yaml", "out/context.json",
    target_cases=50, runbook_path="out/domain_compliance_runbook.json")  # Step 3
```

Every artifact is plain JSON. Two verbatim (abridged) excerpts from the committed FinanceBench
run show what the pipeline actually produces — a seeded *domain fact* from Step 2's
`domain_compliance_runbook.json`:

```json
{"group": "Definitions", "label": "Free cash flow",
 "detail": "Operating cash flow minus capital expenditures (purchases of property,
            plant & equipment). Both terms must come from the cited cash-flow statement."}
```

and one Step-3 `GoldenCase` from `golden_set_with_domain_context.json`:

```json
{"id": "wdc-001",
 "input": "Using 3M's FY2018 consolidated statement of cash flows, what was 3M's
           capital expenditure (purchases of property, plant and equipment) in USD millions?",
 "expected": "$1,577 million. Source: 3M_2018_10K, Consolidated Statement of Cash Flows,
              'Purchases of property, plant and equipment (PP&E)'.",
 "question_type": "metrics-generated",
 "probes_criteria": ["Citation & non-misleading disclosure"]}
```

## Disclosures

**Generative AI.** Per the SciPy generative AI policy: generative AI (Anthropic Claude, via the
Claude API and Claude Code) was used both as the *subject* of this work — the framework invokes
Claude for extraction, generation, and judge scoring — and as a *writing aid*. All outputs were
reviewed, verified, and revised by the author, who takes full responsibility for the final
content. **Affiliation.** This work is an independent open-source contribution, separate from
the author's employer affiliation.
