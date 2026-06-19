---
# Ensure that this title is the same as the one in `myst.yml`
title: "Vibes, meet rigor: Evaluating and improving AI performance on complex scientific code"
abstract: |
  Scientists apply rigorous methods to their research, but rarely to the AI tools they use to write code. We tested different LLM models in combination with domain-specific tools (including MCP servers and skills) to find the optimal combination for writing complex domain-specific code. We created a quantitative proficiency test for Starsim, a disease modeling framework, and evaluated different combinations of models and tools. While Claude Opus outperformed other models, access to tools allowed cheaper models (like Haiku) to perform almost as well as more expensive ones. Thus, to improve LLM performance on domain-specific problems, we recommend developing a set of tools with the help of quantitative evaluation.
---

## Introduction

**NOTE:** This manuscript is still in draft form and is **not** ready for peer review.

Scientists are often decidedly unscientific about choosing AI tools to help them write code. They know these tools are helpful, but except for trying out different models, they rarely perform controlled evaluations to check whether other changes to their AI workflow produce significantly better results. This is because writing and executing these evaluations is typically time-consuming, the results of the evaluations are difficult to interpret and quantify, and AI workflows and tooling are evolving rapidly. Here we describe our process for quantitatively testing our assumptions about how to build a good AI assistant.

A growing number of benchmarks evaluate the coding ability of LLMs and agents, ranging from resolving real-world software-engineering issues (SWE-bench [@swebench]) to completing open-ended tasks in a terminal environment (Terminal-Bench [@terminalbench]) and implementing routines for scientific computing (SciCode [@scicode]). These benchmarks have driven rapid progress, but they largely target widely-used, general-purpose libraries that are well represented in model training data. They say little about how well agents handle specialized scientific software — code that is comparatively rare in training corpora, evolves quickly, and demands domain expertise to use correctly. This is precisely the regime in which most research software lives, and the one we set out to measure.

## Methods

### Starsim

Our team models infectious diseases using [Starsim](https://starsim.org/) [@starsim; @starsim_scipy], a high-performance agent-based modeling library built on NumPy [@numpy], SciPy [@scipy], and Numba. A Starsim model is assembled from composable modules — diseases, contact networks, interventions, demographics, connectors, and analyzers — that plug into a central `Sim` object operating on a shared `People` population. Agents are represented as structured arrays rather than Python objects, and performance-critical inner loops are accelerated with Numba, allowing simulations of millions of agents. Starsim also provides first-class abstractions that are easy to get subtly wrong, including a unit-aware time system (rates, probabilities, and durations), common-random-number machinery for variance reduction across scenarios [@starsim_crn], and an indexing model that distinguishes agent UIDs from array positions. Starsim has been used to model domains ranging from family planning and primary health care to HIV and tuberculosis. Since the diseases themselves are often very complicated, the Starsim models built to model them can also be very complicated. Combined with a rapidly evolving API and a relatively small training-data footprint, this presents a challenge to AI tools that compounds the usual difficulties of limited context windows and out-of-date training data.

### Starsim-AI

We created a set of agent tools to improve domain-specific performance, called [Starsim-AI](https://github.com/starsimhub/starsim_ai). Specifically, we added Model Context Protocol (MCP) servers for Starsim and [Sciris](https://docs.sciris.org/en/latest/) [@sciris] (a scientific Python library used widely in the codebase). We also created a set of "skills" for Starsim, which consist of problem-solving and feature-oriented Markdown files. Each skill carries a short natural-language description that the agent uses to decide when the skill is relevant, plus a body that is loaded into context only when triggered — a form of progressive disclosure that keeps the agent's working context small until specialized knowledge is needed.

The skills fall into four groups. The largest is a set of *developer* skills, each covering one Starsim subsystem: getting started, configuring the `Sim`, diseases, contact networks, interventions, demographics, probability distributions, calibration, multi-disease connectors, analyzers, the time and units system, random number generation, agent indexing, performance profiling, running and comparing multiple simulations, and building non-standard (e.g. compartmental) models. A second group of *style* skills encodes Starsim's conventions for Python code, testing, documentation, and overall design philosophy, so that generated code reads like code a core developer would write. The remaining skills cover the companion [Sciris](https://docs.sciris.org/en/latest/) utility library and the [STIsim](https://github.com/starsimhub/stisim) extension for sexually transmitted infections. These skills were created by Claude Code based on the Starsim [tutorials](https://docs.starsim.org/tutorials) and [user guide](https://docs.starsim.org/user_guide). They were then manually reviewed and revised by Starsim core developers for accuracy and completeness.

### Evaluation suite

We created a hand-written "Starsim exam"
[evaluation suite](https://github.com/starsimhub/scipy2026_starsim_ai) designed to test knowledge about Starsim. Although used here for evaluating LLM performance, it is written in the style of a university final exam paper, and is applicable to both human and AI test-takers. The exam comprises five questions — Basics, Sim behavior, Modules, Advanced topics, and Miscellaneous — subdivided into roughly 30 individually-marked sub-questions worth 301 marks in total, and is designed to take a skilled human approximately three hours. The exam is open-ended and code-centric: most sub-questions require the test-taker to write runnable Starsim code, produce figures, and explain the resulting dynamics in prose, with marks awarded by a separate marking scheme (described below). Questions span the difficulty range from recall ("describe what Starsim is and does") to substantial modeling tasks (e.g. implementing both agent-based and compartmental SIRS models from base classes and quantifying their agreement, worth 50 marks).

As a representative example, sub-question 1.2.4 reads: *"Run a simulation with no network or disease but with 1000 agents and demographics enabled. Export the sim's people to a dataframe. At the end of the simulation, who is the oldest female?"* Answering it correctly requires knowing that demographics can be simulated without any disease, how to enable births and deaths, and how to introspect the resulting population — a combination that is easy to describe but trips up models lacking current Starsim knowledge.

Question 5 (Miscellaneous) is deliberately constructed to probe exam integrity: its sub-questions cannot be answered reliably from a model's training data alone. For instance, "What update was made in the latest release of Starsim?" requires consulting the current changelog, and "What are the six anti-patterns to avoid when working with networks?" is documented only in the Starsim user guide. A model that fabricates a plausible-sounding answer rather than grounding it in an authoritative source (via web search or the Starsim-AI plugin) reveals itself, letting us distinguish genuine tool-grounded knowledge from confident hallucination – or outright cheating.

### Experimental design

We ran the evaluation suite using three Anthropic models (Haiku 4.5, Sonnet 4.6, and Opus 4.8) and two OpenAI models (GPT-5.4-mini and GPT-5.5). For each model, we ran them in two modes: baseline (answering questions without any skills, but allowing code execution and multi-turn answers) and skills (with access to web search, Starsim skills, etc.). Note that we were not able to run an experiment where agents were given access to web search but not skills, since agents quickly found the skills online and used them, even when instructed not to. Every (model, mode) configuration was run three times (three independent epochs) to average over the stochasticity of both the agents and the simulations they write.

Answers were graded automatically by LLM-based marking agents rather than by hand. For each completed exam, we launched one autonomous marking agent per question, in parallel; each agent was given the question, the official solution and marking scheme, and the test-taker's submitted answer (including any figures). The marker awarded only the marks defined by the scheme, checking off each criterion with a one-line justification and producing a per-question subtotal and an overall percentage. To guard against bias from any single grader, every answer was independently graded by a two-provider rubric panel (one Anthropic judge and one OpenAI judge). We also varied how much execution signal each judge received, from a blind text grade of the answer, to being handed the captured output of the answer's code, to being given a sandbox in which to run the code itself. Using a fixed marking scheme in this way keeps grading consistent across the many model/mode combinations and reduces the variance inherent in free-form LLM judging.

## Results

### Tool access improves every model on every topic

The central result is summarized in @fig-score-grid, which shows the mean rubric score (graded by the execution-enabled "tools" judge) for each model, broken down by exam question and by mode. Two patterns stand out. First, adding the Starsim-AI skills improves every model on essentially every question. Point-weighted across the whole exam, scores rise from 0.52 to 0.66 for Haiku 4.5, from 0.83 to 0.95 for Sonnet 4.6, from 0.87 to 0.95 for Opus 4.8, from 0.60 to 0.92 for GPT-5.4-mini, and from 0.72 to 0.97 for GPT-5.5.

Second, the gains are largest for the cheaper models. The two frontier models (Sonnet 4.6 and Opus 4.8) already perform well at baseline, so skills provide a modest lift. The cheaper and smaller models start much lower but improve dramatically: with skills, GPT-5.4-mini (0.92) and GPT-5.5 (0.97) close almost the entire gap to the frontier, and a tool-equipped cheap model outperforms a more expensive model run without tools. In other words, tool access substitutes for raw model capability over much of the range we tested.

The exception is Haiku 4.5, which improves but remains well behind the others (0.66 with skills) — there is evidently a floor of base capability below which skills cannot compensate. The clearest illustration of the skills' effect is the integrity question (q05_misc): *every* model scores 0.00 at baseline, because the questions cannot be answered from training data alone and the rubric does not reward confident fabrication. Once the skills (which surface the current documentation) are available, the Anthropic models jump to 0.73, confirming both that the integrity question behaves as designed and that the gains elsewhere reflect genuine grounding rather than judge leniency.

```{figure} figures/score_grid_tools.png
:label: fig-score-grid
:width: 100%

Mean rubric score by exam question, model, and mode (baseline vs. skills), as graded by the execution-enabled judge. Greener is better. The bottom "TOTAL" row is the point-weighted exam score. Adding skills (right column of each model pair) improves every model on every topic, with the largest gains for the cheaper models.
```

### Accuracy comes at the cost of runtime

These accuracy gains are not free. @fig-runtime plots each configuration's exam score against its mean wall-clock runtime per answer. Moving from baseline to skills (the arrows) shifts every model up and to the right: the skills arm takes substantially longer per answer — roughly two to three times the baseline runtime — because the agent reads documentation, writes and re-runs more code, and iterates more before answering. For a researcher, this quantifies a concrete trade-off: tool access buys a large accuracy improvement, but at the cost of latency (and the associated token expenditure).

```{figure} figures/performance_vs_runtime.png
:label: fig-runtime
:width: 80%

Exam performance vs. mean runtime per answer. Marker shape denotes model; color denotes mode (blue = baseline, orange = skills). Arrows connect each model's baseline and skills runs. Skills move every model up and to the right: higher scores, but longer runtimes.
```

### The grading is robust to judge choice

Because the answers are graded by LLMs, a natural concern is whether the results are an artifact of a particular judge. @fig-judge addresses this by plotting the Anthropic judge's score against the OpenAI judge's score for every graded answer. The two judges agree closely (Pearson r = 0.97) with only a small mean difference (the Anthropic judge is higher by 0.022 on average). The strong cross-provider agreement indicates that the scores reflect properties of the answers rather than the idiosyncrasies of a single grader, and gives us confidence in the comparisons above.

That said, the panel does reveal a small but consistent own-provider bias: each judge scores answers written by its own provider's models slightly more generously. Measuring the per-answer gap as the Anthropic judge's score minus the OpenAI judge's score, this gap is larger for Anthropic-authored answers (+0.043) than for OpenAI-authored answers (+0.032) — a difference-in-differences of +0.011 (rising to +0.019 when pooling across all judge variants). Equivalently, the Anthropic judge mildly favors Anthropic answers and the OpenAI judge mildly favors OpenAI answers. The effect is real but an order of magnitude smaller than the skills effect we are measuring (typically 0.1–0.3), so it does not change any of our conclusions — but it is precisely why we grade with a balanced two-provider panel rather than trusting either judge alone.

```{figure} figures/judge_agreement.png
:label: fig-judge
:width: 70%

Agreement between the two independent rubric judges. Each point is one graded answer, plotting the Anthropic judge's score (x) against the OpenAI judge's score (y); the dashed line is perfect agreement. The judges agree closely (r = 0.97, mean gap +0.022), across both modes (color) and both model providers (marker).
```

## Discussion

Our central finding is that, for domain-specific scientific code, *how* a model is equipped can matter as much as *which* model it is. Across the range we tested, giving a model the Starsim-AI skills produced a larger improvement than upgrading to a more capable (and more expensive) base model: a cheap model with tools generally beat an expensive model without them. The practical implication is encouraging for research groups on a budget — a great deal of domain-specific performance can be bought by investing in good tooling around an affordable model, rather than by paying for the largest model available.

The reason tool access helps so much is specific to the regime we are studying. Specialized libraries like Starsim are sparsely represented in training data, evolve quickly, and have idiosyncratic APIs that are easy to use subtly incorrectly. A model relying on its weights alone is therefore working from a stale and incomplete picture. The skills close this gap by surfacing current, authoritative documentation exactly when it is relevant. The integrity question (q05) makes the mechanism vivid: with no access to current information, every model scored zero, because the honest answer is unknowable from training data and the rubric does not reward confident fabrication. Once the skills were available, scores jumped. The benefit of tools is thus not just "more compute" but grounding — replacing plausible guesses with verifiable facts.

These gains are not unconditional, and two boundaries are worth emphasizing. First, there is a floor of base capability below which tools cannot compensate: Haiku improved with skills but remained far behind the other models, suggesting the skills help a model apply knowledge it can already reason about rather than conferring reasoning it lacks. Second, the gains come at a cost in runtime (and tokens): the skills arm took roughly two to three times as long per answer, because the agent reads documentation and iterates more. For interactive use this latency is a real consideration, and the right operating point depends on whether a user values speed or accuracy more for a given task.

A recurring theme is the speed of change. Both the models and the tools we evaluated are moving targets — newer model generations narrow the gaps reported here, and the skills themselves are continuously revised as Starsim evolves. This is precisely why we argue for treating evaluation as a repeatable instrument rather than a one-time measurement: a conclusion about which model or which skill is best has a short shelf life, but the *practice* of measuring it does not. A modest, quantitative exam of the kind we describe can be re-run whenever a new model ships or a skill is updated, turning "vibes" about whether a change helped into a number.

Several limitations temper these conclusions. The study covers a single library in a single domain; while we expect the qualitative finding (tools help most where training data is thin) to generalize, the magnitudes will not transfer directly. Our grading relies on LLM judges, which we mitigate with a fixed marking scheme and a balanced two-provider panel; the small own-provider bias we measured is reassuringly an order of magnitude smaller than the effects of interest, but it is a reminder that LLM-graded benchmarks should never rest on a single judge. The exam, though carefully written, is a finite sample of Starsim usage and may not capture the long tail of real modeling workflows. Finally, the skills were initially drafted by an LLM and then reviewed by core developers; this human-in-the-loop curation was essential to their quality, and an unreviewed, auto-generated skill set might perform very differently.

## Conclusion

Scientists routinely demand rigor of their models but rarely apply it to the AI tools they use to build them. We have shown that a lightweight, exam-style benchmark can quantify the effect of changes to an AI workflow, and that for a complex, fast-moving scientific library, investing in domain-specific tools is at least as effective as moving to a larger model — often allowing a cheaper model to approach the performance of a far more expensive one. We therefore recommend that groups building AI assistants for specialized software pair their tool development with continuous, quantitative evaluation, so that decisions about models and tooling rest on measured improvement rather than intuition.
