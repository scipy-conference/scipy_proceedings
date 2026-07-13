---
title: "Vibes, meet rigor: Evaluating and improving AI performance on complex scientific code"
abstract: |
  Scientists apply rigorous methods to their research, but rarely to the AI tools they use to write code. We tested different large language models (LLMs) in combination with domain-specific tools (including MCP servers and custom-written skills) to find the optimal combination for writing complex domain-specific code. To evaluate LLMs' understanding of Starsim, a Python-based disease modeling framework, we wrote a quantitative proficiency exam. Using this exam, we tested different LLMs and varied their access to tools. Scores ranged widely, from ~50% for Claude Haiku 4.5 and GPT-5 mini without tools to 98.7% for Claude Opus 4.8 with tools. While LLM choice had the biggest impact on exam scores, tool access also significantly improved performance for both the cheapest (Haiku) and most capable (Opus) models, with more modest benefits for intermediate models. Thus, to improve LLM performance on domain-specific problems, we recommend developing a set of AI tools with the help of quantitative evaluation.
---

## Introduction

Scientists are often decidedly unscientific about choosing AI tools to help them write code. Choosing an AI tool is not like choosing a text editor; the assistant not only shapes the code itself, but also the correctness of the results that code produces. Intuition is a poor guide to that choice: in one randomized trial, experienced developers systematically misjudged whether AI helped them [@becker2025measuring]. Scientists who program tend to favor general-purpose chat interfaces over developer-specific tooling, and their sense of productivity tracks most closely with how much generated code they accept, rather than with whether that code is correct [@obrien2025scientists]. 

Beyond trying different models, few developers rigorously test whether other changes to their workflow help. This is because writing and executing these evaluations is typically time-consuming, the results of the evaluations are difficult to interpret and quantify, and AI workflows and tooling are evolving rapidly. Even among engineering leaders who track AI at all, a minority do so systematically [@leaddev2025ai]. Here we describe our process for quantitatively testing our assumptions about how to build a good AI assistant.

If the antidote to vibes is measurement, the obvious place to look is existing benchmarks. A large and growing body of them evaluates the coding ability of LLMs and agents [@jiang2024survey], spanning tasks from resolving real-world software-engineering issues (SWE-bench; @swebench) to open-ended work in a terminal environment (Terminal-Bench; @terminalbench) and implementing routines for scientific computing (SciCode; @scicode). These have driven rapid progress, but they largely target widely-used, general-purpose libraries that are well represented in model training data, and they risk saturation and contamination as newer models are trained on published solutions [@white2025livebench].

We wanted to understand how well LLMs handle specialized scientific software: code that is comparatively rare in training datasets, evolves quickly, and demands domain expertise to use correctly. This is precisely the regime in which most research software lives; yet, by its very nature, it cannot be well captured by centralized, standardized benchmarks.

We study this regime concretely through Starsim, an open-source framework for disease modeling developed and maintained by our team. Starsim is a representative hard case: it has a relatively small set of online training data, a frequently updated API, and enough methodological subtlety that using it correctly requires domain expertise. Because we maintain it, we are also well placed to judge whether an AI's answers are actually correct.

## Methods

Our study has three ingredients: a proficiency exam that measures competence with Starsim, a suite of domain-specific AI tools intended to improve that competence, and an evaluation harness that runs a range of LLMs on the exam while allowing or disallowing their access to those tools. The exam is the heart of the method. It is human-written, open-ended, and marked against a detailed rubric, and it is designed so that a score reflects genuine command of the framework rather than recall of common examples. With it we ask two questions: how well do current models handle Starsim unaided, and does domain-specific tooling make them better? We describe each ingredient in turn, then report how the models and configurations performed.

### Starsim

[Starsim](https://starsim.org/) [@starsim; @starsim_scipy] is a high-performance agent-based disease modeling library built on NumPy [@numpy], SciPy [@scipy], and Numba [@lam2015numba]. A Starsim model is assembled from composable modules: diseases (such as HIV or tuberculosis), transmission networks (such as airborne transmission in workplaces or sexual networks), health interventions (such as vaccines and treatments), and demographics (such as pregnancies, births, and deaths). Agents correspond to indices in structured arrays, which provides performance orders of magnitude faster than a fully object-oriented implementation [@kerr2022python]. Starsim has been used to model domains ranging from family planning [@o2023fpsim] and sexually transmitted infections [@stuart2024hpvsim; @stuart2026reduction] to COVID [@kerr2021covasim]  and Ebola [@delport2024estimating].

The features that make Starsim powerful are also what make it hard for an AI to use correctly. It has a unit-aware time system spanning rates, probabilities, and durations; a custom random-number implementation that guarantees exact deterministic comparability across scenarios [@starsim_crn]; and performance-optimized indexing that tracks agents across births and deaths. In addition, since the diseases being modeled are often very complicated, the Starsim code needed to model them can also be very complicated. Using all of Starsim's features in a scientifically correct way demands familiarity with the framework itself as well as a strong grasp of disease modeling principles. The skills in our AI tools, and the questions in the exam, are primarily focused on this specialized knowledge.

### Starsim-AI

We created an open-source set of agent tools to improve LLMs' performance when working with Starsim, called [Starsim-AI](https://github.com/starsimhub/starsim_ai). Specifically, we added Model Context Protocol (MCP) servers for both Starsim and [Sciris](https://docs.sciris.org), which is a scientific Python library used widely in the codebase [@sciris]. We also created a set of "skills" for Starsim, which consist of problem-solving and feature-oriented Markdown files. Each skill carries a short natural-language description that the agent uses to decide when the skill is relevant, plus a body that is loaded into context only when triggered.

The skills fall into four groups. The largest is a set of *developer* skills, each covering one Starsim subsystem: getting started, configuring the simulation object, diseases, contact networks, interventions, demographics, probability distributions, calibration, multi-disease connectors, analyzers, the time and units system, random number generation, agent indexing, performance profiling, running and comparing multiple simulations, and building non-standard (e.g. compartmental) models. A second group of *style* skills encodes Starsim's conventions for Python code, testing, documentation, and overall design philosophy, so that generated code reads like code a core developer would write. The remaining skills cover the companion Sciris utility library and the [STIsim](https://github.com/starsimhub/stisim) extension for sexually transmitted infections. These skills were written by Claude Code using the `skill-creator` [plugin](https://code.claude.com/docs/en/skills), based on the (human-written) Starsim [tutorials](https://docs.starsim.org/tutorials) and [user guide](https://docs.starsim.org/user_guide). The skills were then manually reviewed and revised by Starsim core developers for accuracy and completeness.

### Starsim exam

We created a human-written "Starsim exam" evaluation suite designed to test knowledge about Starsim. Although used here for evaluating LLM performance, it is written in the style of a university final exam paper, and is applicable to both human and AI test-takers.

The exam is deliberately open-ended and code-centric. Most of its roughly 30 individually marked sub-questions require the test-taker to write runnable Starsim code, produce figures, and explain the resulting dynamics in prose, with marks awarded against a detailed rubric rather than a pass/fail unit test. This partial-credit design suits scientific code, where an answer can be largely right, and rewards command of the framework over recall of common patterns. The sub-questions total 300 marks and are designed to take a skilled human approximately three hours; they span the difficulty range from simple recall ("describe what Starsim is and does") to substantial modeling tasks (@fig-exam-question).

The exam covers five topics:

1. **Basics**: general questions about Starsim, and running simple simulations.
2. **SIS dynamics**: exploring the dynamical behavior of susceptible-infectious-susceptible disease models.
3. **SIRS models**: comparing agent-based and compartmental models.
4. **Modules**: calibrating models to data and implementing multi-disease models.
5. **Advanced**: profiling and debugging code; understanding time units.

A sixth section (innocuously labeled "Miscellaneous") checks for cheating, with questions answerable only with access to (a) the latest Starsim changelog, (b) the Starsim-AI skills, or (c) the solutions themselves.

```{figure} ./figures/fig0_exam_question.png
:label: fig-exam-question
:width: 80%

Example question from the Starsim exam, testing the ability to implement disease models, extract and analyze results, and understand the tradeoffs of different modeling approaches.
```

### Running the exam

We ran the exam through two independent harnesses to ensure the results were not an artifact of a particular tooling choice. In both, agents wrote their answers as Markdown files with executable Python code blocks. In "pipeline mode", evaluations were run using [Inspect](https://inspect.aisi.org.uk/), an evaluation framework developed by the UK AI Security Institute; agents ran in sandboxed Docker containers using Inspect's agent orchestration infrastructure, and Inspect also scored the output against the (human-written) marking schema. In "humanoid mode", agents were run directly using the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), which also performed marking based on the schema. The two approaches produced comparable results: pipeline mode gave greater control over the agents' environment and choice of LLM (including OpenAI models), while humanoid mode gave easier direct access to the agents' thinking and outputs.

In pipeline mode, we ran evaluations with two OpenAI models (GPT-5.4-mini and GPT-5.5) and three Anthropic models (Haiku 4.5, Sonnet 4.6, and Opus 4.8). Each model was run with three configurations: "chat only" (a single-turn response, with no ability to run or iterate on code), "agent only" (a multi-turn response with access to a Python environment, but without Starsim-AI, web search, or other tools), and "agent + skills" (a multi-turn response with access to Starsim-AI skills and MCP servers, plus regular web search). We could not run an agent with web search but without skills, since agents proved surprisingly effective at using web search to discover the Starsim-AI plugin and use it even when it had not been formally installed.

In humanoid mode, we ran the three Anthropic models (Haiku 4.5, Sonnet 4.6, and Opus 4.8). For Sonnet and Opus we also varied the "effort" level (low, medium, and high; exploratory runs at extra-high and max levels performed similarly to high and were not pursued further, to save tokens). These models were run with three configurations, slightly different from pipeline mode. "Chat only" was unavailable, since the Claude Agent SDK is multi-turn by default. "Agent only" was enforced through prompt instructions rather than Docker firewalls (cheating attempts, such as `curl` from Bash and `urllib` from Python, were surprisingly frequent, but were intercepted and blocked). "Agent + skills" was similar to pipeline mode. Humanoid mode also used a fourth configuration, "skills + nudged", in which the system prompt included the following nudge, added because exploratory runs showed plugin usage was low:

> You have the `starsim-ai` plugin loaded. It provides a set of Starsim-specific skills (covering e.g. diseases, networks, interventions, calibration, demographics, time and units, indexing, distributions, analyzers, and profiling) plus Context7 documentation lookup. USE THE `starsim-ai` SKILLS WHEREVER THEY ARE RELEVANT: before writing Starsim code for a sub-part, invoke the matching skill to ground your approach in the current Starsim API and idioms, and consult the relevant skill whenever you are unsure of the correct Starsim usage. These skills are authoritative for this version of Starsim; prefer them over your prior assumptions.

Each combination (LLM, configuration, and effort level where applicable) was run five times to account for stochasticity in both answering and marking. Answers were graded by LLM-based marking agents rather than by hand, with marks point-checked and cross-validated against the schema by the human experts who wrote the exam. Hand-marking was infeasible: a completed exam was typically about 2,500 lines, roughly half of it executable Python, amounting to approximately 400,000 lines of output across all experiments.

## Results

### Performance scales with cost and skills

Unsurprisingly, costlier models perform better. The cheapest models with single-turn ("chat") configurations, GPT-mini-chat (USD0.11) and Haiku-chat (USD0.21), also performed by far the worst (56.6% and 54.7%, respectively). The best-performing model, Opus-nudged (98.7%), was also among the most expensive (USD13.38). These results are shown in @fig-cost-score. Opus outperformed Sonnet on both performance and cost.

```{figure} ./figures/fig1_cost_vs_score.png
:label: fig-cost-score
:width: 90%

Performance as a function of model cost. Shape shows LLM type, and color shows model configuration. Individual runs are shown as small symbols; large symbols are the mean for that model-configuration combination. Not all models were run with all configurations.
```

Since skills tended to be evoked relatively infrequently without nudging (see below), we will focus on results that included nudging. Skills (with nudging) improved the performance of Haiku running in agent mode by 6.7% (from 68.9% to 75.6%), Sonnet by 2.0% (from 94.1% to 96.1%), and Opus by 1.7% (from 97.0% to 98.7%). While these gains may seem small, another way of looking at it is that skills reduced Opus' *mistakes* from 3.0% to 1.3% – a 57% improvement.

### Model effort quickly saturates

Claude models include a configurable "effort" parameter; higher effort consumes more tokens but is supposed to produce better results. As shown in @fig-effort-score, performance saturated above medium effort (extra-high and max effort levels were also run for a subset of trials, which also showed no further improvement; results not shown). Interestingly, skills (with nudging) improved performance in both the lowest-performance case (Sonnet-low, 92.0→95.1%) and the highest-performance case (Opus-high, 98.3→99.4%), but less in moderate-performance cases (Sonnet-medium, 96.7→96.1%; Opus-low 97.4→97.4%).

```{figure} ./figures/fig2_effort_vs_score.png
:label: fig-effort-score
:width: 90%

Performance as a function of effort for Sonnet and Opus models. Note that vertical bars show minimum and maximum scores, not confidence intervals.
```

### Marks are lost for multiple reasons

No single factor accounted for a majority of marks lost. As shown in @fig-lost-marks, marks were lost approximately equally due to the answer being (a) incorrect, (b) omitted, or (c) for a different reason. Interestingly, the primary improvement in marks was due to a reduction in the number of omitted answers, rather than incorrect answers. This suggests that skills can be helpful in prompting the LLM to bring the solution to completion when it may have otherwise become stuck.

```{figure} ./figures/fig3_lost_marks.png
:label: fig-lost-marks
:width: 90%

Exam marks lost as a function of reason.
```

### Marking is robust to judge choice

Because the answers are marked by LLMs, a natural concern is whether the results are an artifact of a particular judge. @fig-judge-agreement addresses this by plotting the Anthropic judge's score against the OpenAI judge's score for every marked answer. The two judges agree closely (Pearson $R^2$ = 0.96) with only a small mean difference (the Anthropic judge is higher by 2.2% on average). The high cross-provider correlation indicates that the scores reflect properties of the answers rather than the idiosyncrasies of a single marker, and gives us confidence in the comparisons above.

That said, the panel does reveal a small but consistent own-provider bias: each judge scores answers written by its own provider's models slightly more generously. On average, there was a preference of 2.1% for a judge towards its own work versus the other provider's. While much smaller than the variance in scores (in one extreme case, there was a 40-point gap between the Anthropic and OpenAI judges' scores on the same exam), it is still notable since even with a rigorous and explicit marking schema, it shows that a model's "personal" bias is still present.

```{figure} ./figures/fig4_judge_agreement.png
:label: fig-judge-agreement
:width: 100%

**(a)** Agreement between the two independent rubric judges. Each point is one graded answer, plotting the Anthropic judge's score (x) against the OpenAI judge's score (y); the dashed line is perfect agreement. The judges agree closely, across both agent configurations (color) and model providers (marker). **(b)** Distribution of score differences between the two judges. An unbiased judge would have a distribution centered on 0; as it is, the Claude judge has a slight preference for Claude models (shifting the distribution to the right), while the Anthropic judge is the opposite (shifting to the left).
```

### Agents were reluctant to use skills

To preserve tokens, agents only invoke skills (which get added to the context window) when they think they need them. However, agents can be overconfident ("you don't know what you don't know"), which in turn results in under-utilization of skills.

In order to quantify skill utilization, we logged every time any agent invoked a skill. The combined set of skill uses across all runs is what we defined the denominator to be, i.e. "100% utilization". (While technically this number is dependent on the number of runs, we found it had mostly saturated in the 5 runs included here.) Out of these possible skill uses, we counted how many actual uses any given run had.

As shown in @fig-skill-utilization, without nudging, agents invoked skills only 7.8% of the possible times they could have. This accounts for the relatively small performance gain provided by simply including skills. When provided with the additional prompt nudge, skill usage increased to 28.4% – a 3.6-fold increase. We attribute to skill under-use to agent overconfidence rather than an optimal efficiency tradeoff, since the nudged agents not only had better performance, they also sometimes had lower cost (see @fig-cost-score). Surprisingly, skill usage was relatively uniform across the questions, even though the difficulty of the questions varied markedly, and skills were of much more help for some questions than others.

```{figure} ./figures/fig5_skill_utilization.png
:label: fig-skill-utilization
:width: 90%

Utilization of Starsim-AI skills for each question in the exam.
```

## Discussion

AI has progressed so rapidly that even over the 6-month duration of this project (January – June 2026), our initial findings are no longer valid. Specifically, when we first ran these analyses, performance from the highest performing model (Opus 4.6) was 70% without skills and 91% with skills – a staggeringly different result from what we find now for the best model (Opus 4.8; 97.0% without skills and 98.7% with skills). Even so, most of the qualitative trends remain: skills can still fix a large proportion of the errors (although 60% of a 3% error rate looks less impressive than 60% of a 30% error rate), and more expensive models do tend to perform better (although the worst models we tested now are roughly equivalent to the best models of 6 months ago).

In a sobering finding for anyone working on AI safety, we were surprised at the lengths to which agents went to cheat: scanning local folders and public GitHub repos for solutions, and attempting to use Bash and Python to circumvent explicitly blocked web search tools (to our amazement, `curl -s https://duckduckgo.com` became a frequently invoked command). This suggests that even with unambiguous prompting ("This is a CLOSED-BOOK exam and network access is PROHIBITED. Do NOT try to work around this restriction by any means."), when an agent is told it is sitting an exam, it seems it will often try to maximize its score at all costs. This was especially jarring in light of the agents' reluctance to use all the skills that had been provided to them.

This study has several limitations. First, there are nearly infinite possible variations in terms of skill content, description text for skill triggering, agent prompts, exam question content, etc. It is possible different choices made in each of these areas would have significantly impacted results, since skill content was tuned primarily based on human user feedback. Second, while the exam was human-written, the marking was done by LLMs; although we checked these marks where we could, we were not able to check everything, and given the relatively small differences between different models and configurations, differences of as few as 5 marks out of 300 could have a significant impact on the results. Finally, new models are continually released; whether or not these results remain valid for newer models like Claude Fable remains to be seen.

## Conclusion

One of the most valuable benefits of writing quantitative evaluations for an LLM is that it forces a decision on what questions are most important, and what "correct" means for those questions. While these are often relatively clear-cut for software engineering tasks ("Does the server run?"), they become decidedly murkier on many scientific questions. Especially in disease modeling, since "all models are wrong" [@box1976science], deciding which answers count as "correct" is often a matter of personal preference. However, even if unambiguous rigor is typically unattainable, imperfect rigor is still preferable to placing one's faith in the vibes.

## Disclosures

Portions of this work were assisted using generative AI tools (Claude Code and GPT). The tools were used to assist with (1) writing code, (2) taking the exams, (3) marking the exams, (4) extracting data and summarizing results, (5) generating figures, and (6) drafting the initial outline of the manuscript. At each stage of the process, outputs were reviewed, verified, and revised by the authors, and the vast majority (>95%) of the text in the manuscript was human-written. The authors take full responsibility for the accuracy and integrity of the final content.
