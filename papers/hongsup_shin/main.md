---
# Keep this title identical to the one in myst.yml
title: "Where Agentic Judgment Earns Its Place in High-Stakes Data Enrichment"
abstract: |
  Automated data enrichment, filling missing fields in structured records from unstructured sources, is the canonical case for pointing an autonomous agent at a database and letting it fill every blank. In high-stakes data that instinct is dangerous. A confidently wrong value is worse than a blank, and retrieval-grounded extraction reduces but does not remove the tendency to assert what the source never stated. The question is not whether an LLM can extract these fields (it can) but where agentic judgment earns its place and where it is a liability.

  We study this on the Texas Justice Initiative's police shooting databases, where nearly two thousand records are missing the weapon, the subject's race, or the outcome, whose fields volunteers typically recover by hand, fifteen to thirty minutes each. Our deterministic LangGraph pipeline searches, validates, extracts, and escalates hard cases to a human. It completes 92% of officer and 70% of civilian records and invents zero facts across twenty fabricated incidents. The recovery itself came from a deterministic prompt fix without any agent.

  Agency earns its place only in a judgment layer above extraction, where we separate two senses: redirecting control flow versus holding authority over a produced value. One component reaches control flow: a relevance judge whose veto escalates a wrong-article record. Two more hold bounded authority, to null an unstated value or advise a reviewer; extraction calls an LLM yet only proposes, so it is not agentic. Each cleared a reward-hacking-resistant evaluation gate before shipping. The contribution is a discipline (an "earn-it" protocol) for deciding where agentic judgment belongs in high-stakes data work.
---

## Introduction

In Texas, the Office of the Attorney General (OAG) is required by law to collect a report on every officer-involved shooting (OIS) and to publish an annual summary. Those summaries are high-level: they omit age demographics, report dates, and the number of officers involved, and they carry no intersectional analysis [@tji2020ois]. The Texas Justice Initiative (TJI), a nonprofit, re-publishes the underlying incident records with far more granularity, and that granularity is what makes independent analysis possible at all. But the records are incomplete. About a quarter of incidents are filed with an unidentified cause; race is recorded with a coarse vocabulary that law enforcement is known to mischaracterize; weapon and outcome fields are frequently blank [@tji2020ois]. These gaps carry substantive consequences: undercounting and misclassification of police violence are documented at the national scale [@gbd2021police], and each blank field leaves a question about a police shooting that the public record cannot answer.

TJI addresses these gaps through the standard practice of accountability nonprofits: volunteers search news archives one incident at a time, read the coverage, and transcribe the details. The work takes fifteen to thirty minutes per record, and there are nearly two thousand records. This is precisely the kind of tedious, search-and-extract task that modern large language models (LLMs) appear built for, and the temptation is to point an autonomous agent at the database and let it fill every blank. Human-factors research has long documented that operators over-trust automation in exactly such conditions [@parasuraman2010complacency].

This paper argues against that approach for high-stakes domains; that argument, together with a system that embodies the alternative, is its contribution. The failure mode that matters is a *confidently wrong* field rather than a missing one (volunteers already live with blanks): a fabricated weapon, a misattributed race, a detail lifted from a different shooting that shares a city and a date. Generative extraction grounded in retrieved documents [@lewis2020rag] reduces but does not remove *unfaithfulness*, the tendency to emit content the source does not support, a failure documented in summarization [@maynez2020faithfulness] and surveyed broadly for LLMs [@ji2023hallucination; @huang2025hallucination]. Uncritical automation therefore recovers and fabricates in the same pass [@bender2021parrots], and a fabricated fact in an accountability database is worse than the blank it replaced. The engineering question, then, is less whether an LLM can extract these fields (it can) and more how to earn the right to trust what it extracts, and how to decide which decisions deserve an LLM at all.

Our answer is a system designed as a *workflow* with bounded agentic judgment rather than an autonomous agent. The distinction is a useful design axis [@anthropic2024agents]: a workflow orchestrates models and tools along predefined code paths, whereas an agent lets the model direct its own process, as in ReAct-style loops that interleave reasoning and action under the model's control [@yao2023react]. Frameworks such as LangGraph [@langgraph] make either style straightforward to build as a stateful graph, but autonomy is a cost as much as a capability: every place the model is allowed to choose is a place it can choose wrong. Two senses of *agentic* are worth separating: whether a model *directs control flow* (chooses what runs next), and whether a model *holds authority over a record*, disposing of an already-produced value rather than only proposing one ({ref}`tbl:senses`). Our orchestration is deterministic in the first sense, a state machine in which no model directs the sequence of operations; the one routing outcome attributable to a model judgment is the relevance judge's veto, which the Coordinator reads to escalate a wrong-article record while deterministic code executes the move. What separates the judges from extraction is the second sense: extraction only proposes content, whereas three small bounded judges hold authority to veto, null, or flag what it produced, applying the now-common pattern of using an LLM to judge a narrow, checkable question [@zheng2023judge] ("do these articles actually describe this incident?"). Their authority is graduated, from the relevance judge's veto down to the conflict annotator's advisory note. Each judge earned its place by clearing an offline evaluation gate designed to resist the most common way such systems fool their builders: optimizing the headline metric (here, completion rate) at the expense of the property that actually matters (faithfulness). We make three claims that should transfer beyond this case:

1. **Agentic judgment belongs to the evidence layer, above extraction.** A deterministic prompt change produced the single largest gain in this project; the judges add faithfulness and leave recall to retrieval.
2. **A multi-objective, reward-hacking-resistant evaluation gate is the principal safeguard.** Optimizing a proxy metric reliably degrades the true objective once the two diverge [@amodei2016concrete; @skalse2022reward], and "complete more records" is trivially gamed by accepting weak extractions; the gate therefore scores every change on three separate guards: completion (tracked and surfaced, so a drop is never silent), a hard zero-hallucination veto, and field-level correctness on a stable cohort, measured so that a completion gain cannot launder a correctness loss.
3. **Authority should be calibrated to stakes.** Our three judges can *block* a record into human review, *null* an unsupported value, or merely *advise* a reviewer; they never silently overwrite the record.

The system is open source, and all quantitative results reported below are produced by the pipeline's evaluation harness, applied to the data described next.

## Methods

### Datasets

TJI publishes two related datasets that we treat throughout: `civilians_shot` (police shooting civilians; 1,674 records) and `officers_shot` (civilians shooting police; 282 records), spanning 2014–2024, for 1,956 records total. They use different field names for analogous concepts (in `officers_shot` the civilian is the *suspect/shooter* and the outcome is the *officer's* injury or death), a divergence the pipeline handles explicitly through a `DatasetType` enum that switches the database queries and tailors the extraction prompt to each dataset's semantics. In the source data, 57% of civilian records are missing the weapon, 22.5% are missing the subject's name, and 39% of officer records are missing the officer's name. We measure accuracy against a held-out sample of these records, defined in Evaluation design below.

### Pipeline architecture

The pipeline is a seven-node LangGraph state machine ({ref}`fig:pipeline`). Each node accepts and returns a single typed state object (a `pydantic` model [@pydantic]), and a deterministic **Coordinator** reads the stage that just ran and routes the record to retry, proceed, or escalate:

- **Load** reads incident anchors from PostgreSQL [@postgresql] via `psycopg2` [@psycopg2].
- **Search** queries the Tavily web-search API [@tavily] within a date window around the incident.
- **Validate** keeps only articles that match the incident on a tiered rule: one with a parsed publication date must match on date proximity (±5 days) and location; one lacking a date must match on location and the victim's name; with neither, it falls back to location alone, keeping date-less results usable without admitting articles that share only a city.
- **Synthesize** runs the LLM extraction and the agentic layer described below.
- **Complete** and **Escalate**, the two terminal nodes, write a JSON result or an escalation report for human review.

When validation leaves too few usable articles, the Coordinator climbs a fixed three-rung search ladder before giving up, broadening the query from an exact-date match to a month-and-year window and then to a name-only query (dropping the officer's name, keeping the civilian's), and escalating only after the third rung fails.

::::{figure}
:label: fig:pipeline

```{mermaid}
flowchart TD
  Start([Start]) --> Load[Load]
  Load --> Coord{0. Coordinator}
  Coord -- "✅" --> Search[1. Search: Tavily API]
  Search --> Coord
  Coord -- "✅" --> Validate[2. Validate: date / loc / name]
  Coord -- "🔁" --> Search
  Validate --> Coord

  subgraph Synth [3. Synthesize]
    direction LR
    Extract[Extraction] --> RJ{{Relevance judge: Block}}
    RJ --> RV{{Race verifier: Null}}
    RV --> CA{{Conflict annotator: Advise}}
  end

  Coord -- "✅" --> Synth
  Synth --> Coord
  Coord -- "✅" --> Complete([4A. Complete: Write JSON])
  Coord -- "🚩" --> Escalate([4B. Escalate: Human review])

  classDef judge fill:#dbe9ff,stroke:#2f6fb0,color:#111
  class RJ,RV,CA judge
```

The seven-node pipeline. After the Load entry node, the deterministic Coordinator (0) is the hub through which every transition passes (proceed, retry, or escalate); the stage numbers give the happy-path order, 1 Search, 2 Validate, and 3 Synthesize, ending at one of two terminals, 4A Complete or 4B Escalate (a ✅ edge is a passed gate, 🔁 a retry, 🚩 an escalation to human review). On thin retrieval it climbs a fixed three-rung search ladder (exact, then temporal, then name-partial) before escalating. Inside the Synthesize node, after extraction, three bounded LLM judges (the shaded hexagons) run in sequence as the agentic layer, each with authority calibrated to stakes: the relevance judge can *block* a wrong-article completion (the Coordinator then escalates it as `irrelevant_sources`), the race verifier *nulls* an unstated race, and the conflict annotator *advises* the human reviewer. The judges run as sub-steps inside the Synthesize node, distinct from the graph's nodes.
::::

Each run ends at one of two terminal nodes, and the JSON it writes ({ref}`fig:output`) is what a human reviewer reads. A completion records every extracted field with a confidence label and its source URLs; an escalation names the reason and routes the record to review rather than committing anything.

::::{figure}
:label: fig:output

A completed record (`civilians_shot` incident 792):

```json
{
  "incident_id": "792", "dataset_type": "civilians_shot",
  "extracted_fields": [
    {"field_name": "weapon", "value": "Knife", "confidence": "medium",
     "sources": ["click2houston.com/news/local/2020/..."], "extraction_method": "llm"}
    // 6 more: time_of_day, circumstance, officer_name, civilian_name, location_detail, outcome
  ],
  "search_strategy": "name_partial", "retry_count": 2,
  "outcome_summary": "Enriched 7 fields for incident 792 (civilians_shot)"
}
```

An escalated record (`officers_shot` incident 75), vetoed by the relevance judge:

```json
{
  "incident_id": "75", "dataset_type": "officers_shot",
  "escalation_reason": "irrelevant_sources", "relevance_vetoed": true,
  "retrieved_articles": [
    {"title": "Capital murder trial of man accused of killing SAPD officer during 2013 chase begins"}
  ],
  "outcome_summary": "Escalated incident 75: no retrieved article reports this 2018 incident"
}
```

The two terminal outputs (field names verbatim from the pipeline's JSON schema; values abridged). The completion (top) logs each field's confidence and sources; the escalation (bottom) is the relevance judge blocking an article that passes every rule-based check but reports a different (2013) case.
::::

Two design choices are deliberate. First, **the orchestration is deterministic**: the Coordinator is `if`/`match` logic over state fields and the retry ladder is a fixed list. We chose this over an LLM router or supervisor because, in a high-stakes domain, predictable control flow is itself a safety property. Second, the system is **human-in-the-loop (HITL) by construction**: it never writes back to the source database, and escalation is a first-class terminal outcome rather than a failure.

The entire graph is assembled in one function (simplified from `src/agents/graph.py`):

```python
def build_graph(checkpointer=None) -> CompiledStateGraph:
    workflow = StateGraph(EnrichmentState)        # state schema = a pydantic model
    workflow.add_node("load", load_node)
    workflow.add_node("search", search_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("complete", complete_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("coordinate", coordinate_node)

    workflow.add_edge(START, "load")
    workflow.add_edge("load", "coordinate")       # every processing node
    workflow.add_edge("search", "coordinate")     #   returns to the coordinator hub
    workflow.add_edge("validate", "coordinate")
    workflow.add_edge("synthesize", "coordinate")
    workflow.add_edge("complete", END)
    workflow.add_edge("escalate", END)

    # the only branch in the graph, and it is deterministic (not an LLM router)
    workflow.add_conditional_edges("coordinate", route_after_coordinator)
    return workflow.compile(checkpointer=checkpointer)
```

#### How we use LangGraph

LangGraph models a pipeline as a graph whose nodes are plain functions and whose state is a single typed object that each node reads and returns. We use it deliberately narrowly. The state schema is the `EnrichmentState` `pydantic` model, so every field is validated and the whole run is one inspectable object, and there is exactly one conditional edge, out of the Coordinator, whose target is chosen by a deterministic `match` on the stage that just ran (simplified from `src/agents/coordinate_node.py`):

```python
match state.current_stage:
    case PipelineStage.LOAD:       state = check_load_results(state)
    case PipelineStage.SEARCH:     state = check_search_results(state)
    case PipelineStage.VALIDATE:   state = check_validate_results(state)
    case PipelineStage.SYNTHESIZE: state = check_synthesize_results(state)
# route_after_coordinator(state) then returns state.next_stage.value
```

This sits at the opposite end of the autonomy spectrum from the agent abstractions the same ecosystem also offers: where LangChain's agent executors, CrewAI's crews, or AutoGen let a model choose tools and coordinate other agents, LangGraph exposes the graph and its typed state directly [@langgraph; @langchain], which is what we wanted for deterministic, inspectable control. The one routing outcome attributable to a model judgment is the relevance judge's veto, which escalates a wrong-article record; its sibling escalations, on empty or conflicting extractions, are model-agnostic completeness guards that fire the same way for a regex extractor, whereas the veto has no rule-based equivalent. The Coordinator reads the veto like any other state field: the model supplies the judgment and deterministic code executes the move. Runtime dependencies (the LLM client, the `Settings`) are injected through the framework's `RunnableConfig`, so in production a node reads a Claude [@anthropic_claude] client and in unit tests a `MagicMock`, which is what makes the agentic judges testable without a live model or network.

### Deterministic conflict reduction first

Before any LLM is consulted, deterministic code exhausts what rules alone can resolve, so the judges are reserved for what code genuinely cannot do. The steps run in pipeline order:

1. **Aggregation sources are excluded** outright: multi-incident pages such as Wikipedia lists or Fatal Encounters records contaminate extraction by folding several incidents into one document, so the search and validation nodes drop those domains and any PDF or CSV URL, leaving only documents about a single incident.
2. **Extraction is anchored to the record's subject**: when an article describes several people (a second victim, a bystander, the officer), the prompt extracts fields only for the target individual instead of blending them.
3. **Names, race terms, and weapon categories are normalized** before comparison, so that honorific and phrasing variants do not register as disagreements.
4. **A consensus resolver** (`field_normalizers.py`) commits an `outcome` or `time_of_day` value when at least two articles agree on the same canonical form; relatedly, a record can **complete partially**, keeping the fields the sources agree on while routing only a genuinely conflicting field to review.
5. **A race taxonomy** (`race_taxonomy.py`) maps race terms to TJI's buckets for matching while preserving the raw value and flagging divergences for review.

{ref}`tbl:examples` shows several of these behaviors on concrete cases.

### Agentic judges with graduated authority

Only after deterministic extraction do three LLM judges reason about what rules cannot, each granted authority **calibrated to the stakes** of its decision, from block for the highest-confidence check down to advise for the fuzziest. Every judge is deliberately bounded: a single structured-output call (no loops, no tools), *fail-open* (an error is logged and the pipeline proceeds as if the judge had not run), and read-only with respect to the database. The worst case any judge can produce is one extra human review, never a corrupted record.

Each judge is a typed schema plus a single structured-output call; the relevance judge's is representative (simplified from `src/synthesize/relevance_judge.py`):

```python
class RelevanceVerdict(BaseModel):
    relevant_any: bool = Field(description="True if >=1 article reports THIS incident")
    relevant_indices: list[int] = Field(default_factory=list)
    reasoning: str = Field(description="One sentence.")

def judge_relevance(llm_client, state, articles) -> RelevanceVerdict:
    prompt = _build_prompt(state, articles)
    return llm_client.with_structured_output(RelevanceVerdict).invoke(prompt)
```

#### Relevance judge: block

The relevance judge runs on both datasets, asking whether the retrieved articles actually report *this* incident; when they do not, it vetoes the completion and escalates the record as `irrelevant_sources`. It catches the "right structure, wrong incident" failure that rule-based validation cannot: an article from the correct city, published on the incident date, that nonetheless reports a different shooting. The clearest case is the San Antonio March 2018 officer shooting of {ref}`fig:output`, where an article published on the incident date in fact recounts a *2013* trial; reading the text, the judge caught what the rules could not, and the record escalated. The same gate closes the civilian famous-name collision, a victim sharing a name with a high-profile case whose coverage is all about the other person ({ref}`tbl:examples`).

#### Race verifier: null

The race verifier runs on civilians only and asks whether the source *explicitly states* the race of this subject. It is a faithfulness filter: it nulls any `civilian_race` the source does not support, never inferring race from a name, neighborhood, or photo, which mirrors TJI's own caution that race is frequently mischaracterized [@tji2020ois]. Its restraint is visible on the holdout: of the eleven civilian races it committed, only one disagreed with the database, recording Black where the database says Hispanic.

#### Conflict annotator: advise

The conflict annotator, run on the cheaper Claude Haiku model across both datasets, writes an advisory triage note when a deep conflict reaches a human, explaining *why* the sources disagree (a lawsuit and a police report diverging on the weapon and circumstances, for instance). Because it never commits a value, it cannot change accuracy or coverage by construction; its only effect is to speed the reviewer.

Building these judges taught design lessons that generalize. Date works as a *hard* gate at two granularities, which the San Antonio case makes concrete: the Validate node checks the article's *publication* date (within ±5 days of the incident), while the relevance judge checks the date of the *event the article describes*, since a story published on the incident date can still recount a shooting from years earlier. The recorded outcome, by contrast, is only *supporting* context, because the database's own harm field is sometimes stale, so the judge treats it softly. Cost discipline runs throughout: high-frequency or simple calls use Haiku, rare or complex ones use Claude Sonnet [@anthropic_claude].

```{list-table} Where agency lives. Each component is classified by the two senses of *agentic*: does a model direct control flow (a model-attributable routing outcome), and does a model hold authority over the record (veto, null, or flag a produced value)? Four components call an LLM, yet only the three judges hold authority and only the relevance judge reaches control flow; extraction calls an LLM but only proposes content, so it is non-agentic despite that call, a class of its own apart from the deterministic substrate.
:label: tbl:senses
:header-rows: 1

* - Component
  - LLM?
  - Directs control flow?
  - Holds authority?
  - Class
* - Relevance judge
  - Yes
  - Yes (routes to escalation)
  - Yes (block)
  - Agentic, both senses
* - Race verifier
  - Yes
  - No
  - Yes (null)
  - Agentic, authority
* - Conflict annotator
  - Yes (Haiku)
  - No
  - Yes (advise)
  - Agentic, authority
* - Extraction
  - Yes
  - No
  - No (proposes)
  - Not agentic
* - Orchestration, conflict reduction, evaluation gate
  - No
  - No
  - No
  - Deterministic
```

### Multi-objective evaluation gate

A multi-objective gate decides whether a candidate change is safe to ship (`gate.py`). It plays the role a continuous-integration check plays for ordinary software, except that here the "tests" are accuracy and safety metrics, a pattern any developer who has gated a merge on a passing suite will recognize. The gate is a *pure function* over two saved holdout reports (before and after), so a candidate change can be judged with zero new inference cost. It computes three guards and accepts a change only when all three hold (`accept = target_ok and adversarial_ok and correctness_ok`): a **target** guard (completion must not regress beyond tolerance), a hard **adversarial** veto (the fabricated-incident hallucination count must be exactly zero), and a **correctness** guard (volume-weighted field accuracy on a *stable cohort* must not drop beyond tolerance). With the default tolerances (target 0.0, correctness 0.02, adversarial exactly 0), the gate flags any completion regression and any cohort-correctness drop beyond two points, and treats a single fabricated-incident hallucination as a hard, never-overridden veto. The cohort restriction (incidents completed in *both* runs) and the volume weighting are what make it resistant to reward hacking: without them, a change that raises completion by pulling easier incidents into the completed set could mask a correctness regression on the records that were always there, a Simpson's paradox in disguise. Per-race completion equity is reported as a *non-gating warning* (tolerance 0.05), never a veto, because at these group sizes it is underpowered.

A failed guard flags a change for closer review without automatically rejecting it. The relevance judge illustrates this: enabling it lowered officer completion from 95% to 92% by vetoing wrong-article completions, which trips the target guard. The change shipped regardless, because the lost completions were verified offline to be wrong-article and the hard guards held (zero hallucinations, no cohort-correctness drop). The gate's value is that it made the completion-for-faithfulness trade explicit and measurable.

### Earn-it protocol

No agentic component shipped on intuition. Each was first evaluated offline, on saved data, against the single dimension that matters for it (veto precision for the relevance judge, faithfulness for the race verifier, note quality for the annotator), and shipped only if it cleared that bar and the multi-objective gate. The discipline cut both ways: it gated *out* most of the agentic ideas we tried, and those negative results are as much the contribution as the components that survived. {ref}`tbl:process` summarizes the whole process: of seven ideas, three shipped (two of them moving the reported metrics; the conflict annotator is advisory and inert to them by construction) and four were gated out, failed, deferred, or declined.

```{list-table} The agentic ideas we considered, how each was evaluated offline on saved data, and the verdict.
:label: tbl:process
:header-rows: 1

* - Idea
  - What it would add
  - How we evaluated it (offline, on saved data)
  - Verdict
* - Relevance judge
  - Block wrong-article completions
  - Veto-precision audit on saved completions
  - Shipped
* - Race verifier
  - Null an unstated `civilian_race`
  - Faithfulness check on saved race extractions
  - Shipped
* - Conflict annotator
  - Advisory triage note on deep conflicts
  - Note-quality read on saved conflicts (Haiku)
  - Shipped (advisory)
* - ReAct extraction loop
  - Iterative reason-and-act extraction
  - Re-extraction on the officer insufficient-source cases
  - Gated out (a deterministic prompt closed the gap; residual was variance)
* - Race-divergence typer
  - Classify news-vs-record race disagreements
  - Earn-it on the bucket-different cases
  - Failed (every divergence was unfaithful, already nulled)
* - LLM conflict reconciler
  - Resolve field conflicts
  - Zero-cost sizing of scored conflicts vs ground truth
  - Deferred (most conflicts are coverage gaps; few need reasoning)
* - Self-consistency (k-pass)
  - Majority vote over k extractions
  - Cost-versus-residual-variance estimate
  - Declined (cost scales with k; variance already absorbed by the prompt fix)
```

### Evaluation design

Evaluation is a first-class part of the engineering here, rather than an afterthought: an accuracy dev set measures field-level correctness, an adversarial suite measures whether the pipeline refuses when it should, and both feed the multi-objective gate above.

The accuracy signal comes from fields that already exist in the database but are *hidden* from the pipeline during enrichment (age, race, weapon, location, time, outcome). The pipeline never sees them; we compare its independently extracted values against them afterward, a natural held-out test. We draw a deterministic, year-stratified sample of 100 records per dataset, with field-appropriate comparators: exact match for age, race, and outcome; category normalization for weapon; fuzzy string match (`rapidfuzz` [@rapidfuzz], threshold 80) for location; and a ±2-hour comparison for time of day.

The safety signal comes from an adversarial probe of 20 fabricated incidents: invented names, real Texas cities and dates, including six "traps" placed within days of real high-profile events so that real articles about the *wrong* person would pass date and location checks. The twenty span five designed categories: obscure towns where no coverage should exist, dates deliberately outside the validation window, the six hallucination traps, common-name confusions (a fabricated "Michael Brown" in a major city), and null-name edge cases. Only the database fetch is patched; every downstream node runs live, so the probe exercises the real retrieval, validation, and agentic layers.

## Results

### Held-out accuracy

A record *completes* when it reaches the Complete node with at least one field extracted; otherwise it escalates to human review. On the held-out set (100 records per dataset; {ref}`tbl:holdout`), the pipeline completes 70% of civilian records and 92% of officer records; officers complete far more often because officer-involved shootings draw denser news coverage, so retrieval rarely fails. Among extracted values, aggregate exact-match precision is 77% for civilians and 71% for officers, rising to 89% and 86% under fuzzy match. Of the 30 civilian escalations, 22 are retrieval gaps (no articles found), 7 are relevance-judge vetoes, and 1 is insufficient sources; the officer escalations split 4/3/1 the same way. No record escalates on a conflict alone, since partial completion commits the agreed fields and routes only the contested one to review.

```{list-table} Held-out results, 100 records per dataset, all judges on; Claude Sonnet, with the conflict annotator on Claude Haiku.
:label: tbl:holdout
:header-rows: 1

* - Metric
  - civilians_shot
  - officers_shot
* - Completion rate
  - 70% (70/100)
  - 92% (92/100)
* - Escalation rate
  - 30% (30/100)
  - 8% (8/100)
* - Aggregate exact
  - 77% (210/272)
  - 71% (147/207)
* - Aggregate fuzzy
  - 89% (243/272)
  - 86% (179/207)
```

Per-field, the strongest civilian fields are age (95% exact) and outcome (92%); weapon is 83% after category normalization and time of day 82% ({ref}`tbl:perfield`). Location is the familiar exact/fuzzy split (16% exact but 91% fuzzy) because the pipeline returns the correct city inside a fuller string that does not match the street-level ground truth. Officer fields follow the same shape ({ref}`tbl:perfield-off`). Outcome errors are almost entirely *conservative*: civilians have 6 of 74 wrong, all "fatal" where the truth is "survived" (100% fatal recall); officers have 12 of 94 wrong, of which 11 are conservative and one is a reverse error, the pipeline reporting a non-fatal outcome where the database records a death (the single fatal-recall miss across both datasets). These outcome errors share a mechanism: they cluster on *outcome-only completions* (12 officers, 4 civilians) whose only strongly supported field is the generic outcome, the signature of a different shooting at the same place and time having been matched.

```{list-table} Per-field accuracy, civilians_shot. *Evaluable* is the number of records with non-null ground truth; *extracted* is how many the pipeline filled. The extracted counts sum to 272, the aggregate denominator. Accuracy is over extracted values.
:label: tbl:perfield
:header-rows: 1

* - Field
  - Evaluable
  - Extracted
  - Exact
  - Fuzzy
* - civilian_age
  - 100
  - 58
  - 95%
  - 95%
* - outcome
  - 100
  - 74
  - 92%
  - 92%
* - weapon
  - 84
  - 46
  - 83%
  - 83%
* - time_of_day
  - 92
  - 39
  - 82%
  - 82%
* - location_detail
  - 100
  - 44
  - 16%
  - 91%
* - civilian_race
  - 100
  - 11
  - 91%
  - 91%
```

```{list-table} Per-field accuracy, officers_shot (extracted counts sum to 207, the officer aggregate). Here civilian_age and civilian_race are the suspect's.
:label: tbl:perfield-off
:header-rows: 1

* - Field
  - Evaluable
  - Extracted
  - Exact
  - Fuzzy
* - civilian_age
  - 98
  - 67
  - 82%
  - 82%
* - outcome
  - 100
  - 94
  - 87%
  - 87%
* - location_detail
  - 100
  - 37
  - 11%
  - 97%
* - civilian_race
  - 99
  - 9
  - 67%
  - 67%
```

Denominators differ by field because each is scored only where ground truth exists. Small cells warrant caution: `civilian_race` accuracy is over the 11 values the verifier committed, so its 91% (and the 65%→91% gain reported below) is a point estimate on roughly a dozen records. Each extracted value also carries a self-reported confidence label, and that label is usefully calibrated: high-confidence extractions are markedly more accurate than medium-confidence ones (roughly 93% versus 68% exact on civilians, 86% versus 54% on officers), so it is a usable triage signal for a human reviewer.

### Qualitative behavior

Beyond aggregate rates, {ref}`tbl:examples` collects representative cases, the kind of "right structure, wrong fact" situations that motivate the whole design, and shows which mechanism handles each.

```{list-table} Illustrative cases, drawn from the evaluation and pilot analyses, of how each mechanism behaves. They illustrate the final system's behavior on each kind of case, separate from the holdout frequencies above.
:label: tbl:examples
:header-rows: 1

* - Case
  - What the source had
  - What the pipeline did
  - Mechanism (authority)
* - Famous-name collision
  - A victim sharing a name with a high-profile case; every retrieved article covers the famous case
  - Escalated as `irrelevant_sources`
  - Relevance judge (**block**)
* - Race not stated
  - Coverage naming the victim but never stating their race
  - Left `civilian_race` null
  - Race verifier (**null**)
* - Genuine disagreement
  - A lawsuit and a police report differ on the weapon and circumstances
  - Surfaced to a human with a note on why the sources differ
  - Conflict annotator (**advise**)
* - Multi-party article
  - One article covering the shooting subject and an unrelated carjacking victim
  - Extracted fields for the subject only
  - Victim-anchored extraction
* - Honorific variants
  - "Master Sgt. Alva Joe Gwinn" in one article, "Alva Joe Gwinn" in another
  - Treated as the same name, no conflict raised
  - Name normalization
* - City inside an address
  - "100 block of Couch Court, Springtown, Parker County" vs ground truth "Springtown"
  - Counted correct (the right city)
  - Fuzzy location match
```

### Agentic faithfulness *after* deterministic recovery

The single largest gain on the officer dataset came from a **dataset-aware extraction prompt**; no agent was involved. `officers_shot` needs the model told that the civilian is the suspect and the outcome is the officer's fate. Without that framing, most officer records escalate as insufficient sources; with it, they become completable. Officers complete at 92% on the holdout, with insufficient-source escalations all but eliminated (1 of 100). We had expected the recovery itself to be where an agent would help, and we tested that: a reasoning-and-acting extraction loop, run offline on the same officer failures, recovered nothing the deterministic prompt had not. That null result is why we locate agentic judgment above extraction rather than inside it.

The judges, by contrast, are visible precisely as a *faithfulness* layer that often trades a little coverage for correctness. The race verifier drops civilian-race coverage from 17% to 11% while lifting exact accuracy from 65% to 91%: it removes races the source never stated. The relevance judge lowers officer completion from 95% to 92% by vetoing three wrong-article completions into review, and accounts for 7 of the 30 civilian escalations in the same way. Rather than accuracy regressions, these are the system declining to assert what it cannot support. We therefore read the title's "recovery" as *trustworthy* recovery: the deterministic core recovers, and the agentic layer is what makes the recovered data safe to stand behind.

### Adversarial robustness

The adversarial suite (Evaluation design) tests whether the system refuses to answer when it should. On its 20 fabricated incidents, **all 20 escalated; none completed; zero fabricated names appeared in any extracted field.** Seventeen escalated at search, two at the relevance judge, and one as insufficient sources. Because seventeen were stopped at search before any extraction ran, the agentic faithfulness defenses were exercised on only the two records that retrieved real articles: the deterministic retrieval and validation front-end does much of the safety work, with the agentic layer (the relevance judge, then conflict detection, then human review) as the deeper net that caught every case in this small but deliberately hard probe.

### Cost and latency

End to end, a record costs roughly \$0.20 (about \$0.16 of LLM calls plus \$0.04 of search). Latency is bimodal: records with no coverage escalate at search within seconds, while completions, dominated by the per-article LLM extraction plus the agentic layer, average tens of seconds (about 50s for civilians and 90s for officers in the N=100 runs, including eval-harness overhead). Escalated records that never reach extraction cost only search. Across all 1,956 records that is on the order of \$400, set against the hundreds of volunteer-hours the manual workflow would otherwise take (fifteen to thirty minutes each).

## Discussion

### Design principles and tradeoffs

Four principles shaped these decisions, each a stance on a tradeoff.

#### Deterministic-first

An LLM is invited only where a rule provably cannot answer the question, because every place a model is allowed to choose is a place it can choose wrong [@anthropic2024agents]. For these checks a rule provably cannot: the discriminator is genuinely semantic, and the ways an article can be the *wrong* one are open-ended (a coincidental same-city event, a same-named person, a correctly-dated article that recounts an earlier case, a list that never names the victim). One cannot enumerate them in advance, and a rule set that tried would grow brittle with every new failure, whereas a judge that answers the general question (is this the same event?) degrades more gracefully. Retrieval grounds that judgment in the source articles [@lewis2020rag] instead of parametric memory, and the same foundation-model capability that makes extraction tempting [@narayan2022wrangle] is what makes the judges viable.

#### Authority calibrated to precision

A judgment we can make sharply (do these articles even describe this incident?) is allowed to block, whereas an inherently fuzzy judgment (why do two sources disagree?) may only advise. This is why we use an LLM as the judge of a narrow, checkable question [@zheng2023judge] rather than as an open-ended author of records.

#### Faithfulness over coverage

A false keep is worse than a false veto: in an accountability database a wrong value is more damaging than a blank, so under uncertainty the system escalates rather than asserts. The visible completion drops mark that restraint, the system declining to claim what it cannot support; this design treats unfaithfulness [@maynez2020faithfulness; @ji2023hallucination; @huang2025hallucination] as the primary risk.

#### Distrust of the headline metric

Completion rate is a proxy, and the specification-gaming literature shows a proxy gets optimized at the true objective's expense once the two diverge [@amodei2016concrete; @skalse2022reward]. The multi-objective gate exists precisely so a completion gain cannot launder a correctness loss, and it is what let us deliberately ship a change that *lowered* completion because the lost completions were wrong and the hard guards held.

### Rejected designs

The earn-it protocol rejected more agentic ideas than it accepted, and the pattern of rejections ({ref}`tbl:process`) is as much the contribution as the components that shipped. Deciding *not* to build is itself a design act with its own discipline [@barocas2020whennot]; our rejections are a metric-driven instance of that stance at the component level, each a deliberate non-deployment backed by an offline test rather than a silent omission. A reasoning-and-acting extraction loop [@yao2023react] was gated out because a deterministic prompt closed the gap, leaving only run-to-run model variance, with no reasoning deficit for an agent to address. An LLM that would *type* race disagreements failed its earn-it test because every disagreement it found was an unfaithful news value the verifier had already nulled. An LLM conflict *reconciler* was deferred after a zero-cost sizing showed that most conflicts are coverage gaps a deterministic resolver already handles, with only a small remainder being genuine reasoning problems. Self-consistency (k-pass voting) was declined on cost, since the variance it would damp had already been absorbed by the prompt fix and its cost scales with k.

### Transferability and its limits

We demonstrated this discipline once, on one nonprofit's data, so we offer it as a *proposed* methodology rather than an established one. Its components rest on three domain-general preconditions rather than on TJI specifics: a held-out signal exists to score against; a hard-veto safety metric exists (here, fabricated-incident hallucinations) that no other gain may override; and decisions can be ranked by stakes so authority can be calibrated to them. Where those hold (do everything deterministic first, reserve agentic judgment for genuinely semantic questions, calibrate each judge's authority to its precision, and gate every change through a multi-objective evaluation that cannot be satisfied by trading correctness for completion), the pattern should carry. That the same pipeline serves two datasets with different schemas, switched by a dataset type and dataset-aware prompts, is modest evidence the structure is not overfit to one table. What would *not* port unchanged is the search-and-validation front-end, which is tuned to news coverage of Texas shootings. The gate, in particular, is a small pure function over saved reports, which makes the whole loop cheap to re-run, a property we found essential on a nonprofit budget.

### Responsible AI

The system is built for a sensitive domain, and several commitments follow from that. It keeps a human in the loop, never overwrites official government data, prefers faithfulness over coverage, and refuses to infer race from proxies. It also makes every suggestion auditable: each extracted value is written with its confidence label and the source URLs (and, in the escalation reports, the verbatim conflicting values) a reviewer needs to check it against the original coverage.

Equity is monitored as a first-class concern ({ref}`tbl:fairness`). Per-race completion varies: on civilians the Black cohort completes less often (64%) than the Hispanic cohort (83%), while extraction accuracy is comparable across groups (70–80%); on officers, completion is uniformly high (85–100%) because that coverage is dense. The civilian gap tracks a *temporal* coverage bias (the Black cohort skews toward earlier years, which have fewer surviving articles online) instead of a difference in how the pipeline treats groups. We surface it as a *non-gating* diagnostic, visible to a reviewer but too underpowered at these group sizes to act on automatically.

```{list-table} Per-race completion and mean exact accuracy, with per-group record counts in parentheses. One officer record of unknown race (0% completion) is omitted. Counts are small, so these are diagnostic only; the gate does not act on them.
:label: tbl:fairness
:header-rows: 1

* - Group
  - Civ. completion
  - Civ. accuracy
  - Off. completion
  - Off. accuracy
* - Black
  - 64% (25)
  - 77%
  - 85% (20)
  - 83%
* - Hispanic
  - 83% (36)
  - 74%
  - 93% (42)
  - 68%
* - White
  - 62% (34)
  - 80%
  - 97% (30)
  - 74%
* - Other
  - 60% (5)
  - 70%
  - 100% (7)
  - 62%
```

### Limitations

Retrieval recall, more than reasoning, sets the true ceiling on this task: 22 of 30 civilian escalations are simply records for which no relevant article was found, and no agent can extract what was never retrieved. The pattern is temporal: civilian completion peaks for 2019–2021 incidents (90%) and falls for both the most recent 2022–2024 (56%) and the oldest 2014–2016 (66%) cohorts, tracking how thoroughly news coverage is indexed rather than anything the model does (officer completion stays high, 82–100%, across cohorts because that coverage is denser). The fairness reading above is itself limited statistically: the per-group cells are small (as few as five records), so the equity gaps in {ref}`tbl:fairness` are directional at best, and although the evidence points to coverage availability over time, we cannot fully exclude pipeline bias as a contributor at this sample size. More broadly, the evaluation covers about 6% of the full dataset; LLM run-to-run variance remains a source of noise we damp but do not eliminate; ground-truth fields have their own gaps; and the data is single-state. These bound the strength of the per-field and fairness claims.

### Future work

The natural next step is a *self-healing* loop in which agentic effort moves to build time, proposing prompt or rule changes that the same multi-objective gate accepts or rejects automatically, keeping humans as approvers rather than authors. A nearer-term, deterministic fix targets the outcome-only entity-confusion errors noted above: a corroboration guard requiring at least two independently supported fields before committing a generic outcome. Beyond accuracy, the path to impact is operational: batch-processing the full archive in stakes-ranked order, and a review interface that lets a TJI volunteer accept, edit, or reject each suggestion against its cited sources. Larger research bets include prompt-optimization frameworks, knowledge-graph entity resolution across incidents, and a three-way majority vote on the binary judges to damp run-to-run variance where it is cheapest.

## Code and data availability

The pipeline, the evaluation harness, and the multi-objective gate are open source at `github.com/hongsupshin/police-data-intelligence` [code archive DOI: *to be deposited on Zenodo*]. The underlying datasets are published by the Texas Justice Initiative [@tji2020ois] [dataset citation/DOI: *to be confirmed*]. The accept/reject *decision* in this paper is a pure function of two saved holdout reports, so it can be recomputed from those reports at no cost; regenerating the reports themselves (`python -m src.eval.run_eval <dataset> --limit 100 --stratified`) requires inference and, because of run-to-run model variance, is not bit-for-bit reproducible.

## Conclusions

Pointing an autonomous agent at an accountability database is the obvious move and the wrong one. We built an enrichment pipeline that is mostly a deterministic workflow, with three small LLM judges admitted only after each earned its place through offline evaluation, and with authority (block, null, advise) calibrated to how much we could trust each judgment. The deterministic core does the recovering; the judges make the recovered data trustworthy, escalating to a human rather than inventing. On held-out samples the pipeline completes 92% of officer and 70% of civilian records and fabricated nothing on a deliberately adversarial twenty-incident probe. The transferable lesson for scientific-Python practitioners building with LLMs is that the engineering is in the restraint: a reward-hacking-resistant evaluation gate and an earn-it bar are what let agentic components into a high-stakes pipeline without letting wrong answers in with them.

## Acknowledgments

The author thanks the Texas Justice Initiative for collecting and publishing the data that motivates this work, and the SciPy reviewers for their feedback.

*Generative AI disclosure*: Portions of this work were assisted by generative AI tools (Anthropic's Claude, including Claude Code), used to refine and edit prose and to suggest code. All outputs were reviewed, verified, and revised by the author, who takes full responsibility for the accuracy and integrity of the final content.
