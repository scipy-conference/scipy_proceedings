# THE SILMARIL STRIKES AGAIN: building better AI systems with deterministic reasoning capabilities using Ontologies

---

# Abstract   
    
Large language models now power many agentic systems where common interaction patterns include:      
* plan first, then execute [@wang2023plansolve]   
* reason and act in turns, as in ReAct [@yao2023react]   
* reflect on feedback and try again, as in Reflexion [@shinn2023reflexion]    
* call external tools and APIs [@schick2023toolformer]   
* ...and more    
       
These patterns make language models more useful as agents.   
But the agent is still largely driven by model-generated state, reasoning, and decisions.    
      
That creates four pressures:    
* **Grounding:** the system needs stable identities for entities and typed relations between them.   
* **Multi-hop reasoning:** the system may need conclusions that follow through several relations or rules.    
* **Coordination:** several agents or tools need a shared meaning for entities, classes, properties, and constraints.    
* **Verification and audit:** important decisions need a result that can be checked, replayed, and linked to the facts and rules that produced it.      
    
These are areas where language-model generation alone does not provide formal guarantees.   
Entity tracking varies across models and task complexity [@kim2023entitytracking].    
Multi-hop reasoning can fail or follow plausible but incorrect paths [@yang2024latentmultihop; @bhuiya2024multihop].  
Generated chain-of-thought is not necessarily a faithful account of how an answer was produced [@lanham2023faithfulness].   
Model responses can also move toward a user's stated views in tested settings [@sharma2023sycophancy].    
    
Ontologies and reasoning engines provide a different layer. Ontologies represent selected entities, classes, properties, and axioms under explicit semantics [@w3c2012owl2overview]. Reasoners can then derive conclusions, answer queries, check consistency, test constraints, and record justifications under defined procedures.    
    
This paper starts with the Closed World Machine (CWM) as a historical Python reasoner. It then explains forward and backward reasoning, RETE, truth maintenance, description-logic reasoning, stratified negation, resolution, Satisfiability Modulo Theories, and the chase.  
    
The paper then shows how these methods can be composed inside a **propose-check-repair** loop for agentic systems:
* the language model **proposes**
* one or more reasoners **check**
* failed checks return structured evidence
* the model **repairs** the proposal
* accepted results can move to a separate action policy    
    
The language model remains the flexible generative component. The ontology and reasoning layer provides formal checks that language generation alone does not guarantee.    
    
The paper finally discusses columnar, distributed, and GPU execution as possible ways to scale selected reasoning workloads. It does not present a completed production reasoner or new performance results.

---

# Introduction

Large language models are increasingly used as the control layer for agentic systems.

Several interaction patterns are already common:

* **Plan and execute:** the model first breaks a task into steps and then carries them out [@wang2023plansolve].
* **ReAct:** the model alternates between generated reasoning and actions against tools or an external environment [@yao2023react].
* **Reflection:** the model uses feedback from an earlier attempt to produce a revised attempt [@shinn2023reflexion].
* **Tool use:** the model decides when and how to call external functions, APIs, search systems, calculators, or other tools [@schick2023toolformer].

These patterns make language models more useful as agents.

But they do not by themselves answer another question:

* **Which component decides whether an agent's proposed fact, answer, plan, or action satisfies a formal rule or constraint?**

A model can critique another model response.
It can also critique its own earlier response.

That is useful, but the check is still produced by a language model leaving the whole system indeterministic, which in critical situations is flatly unacceptable.

For some tasks, a separate formal checking layer is useful and can add a measure of determinism.

## The argument in a nutshell
1. Agent systems already use planning, reflection, self-critique, and external tools.
2. Those patterns do not necessarily move the acceptance decision out of the generative model.
3. The paper proposes a reasoner-mediated propose-check-repair loop.
4. The LLM proposes a fact, answer, plan, argument, or action.
5. The application grounds that proposal in an ontology-backed formal state.
6. A reasoning engine is called as a tool.
7. The reasoner returns a formal result and whatever evidence that method supports.
8. A failed check goes back to the LLM for repair.
9. A passed check can go to a separate action policy.
10. The model can remain stochastic, while selected acceptance checks can be deterministic and replayable when their formal inputs, rules, solver version, and configuration are fixed.


## Examples used in the paper

We construct a very simple example that can be used to show the mechanics behind each reasoning engine we discuss in this paper.

```python
Fido type Terrier
Fido age 12
Terrier subClassOf Dog
Dog subClassOf Mammal
````

To show that the reasoning mechanics generalize well to any arbitrary domain, we also use 2 other domains - Films and Music:

Films:
```python
Amelie type Film
Film subClassOf CreativeWork
```
Music:
```python
I Choose You song_by Sara Bareilles
song_by domain Song
song_by range Person
Song subClassOf CreativeWork
```

The examples focus on readability and clear communication.
Once the concepts are well understood, the same approach and structure can grow to represent much larger and more complex domains.
* A finance or banking ontology can contain customers, accounts, ownership, transactions, eligibility rules, and permissions.
* A reinsurance ontology can contain contracts, layers, counterparties, risks, and exposures.
* Supply-chain and energy systems can contain suppliers, assets, locations, capacities, obligations, and constraints.
As one can see, the reasoning algorithms do not depend on the Fido example, the small data only makes each operation easier to see.

## Four pressures on agentic systems

Four pressures motivate a separate ontology and reasoning layer.

### Grounding

An agent needs a stable way to refer to things.

For example:

```python
Fido
Amelie
I Choose You
Sara Bareilles
```

should refer to identified entities, not only to similar strings in a prompt.

An ontology can give entities explicit identifiers and define classes and properties with formal meaning.
OWL 2, for example, defines classes, properties, individuals, and ontology semantics [@w3c2012owl2overview].

This matters because entity tracking is not uniformly reliable in language models.
We have observed that entity-tracking ability varies across models and becomes harder on more complex settings [@kim2023entitytracking].

The issue is not that an LLM cannot refer to entities.
The issue is that *stable* entity tracking is an empirical model capability rather than a formal property of the application.

### Multi-hop reasoning

Agentic tasks often require conclusions that follow through several steps.

For example:

```python
Fido type Terrier
Terrier subClassOf Dog
Dog subClassOf Mammal
```

A reasoner can derive:

```python
Fido type Dog
Fido type Mammal
```

under the stated rules.

Language models can perform multi-hop reasoning, but the capability is not uniform, that latent multi-hop reasoning varies strongly by relation and prompt type [@yang2024latentmultihop], and that that plausible but incorrect reasoning paths can substantially reduce multi-hop performance [@bhuiya2024multihop].

It is obvious to see that the situation may tend to generate even more pronounced discrepancies, greatly impacting the reliability of the overall system when:
* using multiple models (for e.g. a "locally deployed" LLM that is finetuned on enterprise vocabulary in conjunction with a frontier model) or
* in complex system reasoning across multiple OLTP / OLAP systems with 10s of tables and 100s of columns.

The distinction is not:

```python
LLMs cannot reason.
```

It is:

```python
An LLM answer does not by itself establish that every required inference step *followed* from a stated formal system.
```


### Coordination

An agent may interact with:

* other agents
* databases
* search systems
* business rules
* APIs
* planning tools, schedulers, triggers, tools,
* reasoning engines
* ...and other systems...

Those components need a shared meaning for the data they exchange.

For e.g., the following nouns (or noun-like vocabulary indicating objects, things, relationships, capabilities etc.):

```python
- Dog
- Mammal
- CreativeWork
- song_by
- hasOwner
```

should mean the same thing to each component no matter which LLM / Agent / language model is engaged.

RDF provides a graph model for statements.
OWL 2 adds formally defined classes, properties, individuals, and ontology axioms [@w3c2014rdf11; @w3c2012owl2overview].

The ontology therefore acts as a shared *formal* vocabulary.

It does not solve every coordination problem.
It provides a common representation on which reasoning and validation can operate.

### Verification and audit

An agent may generate a very convincing sounding explanation for a decision.

That explanation is not automatically (cannot be assumed to inherently be) the derivation that produced the decision.

Work on chain-of-thought faithfulness shows that generated reasoning text can differ from the process that determines the final answer [@lanham2023faithfulness].

For tasks that require verification, the system may instead need:

* which facts were used
* which rules were applied
* which constraints passed or failed
* which conclusion followed
* which version of the formal state was checked

Historical systems such as The Closed World Machine (CWM) already treated derivation information as part of the reasoning process and supported proof-oriented output [@bernerslee2009cwm; @cwmCheckSource].

This is different from asking the stochastic model to explain itself after the fact (lack of deterministic outcomes or reasons for the outcomes).

### Another nuance: model judgment can shift

There can be another reason to separate generation from formal checking.

Language models respond to *conversational context*.

That is often useful.
But it also means that model judgment can move with the conversation - how we frame things, what context is specified, what goes missing, what assumptions are provided, what fail silently etc. etc. - the vagaries of regular human communication - we find sycophantic behavior in several tested AI assistants, including cases where responses align with a user's stated beliefs over more truthful answers [@sharma2023sycophancy].

This does not imply that every model always agrees with the user.

It shows that user framing can affect the same kind of component that is being asked to generate the answer.

For selected checks, an agent can instead call a formal reasoner, providing a greatly improved deterministic quality to the decisions and outcomes.
Its result depends on the formal inputs, rules, semantics, solver version, and configuration rather than on conversational tone.

## Division of work

Thinking about using Ontologies, formal reasoning engines along with LLM based Agentic AI systems, the author proposes that we resolve the system into four roles.

* **The language model interprets and proposes.**
  * It can extract candidate facts, understand language, draft answers, generate plans, and propose actions.

* **The ontology represents the formal state.**
  * It provides identifiers, classes, properties, relations, and axioms.

* **Reasoning engines derive and check.**
  * Different engines answer different formal questions.

* **The action layer decides what may execute.**
  * Passing a reasoning check does not automatically authorize an external action.

This division does not make the whole agent deterministic 100% of the time:
* The model can still be stochastic.
* Inputs can still be wrong.
* The ontology can be incomplete.
* Rules can still encode the wrong policy.

The narrower claim is that selected parts of the decision process can be moved into a formal layer whose behavior is defined more precisely.

### From reasoning engines to an agent loop

To ensure a wider audience can understand the concepts, we will quickly explore the reasoning engines themselves.

We first examine The Closed World Machine (CWM) as a historical Python reasoner.
CWM combines RDF and Notation3 data, forward rule application, built-ins, Web access, and derivation records [@bernerslee2009cwm; @swapRepository].

We then examine a focused set of reasoning methods:

* forward chaining
* backward reasoning
* RETE
* truth maintenance
* description-logic reasoning
* stratified negation
* resolution
* Satisfiability Modulo Theories
* the chase

These methods are not interchangeable.

- Some derive consequences.
- Some answer a selected query.
- Some retain partial matches.
- Some withdraw conclusions when their support disappears.
- Some classify ontology entities or check consistency.
- Some test logical or arithmetic constraints.

Then we explore how we can compose these methods as "tools" inside a carefully crafted interaction pattern that *the author here proposes*:

```python
propose -> check -> repair
```

* The model proposes a candidate.
* The application translates the relevant part of that proposal into the ontology-backed formal state.
* One or more reasoners perform the required checks.
	- If a check fails, its result becomes structured input for another proposal.
* If the required checks pass, the result can move to a separate action policy.
* Related work already demonstrates useful forms of language-model plus solver separation.
* Faithful Chain-of-Thought separates language translation from deterministic symbolic execution, for example [@lyu2023faithfulcot].

The contribution here is not the claim that calling a solver from an LLM is new, though the naming of the pattern may be.

The more valuable idea is the composition of several kinds of reasoning engines around a shared ontology-backed state, and the use of those engines as different checking tools inside one propose-check-repair loop.

For the rest of the paper, we'll consider the following:

* We start from The Closed World Machine (CWM) as a historical example of rule execution, built-ins, Web access, and derivation records in Python.
* We then build simple explanations for a focused set of reasoning engines and the formal questions each one can answer.
* We show how those engines can compose inside the **propose-check-repair** pattern for agentic systems.
* To address high volume and high variety scenarios for large enterprises - we discusses columnar, distributed, and GPU processing as *possible* ways to scale selected reasoning workloads.

This lets us use the language model for its strongest features: **interpretation and generation**.

The author's solution moves selected questions about entailment, consistency, constraints, and justification into components designed to solve for the answers *formally*.

---

# A note on the length of the paper
Because of the SciPy recommends paper [length limit of 6000 words](https://github.com/scipy-conference/scipy_proceedings#general-information-and-guidelines-for-authors), this paper presents only the reasoning methods needed to establish the core argument.

A longer version (8000+ words) in the Silmaril repository includes the additional walkthroughs for description-logic reasoning, stratified negation, resolution, Satisfiability Modulo Theories (SMT), and the chase, together with the companion Python implementations and executable examples [@shauryasilmarilRepository2026].

For the 8K words version, refer to: [The Silmaril strikes again full version v01.md](https://github.com/shauryashaurya/The-Silmaril/blob/main/scipy-2026/The%20Silmaril%20strikes%20again%20full%20version%20v01.md)

---


# Ontologies and reasoners in plain terms

An **ontology** describes the **what** - that is:
1. the *types of things* in a domain
2. the *relations* between them, and
3. formal statements that *constrain or connect* those types and relations.

In the Resource Description Framework (RDF), a basic assertion is a subject-predicate-object triple, and a set of triples forms a graph [@w3c2014rdf11].
An RDF dataset can contain a default graph and named graphs.
Implementations often store a triple together with a graph or context identifier as a quad.
Notation3 (N3) extends RDF with quoted formulae, variables, implication, and built-in predicates [@bernerslee2011n3].

A **reasoning system** can be described through 5 basic parts:

1. **Facts.** These are statements supplied to the system, such as `Fido is a Terrier`.
2. **Vocabulary and ontology.** These define classes and relations such as `Terrier`, `Dog`, and `hasOwner`.
3. **Axioms and rules.** These define allowed conclusions, such as `Every Terrier is a Dog`.
4. **Evaluation procedure.** This determines how the system searches for conclusions.
5. **Results and evidence.** The output may be a derived fact, answer, proof, model, inconsistency, or failed check.

A derivation record states which supplied facts and rules support a conclusion.
It does not establish that the supplied facts are true in the external world.

## Reasoning up and down an ontology

Suppose the ontology states:

- `Terrier` is a subclass of `Dog`.
- `SeniorDog` is a subclass of `Dog` where age is greater than 10
- `Dog` is a subclass of `Mammal`.
- `Fido` is a `Terrier`.


A reasoner can classify Fido as a Dog and a Mammal.
This lets the logic move from a more *specific* class to *generic* classes that contain the instance `Fido`.
The reverse inference is not valid: knowing only that `Fido is a Mammal` does not establish that `Fido is a Terrier` (going from *generic* to *specific* or *derived* needs more qualifying criteria).

A query-directed procedure can start with `Is Fido a Mammal?` and search for support through `Dog` and `Terrier`.
Notice how this is different from deriving every available consequence in advance, we can reason up and down an ontology at runtime.
It should be obvious how this approach is valuable in a complex enterprise where, facts may not be available (as arbitrary transactions are being created or retrieved from archives) at all times.

## Ontologies and object-oriented programming

Ontologies and object-oriented programming (OOP) use some of the same words, including class, instance, property, and inheritance.
The meanings are different.
Python classes bundle data and functionality, and inheritance lets derived classes reuse or override behavior from base classes [@pythonDocsClasses].
Ontology classes describe categories of entities under formal semantics.

For example:
* In OOP (Python) we can create `fido = Terrier()` and inherit methods from a `Dog` base class.
* Setting `fido.age = 12` does not by itself reclassify the object as a `SeniorDog`.
however,
* An ontology reasoner can infer such a classification if `SeniorDog` is formally defined by conditions that Fido satisfies.

So, we see that OOP can be used to implement an ontology tool or reasoner, but OOP inheritance does not itself provide description-logic inference.

# CWM as a historical starting point

The Closed World Machine (CWM) is a general-purpose Semantic Web data processor and forward-chaining reasoner written in Python [@bernerslee2009cwm; @bernersleeReasonerWeb].

Its name does not mean that every CWM operation uses a closed-world assumption.
CWM reads RDF or Notation3 (N3), applies rules, invokes built-in functions, can retrieve Web resources, and can produce records that explain derived results [@bernersleeReasonerWeb; @bernerslee2009cwm; @cwmCommandSource].

CWM is useful here because one Python tool combines:

- formal statements
- rule execution
- external functions
- Web retrieval
- derivation records

These are also useful parts of the agent pattern discussed later in this paper: a formal state, executable rules, calls to external computation, and evidence about derived results.

The paper does not claim that CWM is the first rule engine or the first solver-augmented architecture.

## Rule execution and matching

A simplified CWM workflow is:

- Load RDF or N3 into a working formula.
- Read rules whose antecedents and consequents are N3 formulae.
- Match the antecedents against the working data.
- Add instantiated consequents.
- With `--think`, repeat rule application until no further rule matches, unless a rule or built-in continues to create new results [@cwmCommandSource].

The CWM systems paper describes recursive template matching with two important optimizations:

- It analyzes dependencies between rules so that rules can sometimes be evaluated in a useful order.
- Within a rule body, it tries statement patterns in an order based on the size of the relevant index, preferring smaller candidate sets [@bernersleeReasonerWeb; @bernerslee2009cwm].

### Indexed statement matching

CWM works by maintaining several indexes [@bernersleeReasonerWeb] - a walkthrough of these demonstrates the mechanics of reasoning and is helpful in building our conceptual model of a reasoner, also gives us intuition on how it would be helpful in a Agentic AI based stack:

- **Subject index:** finds statements in which a given term appears as the subject.
- **Predicate index:** finds statements in which a given term appears as the predicate.
- **Object index:** finds statements in which a given term appears as the object.

Consider the following ontology-backed state:

```python
Fido type Terrier
Fido age 12
Terrier subClassOf Dog
Dog subClassOf Mammal
````

For the triple-like statement:

```python
Fido type Terrier
```

CWM makes the statement reachable through several indexes:

```python
subject index:
Fido -> Fido type Terrier

predicate index:
type -> Fido type Terrier

object index:
Terrier -> Fido type Terrier
```

So the mechanics of finding facts should be trivial now:
Suppose a rule needs to match:

```python
Fido type ?class
```

The subject index can first retrieve statements about `Fido`:

```python
Fido type Terrier
Fido age 12
```

The matcher can then test the predicate and retain:

```python
Fido type Terrier
```

This binds:

```python
?class = Terrier
```

Using `?` to mean "any value", the seven useful subject-predicate-object patterns are naturally read as:

- **S P O:** subject, predicate, and object are all known.
- **S P ?:** subject and predicate are known; object is unknown.
- **S ? O:** subject and object are known; predicate is unknown.
- **? P O:** predicate and object are known; subject is unknown.
- **S ? ?:** only the subject is known.
- **? P ?:** only the predicate is known.
- **? ? O:** only the object is known.

So when we think about building a more complex implementation that can respond efficiently to different types of queries - it is plain to see that building additional indexes (one for every '?') would allow us to answer more complicated queries faster.

The general mechanism that we discover is:

```python
known term
	-> indexed lookup
	-> smaller candidate set
	-> additional variable bindings
	-> evaluate the remaining rule conditions
	-> derive a new statement
```

CWM can emit proof information, and `why.py` contains structures used to represent reasons for statements [@cwmWhySource].
CWM also includes `check.py`, described as a simple proof checker [@cwmCheckSource].
These source-supported features make CWM a useful historical example of rule execution with inspectable derivations.

**NOTE**: The systems paper reports that later versions added more indexes to cover almost every subject-predicate-object wildcard pattern, trading faster matching for additional indexing cost [@bernersleeReasonerWeb]. Due to the word-limit of the paper (6K words or fewer), we are not providing a literal analysis of the other indices that actually show up in the CWM code or how these compare to RETE. CWM and RETE are not the same but CWM ported some of the efficiencies of RETE engine used in Pychinko into its own source-code [@bernersleeReasonerWeb].

# Reasoning engines and what they do

We used CWM to mainly illustrate forward rule processing, but an Agent (in any arbitrarily complex Agentic AI stack) may need other forms of reasoning.
The CWM mechanism: keep a formal state, find matching statements, apply rules, and add derived statements is one kind of reasoning.

An agentic system may need other kinds of formal checks.
There's no one "best" reasoner nor are they "the same kind of thing" - instead we should ask "What formal question does the agent need answered?" and choose the reasoner accordingly.

- Forward and backward reasoning describe directions for deriving or searching for conclusions. [@bernerslee2009cwm; @nenov2015rdfox; @chen1995slg]
- RETE is an incremental rule-matching algorithm. [@forgy1982rete]
- Truth and materialization maintenance track what should remain true when inputs change. [@motik2015bf]
- Description logic is a family of formal languages and semantics for ontologies. [@baader2010dlhandbook]
- Stratified negation gives a controlled meaning to some forms of "not known". [@apt1988stratified]
- Resolution is a proof procedure over logical clauses. [@robinson1965resolution]
- Satisfiability Modulo Theories (SMT) checks logical constraints together with theories such as arithmetic. [@demoura2008z3]
- The chase applies database dependencies, including rules that require some entity to exist. [@maier1979chase; @bellomarini2018vadalog]

We group all these reasoners here in the paper because each can provide a different formal service inside an agent.

## Revisiting and updating the ontology-backed states for discussing all the reasoners

To keep the examples easy to follow, let's review our pup Fido's ontology, this is what we will use for each reasoner.
We also introduced a new relationship `disjointWith` to show two completely separate (mutually exclusive) classes.


```python
Fido type Terrier
Fido age 12

Terrier subClassOf Dog
Dog subClassOf Mammal
Mammal disjointWith Reptile
Reptile is Dangerous
```

For rule-oriented examples, the two subclass statements can be read as:

```python
Terrier(x) -> Dog(x)
Dog(x) -> Mammal(x)
```

The examples will also try to reason about Fido's age, if he is dangerous (reptiles are dangerous) etc. this looks something like:

```python
Dog(x) AND age(x) > 10 -> health_check(x)

Dog(x) AND NOT dangerous(x) -> may_enter(x)

```

The age number was chosen arbitrarily to demonstrate how a rule combines class membership with a numeric condition.
Please do not consider it as veterinary guidance.

The SMT example uses a separate small set of numeric constraints:

```python
meals is an integer
2 <= meals <= 4
meal_cost = 3
total_cost = meals * meal_cost
total_cost <= 9
```

Two small examples from other domains are present only to show that the reasoning mechanisms generalize to arbitrary domains:

```python
Amelie type Film
Film subClassOf CreativeWork

I Choose You song_by Sara Bareilles
song_by domain Song
song_by range Person
Song subClassOf CreativeWork
```

We return to those examples briefly after the Fido walkthroughs.

## Forward chaining

**Question:** Given the facts and rules we already have, what else follows?

Forward chaining starts with the current facts.
It finds rules whose conditions are satisfied and adds their conclusions.
New conclusions can then enable more rules.

Start with:

```python
Fido type Terrier
```

and the class relationship:

```python
Terrier subClassOf Dog
```

The first rule application gives:

```python
Fido type Dog
```

Now use:

```python
Dog subClassOf Mammal
```

to derive:

```python
Fido type Mammal
```

The reasoning path is therefore:

```python
Fido type Terrier
-> Fido type Dog
-> Fido type Mammal
```

The engine can continue applying rules until another pass produces no new statements.
That final state is often called a fixpoint.

The important point is the direction of work:

```python
known facts
-> matching rules
-> new facts
```

Forward chaining is useful when the agent expects many later questions to reuse the same derived facts.
Instead of proving `Fido type Mammal` again for every query, the derived statement can already be present in the formal state.

CWM is one example of this style of rule execution [@bernerslee2009cwm; @bernersleeReasonerWeb].

## Backward or query-directed reasoning

**Question:** Can this particular conclusion be supported?

Backward reasoning starts with the question rather than deriving every available consequence.

Suppose the agent asks:

```python
Is Fido a Mammal?
```

The initial goal is:

```python
Fido type Mammal
```

The reasoner asks what could establish that goal.

From:

```python
Dog subClassOf Mammal
```

it obtains a smaller subgoal:

```python
Fido type Dog
```

To establish that, it can use:

```python
Terrier subClassOf Dog
```

which creates another subgoal:

```python
Fido type Terrier
```

That statement is already asserted.

The support chain is:

```python
goal: Fido type Mammal

needs:
Fido type Dog

needs:
Fido type Terrier

found:
Fido type Terrier
```

The answer is therefore supported.

The important direction is:

```python
goal
-> required support
-> smaller subgoals
-> asserted fact
```

This is useful when the agent needs one focused answer and does not need every consequence of the knowledge base.

Practical backward reasoners can remember answers to repeated subgoals so they do not solve the same subproblem repeatedly.
Tabling and SLG evaluation are examples of that implementation technique [@chen1995slg].



## The same mechanisms are not specific to Fido

The Fido example is deliberately repetitive because it lets us compare reasoning methods without changing the domain at every step.

The same formal mechanisms can operate over other vocabularies.
For example, the class statement `Amelie type Film` together with `Film subClassOf CreativeWork` can support `Amelie type CreativeWork`.

Likewise, in a rule or ontology implementation that gives the declared `domain`, `range`, and `subClassOf` statements their intended semantics, the music facts can support:

```python
I Choose You type Song
I Choose You type CreativeWork
Sara Bareilles type Person
```

The reasoning mechanism has not become a "movie reasoner" or a "music reasoner".
Only the formal vocabulary and facts have changed.

# What each reasoner contributes

The reasoners can be separated by the question they answer:

- **Forward chaining:** What new facts follow from the current facts and rules? [@bernerslee2009cwm; @nenov2015rdfox]
- **Backward reasoning:** Can this particular goal be supported? [@chen1995slg]
- **RETE:** Which previous rule matches can be reused when facts change? [@forgy1982rete]
- **Truth or materialization maintenance:** Which derived conclusions remain supported after an update? [@motik2015bf]
- **Description-logic reasoning:** What follows from the ontology semantics, and is the ontology-backed state consistent? [@baader2010dlhandbook]
- **Stratified negation:** Can a rule depend safely on scoped absence? [@apt1988stratified]
- **Resolution:** Can a target be proved by refuting its negation? [@robinson1965resolution]
- **SMT:** Can a set of logical and theory constraints be satisfied? [@demoura2008z3]
- **Chase:** What fresh placeholders are required by existential dependencies? [@maier1979chase; @bellomarini2018vadalog]

These methods are complementary. An agent can select the formal check required by the task rather than run every reasoner on every proposal.

```text
language model proposes
-> formal state represents facts and constraints
-> the appropriate reasoner checks
-> the result supplies evidence
-> the agent accepts or repairs the proposal
```

The shared ontology gives these reasoners common entities and relations. Each contributes different formal evidence.

# Composing reasoning engines into propose-check-repair

The reasoning methods become useful when exposed as tools over a shared formal state.

The loop has six steps:

1. **Propose.** The LLM proposes a fact, answer, plan, argument, or action.
2. **Ground.** The application maps the proposal to identifiers, relations, rules, and constraints.
3. **Select.** The application chooses the reasoner that matches the question being checked.
4. **Check.** The reasoner evaluates the formal state and returns a result and available evidence.
5. **Repair.** A failed check is returned to the agent as structured feedback.
6. **Act.** A separate policy decides whether an accepted proposal may proceed.

A short Python-shaped control flow is:

```python
candidate = agent.propose(task, state)
result = reasoner.check(candidate, state)

if result.accepted:
    policy.act(candidate, result.evidence)
else:
    agent.repair(candidate, result.evidence)
```

`reasoner` need not be one universal engine. A tool interface can route different checks to different engines over the same formal state.

For example:

- classification can go to a description-logic or OWL reasoner
- a rule query can go to forward or backward evaluation
- repeated rule matching can use RETE
- a policy exception can use stratified negation
- a clause proof can go to resolution
- arithmetic or ordering constraints can go to SMT
- an existential dependency can go to the chase

Evidence also differs: a derivation, satisfying model, inconsistency, or refutation proof. The loop should preserve these differences.

## Why the ontology-backed state matters

The ontology-backed state provides:

- **Stable reference.** Checks can refer to the same identified entity.
- **Typed relations.** Candidate statements are represented as explicit predicates and types.
- **Formal semantics.** A reasoner can evaluate rules, subclass relations, constraints, or consistency under stated semantics.
- **Shared state.** Different reasoners can operate on the same entities and relations.

One reasoner's output can become input to another check if its meaning and source are preserved. For example, forward evaluation may derive a permission while SMT checks numeric limits.

## What becomes deterministic

The pattern does not make model generation deterministic. It moves selected acceptance conditions into formal procedures.

For fixed formal inputs and a fixed reasoner configuration, checks such as rule entailment, ontology consistency, SMT satisfiability, policy derivation, or support after an update can be replayed.

The result is only as good as the formalization. The model can map a request incorrectly, the ontology can omit conditions, and a reasoner can apply a wrong rule correctly.

The claim is therefore narrow: selected formal checks can be made repeatable without asking the language model to judge its own proposal.

This separation also limits the role of conversational pressure in the check itself. Once the facts, rules, and acceptance condition are fixed, conversational tone does not change the formal entailment relation. The model can still make a biased proposal or formalization before the check.

# A map from reasoning work to larger-scale execution

The loop does not require a new monolithic reasoner. Larger implementations can combine existing tools.

The ideas below are scaling directions, not results from a completed system.

## Columnar and set-oriented processing

A columnar representation can dictionary-encode terms as integer identifiers and store statement fields separately. Rule bodies can then be evaluated as joins over sets of rows. Existing Datalog and RDF systems provide relevant examples [@nenov2015rdfox; @jordan2016souffle].

Design questions include indexes, derivation storage, updates, and deletions.

Delete-Rederive can remove facts that still have another derivation and then derive them again. Backward/Forward maintenance checks for alternative support and can avoid some of that work [@motik2015bf]. The right choice depends on workload.

## Distributed processing

Relations larger than one machine can be partitioned across workers. Recursive evaluation must then handle data movement, skew, duplicate suppression, failures, and completion of each round.

RDFox and Differential Dataflow provide useful comparison points for parallel materialization and iterative computation over changing inputs [@nenov2015rdfox; @mcsherry2013differential].

Whether distribution helps depends on data size, rule shape, updates, and evidence requirements.

## GPU processing

GPU execution is most promising when reasoning work can be expressed as regular operations over many values, such as filtering, joins, frontier expansion, or sparse matrix operations.

Gunrock provides a frontier-based graph-processing model [@wang2016gunrock], while GraphBLAS expresses graph algorithms through sparse linear algebra [@kepner2016graphblas].

These approaches make different tradeoffs. Matrix methods may reduce rounds while increasing intermediate density and memory use. Frontier methods may use more rounds while touching a smaller active set.

Scaling is secondary to the thesis. The reasoner defines the formal operation; columnar, distributed, and GPU methods concern execution.

# Python example and reproducibility

Python appears in both the historical case and the companion material. CWM is implemented in Python, and its cited source exposes the command flow, store, formula representation, derivation structures, and proof checker [@swapRepository].

The Silmaril repository contains the paper material and Python examples [@shauryasilmarilRepository2026; @agarwal_2025_17297865].

The companion files are:

- `reasoner_loop.py`
- `test_reasoner_loop.py`

They use basic Python data structures rather than a reasoning library and implement small teaching versions of the methods discussed in the paper.

The examples cover:

- forward and backward reasoning
- RETE-style alpha memories and a beta join
- justification and retraction
- description-logic-style classification and disjointness
- stratified negation
- resolution
- a small DPLL(T)-style SMT example
- composition inside propose-check-repair

Each reasoner prints a trace using terms from the paper, including asserted fact, subclass axiom, entailment, justification, retraction, clause, constraint, and model.

The tests can be run with:

```python
python test_reasoner_loop.py
```

These are teaching examples, not general-purpose reasoners or benchmarks.
The files are located in the [author's main GitHub repository](https://github.com/shauryashaurya/The-Silmaril/tree/main/scipy-2026).

# Limits and next steps

The pattern has several limits:

- **Formalization error.** Language can be mapped to the wrong entity, relation, rule, or constraint.
- **Incomplete ontology.** A missing condition cannot be enforced.
- **Reasoner selection.** Different checks require different procedures and semantics.
- **Composition semantics.** Derived facts must preserve their meaning and source when passed between reasoners.
- **Update semantics.** Changing state requires a defined method for retracting unsupported conclusions.
- **Tool trust.** Reasoner software can contain defects, so version and configuration matter for replay.
- **Action policy.** Passing a formal check does not authorize a real-world action.
- **Scale.** Larger execution strategies require representative benchmarks.

These limits do not change the central composition: reasoners provide formal operations over shared ontology-backed state, and the loop decides when to use them.

# Summary

This paper starts with CWM as a historical Python example of RDF and N3 rule processing, built-ins, Web access, and derivation records.

It then explains reasoning methods that answer different formal questions: forward and backward evaluation, RETE, maintenance, description-logic reasoning, stratified negation, resolution, SMT, and the chase.

These methods are composed into a propose-check-repair loop. The LLM proposes, ontology-backed state represents entities and relations, and reasoners check formal properties. Failures return evidence for repair; accepted results can pass to action policy.

The deterministic claim remains narrow. The model and language-to-ontology mapping can still be stochastic or wrong. With fixed formal inputs and reasoner configuration, selected checks can be replayed without asking the model to evaluate its own proposal.

Columnar, distributed, and GPU methods remain possible execution directions for scaling selected reasoning workloads. The next step is to benchmark selected reasoner combinations while preserving the separation between proposal, formal state, checking, repair, and action.


# Appendix

This paper is a follow up from  a four-hour tutorial on ontology engineering presented at the 2025 edition of SciPy.


# References

* The Silmaril on GitHub: [https://github.com/shauryashaurya/The-Silmaril](https://github.com/shauryashaurya/The-Silmaril)
* Scipy 2025 presentation on YouTube: [https://www.youtube.com/watch?v=HlSqH6T-y0Q](https://www.youtube.com/watch?v=HlSqH6T-y0Q)
* TimBL’s semantic web application platform on GitHub: [https://github.com/linkeddata/swap](https://github.com/linkeddata/swap)
