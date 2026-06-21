---
title: "From Hello World to Hello LLM: A Python Developer's Survival Guide"
abstract: |
  AI tooling is moving fast, but many Python developers are unsure where to start or how today's AI patterns fit into systems they already know how to build. This talk is a practical, hands-on overview of modern AI development patterns in Python, focused on what you need to know to go from zero to hero.

  We'll walk through a real-world coding example broken into parts that illustrate the core building blocks of modern AI applications, and explain when each pattern makes sense. This example is designed in a way that doesn't require any prior machine learning experience, and attendees will leave with an understanding of how AI systems work, what problems they're good at solving, and how to maintain and observe what has been built.

  Topics we'll cover:

  - The modern AI stack in Python: LLM APIs, embeddings, tools, and agents
  - Common Python AI patterns: prompts, function calling, RAG, and simple agents
  - When to use a script vs an agent vs a service (and when not to)
  - How to get something working quickly without sacrificing reliability or safety
  - Practical guardrails: handling errors, controlling outputs, and protecting data
  - How to generally stand up common AI workflows, such as LLM-powered scripts to lightweight AI agents / MCP-style services

  Attendees will leave with a clear map of the AI landscape, working Python patterns they can reuse immediately, and the confidence to start building AI features without needing a machine learning background.
---

# What We Learned Building AI Tools with Python

Most Python developers already know how to build real systems, APIs, scripts, services, and pipelines. During a social impact AI hackathon, we applied those same engineering skills to rapidly build AI-powered tools that helped educational institutions become more comfortable using AI to improve student success outcomes.

Like many developers entering the AI space, we had to quickly navigate a growing ecosystem of frameworks, APIs, vector databases, and agent tooling while still shipping working software under tight time constraints.

Here we share the lessons learned, tooling decisions, and implementation patterns that helped us move from experimentation to working applications.

Rather than treating AI as magic, we'll break it down into familiar engineering concepts:

- validation
- retrieval
- orchestration
- structuring inputs
- structured outputs

We'll walk through:

- the core concepts behind modern AI applications
- common Python libraries and tooling
- reusable code snippets and implementation patterns
- understanding how and when to use these patterns in real systems

The biggest shift here is the greater accessibility for anyone to build. You no longer need to train models yourself as modern large language model (LLM) APIs, embedding models, and vector databases are available with minimal setup. We view AI as not replacing your application stack, but another component to be added to existing systems that are useful for ambiguity, language, reasoning, and search. Think of an LLM like a non-deterministic but powerful external API that needs structure, validation, retries, and observability.

Before we delve into these items, we'll explain the modern LLM stack.

---

## The Modern AI Stack in Python

Most AI-powered Python applications are assembled from a small set of reusable components:

- LLM APIs
  - ollama-python, litellm, transformers, vllm, llama-cpp-python
- Embedding models
  - sentence-transformers, transformers, InstructorEmbedding, FlagEmbedding (BGE models)
- Vector databases
  - faiss, chromadb, qdrant-client, weaviate-client, milvus
- Prompt orchestration
  - LangGraph, LangChain, Haystack, PydanticAI, CrewAI
- Structured output validation
  - Pydantic, Instructor, Guardrails AI, jsonschema
- Tool calling
  - PydanticAI, LangGraph, LangChain, smolagents, CrewAI

We'll show how these pieces connect in practice using code snippets and examples from our social impact hackathon project.

The key takeaway: you are still writing Python. LLMs are another system dependency your application coordinates and manages. Before we began using AI in our hackathon, we started with understanding our stakeholders needs.

---

## Key Terms

Now, briefly on key terms you will see throughout this paper. For our purposes, during the hackathon, we used the following concepts:

- **LLM:** Generates text from prompts
- **Embeddings:** Embeddings turn words or sentences into numbers that capture meaning.
  Things with similar meanings end up close together:
  - "dog" ≈ "puppy"
  - "pizza" ≈ "burger"

  That's how AI can find related ideas, not just exact word matches. In real life, you can find related ideas in `<user_prompt>` and structures within your chatbot code.
- **RAG:** Retrieve context before generating answers
- **Agent:** A loop where the model can use tools and react to results

These concepts cover most real-world AI applications.

---

## Requirements Gathering Before Hacking

Now, before coding in the AI hackathon, we spent time in the beginning determining what our stakeholders' thought a good result was.

Requirements gathering was an important step to help us understand the goals of our institutions. This involved journey mapping.

<img width="1400" height="785" alt="user_journey_map" src="https://github.com/user-attachments/assets/c2ee6cd7-9103-4402-86ad-ac343453cf54" />


Once we documented the needs and criteria our stakeholders (educational institutions and staff) had for success, we could begin development work.

Before building, we worked with educational staff to journey-map their workflows, questions, and pain points. That process helped define:

- what kinds of questions the system should answer
- what data it was allowed to access
- what topics were out of scope
- how responses should be phrased

With the requirements distilled from our clients with this method of planning, we were able to identify LLM patterns that for our respective institution's use case. With that in mind, we'll explore patterns of LLM usage with hands on examples.

---

## Core Pattern 1: Functional Calling & Controlled Output

*Separating LLM and analysis workflows*

Let's start off acknowledging that not every problem requires a LLM. This first pattern will illustrate how to organize distinct flows into logic that a LLM could be beneficial for and more deterministic pieces of logic that could be solved with fixed pipelines or analysis.

In Audrey's hackathon project, she had to consider the various goals of the different institutions on her team. Through requirements gathering techniques like journey mapping, she was able to develop a solution that would satisfy all goals -- build a predictive analytics dashboard to help institutions better understand their student's academic trajectories and address issues early on. The predictive analytics dashboard was thus created using a host of ML models, and an associated interactive chatbot was integrated for institutions to easily gather those analysis results. The design behind this dashboard was focused on separating language understanding from analysis workflows. Instead of allowing unrestricted responses, we structured how the LLM could request information and how results from predictive analyses would be returned. This created a safer interface between the chatbot and the underlying datasets. Instead of the LLM generating predictions directly, it acted as an interface layer, deciding which analytics function to call and how to interpret the result.

This included outputs like:

- Retention prediction
  - risk scores per student
  - probability of dropping out
  - attendance trend forecasts
- Credential type prediction
- Gateway course success prediction
- GPA prediction

For example, a flow could look as follows:

```
Question → LLM → One Function → One Model → Result → Explanation
```

```python
from openai import OpenAI

client = OpenAI()

# Predictive model wrapped as an approved function
def get_retention_risk(school_id):

    students = feature_store.get_students(school_id)

    risk_scores = retention_model.predict_proba(students)

    return {
        "school_id": school_id,
        "avg_retention_risk": float(risk_scores.mean())
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_retention_risk",
            "description": "Retrieve student retention risk metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "school_id": {"type": "string"}
                },
                "required": ["school_id"]
            }
        }
    }
]

user_question = (
    "Which students are most at risk of not returning next semester?"
)

# Step 1: LLM determines what analytics are needed
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "Never generate predictions yourself. "
                "Use available analytics functions and explain results."
            )
        },
        {"role": "user", "content": user_question}
    ],
    tools=tools
)

# Step 2: Backend executes predictive model
analytics_result = get_retention_risk(
    school_id="School_123"
)

# Step 3: LLM explains results
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": user_question},
        {
            "role": "tool",
            "tool_call_id": response.choices[0].message.tool_calls[0].id,
            "content": str(analytics_result)
        }
    ]
)

print(final_response.choices[0].message.content)
```

This pattern ensured predictions are:

- reproducible
- auditable
- computed separately from the LLM
- token usage was optimized

---

## Core Pattern 2: Prompting with Language and Constraints

As we delved into prompting, we philosophically defined what the model should and should not be doing. To illustrate this, we'll begin by describing Jasmine's Hackathon project. Jasmine's team built a chatbot to help higher education staff at a University understand how a pivotal dataset can expedite their ability to track student success. They were trying to integrate the dataset, but were unsure where to begin. Additionally, they had an institutional research department that bore the brunt of handling requests for data from around the institution about student success with existing data that often took days to massage into final data presentations. AI was an opportunity to expedite this process and make many of the requests into self service. Unfortunately, institutions were concerned that AI posed a risk of exposing data that shouldn't be widely available. Thus, prompting became the primary way we defined system behavior, privacy, and guardrails.

We also created a data dictionary of approved educational terms and concepts to guide the LLM toward domain-specific language and reduce hallucinations. This was important because the staff at the education institution were aware that despite datasets having overlapping subject matter, there was nuance in how the data could be used. Despite data sounding similar, it was not advised to join certain tables. And since some tables contained PII, they could not be used at all for a chatbot serving data out to internal stakeholders at the university.

This dictionary acted as scaffolding for the LLM:

- defining the language of the domain
- clarifying what concepts existed
- constraining how questions should be interpreted
- helping the model distinguish valid requests from unsupported ones

Rather than relying on the model's general knowledge, we used structured context to shape how it reasoned about the problem space.

The prompt effectively became a lightweight interface layer between users, institutional terminology, and the underlying data systems.

```python
SYSTEM_PROMPT = """
You are an educational analytics assistant.

### brief data dictionary ###
Use only the approved terms below:
- Attendance Rate
- Engagement Score
- Intervention Tier
- Student Success Trend

Rules:
- Only return aggregate data
- No individual student data (PII)
- Say "out of scope" if not supported
- Don't join the Attendance Rate data to an external, aggregate student metric table; that data contains information at an opposing grain.
"""

response = llm.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Which tiers improved attendance?"}
    ]
)
```

To confirm the dictionary was being utilized correctly, our institutional partners wanted to see how the LLM got to its final answer by showing its logic along the way.

---

## Core Pattern 3: Simple Agent Loops

While Core Pattern 1 focused on a single function call, many institutional questions in Audrey's project required chaining multiple tools together. This created a natural opportunity for lightweight agentic workflows, where the system could retrieve data, compute metrics, compare results, and generate explanations through a multi-step reasoning process.

Instead of a single query response cycle, the system often needed to:

- retrieve a student cohort
- compute or fetch predictive metrics
- compare trends across groups
- generate a narrative summary

This created an iterative loop:

```
interpret → retrieve → compute → refine → explain
```

In this setup, the LLM becomes an orchestrator that decides what to analyze next, while Python tools handle each step of computation.

Predictive analytics here becomes less of a model output, and more of a tool-driven reasoning workflow over data.

Example Flow:

```
Question → LLM → Tool 1: Get cohort → Tool 2: Get retention scores → Tool 3: Compare to prior semester → Tool 4: Generate summary statistics → LLM explanation
```

```python
# Available tools
tools = [
    get_student_cohort,
    get_retention_risk,
    compare_to_previous_term
]

query = """
Which first-year students are most at risk,
and is retention improving or declining?
"""

# Agent loop
cohort = get_student_cohort(
    year="first_year"
)

risk_scores = get_retention_risk(
    students=cohort
)

trend = compare_to_previous_term(
    current=risk_scores
)

summary = llm.generate(
    f"""
    Cohort: {cohort}
    Risk Scores: {risk_scores}
    Trend Analysis: {trend}

    Summarize findings for university advisors.
    """
)

print(summary)
```

---

## Core Pattern 4: Retrieval-Augmented Generation (RAG)

Jasmine's hackathon project relied heavily on RAG to ground responses in a trusted educational context related to student success. Her institution was curious about "toxic" course loads, where students may be taking too many science or humanities classes simultaneously. We used the data dictionary and embeddings to determine how a user's request adheres to the domain.

Embeddings and retrieval pipelines are helpful to:

- guide the chatbot toward approved terminology
- retrieve relevant institutional definitions and resources
- constrain answers to known educational concepts
- reduce hallucinations in a sensitive domain

Given more time, we would have extended this to further cement the data dictionary as an important guardrail of the system by:

- more tightly integrating the data dictionary into the embedding pipeline as a curated "semantic layer" over the domain
- improving chunking so definitions, metrics, and relationships were retrieved together rather than in isolation
- adding evaluation checks to verify that retrieved context matched the user's intent before generation
- strengthening guardrails to ensure the model always preferred retrieved institutional context over general knowledge

This ended up being one of the most important guardrail mechanisms in the system.

```python
import numpy as np

client = YourLLM()

# Domain scaffold (data dictionary)
data_dictionary = [
    "Attendance Rate: % of classes attended",
    "Engagement Score: participation metric",
    "Intervention Tier: support level",
]

query = "Which intervention tiers improved attendance?"

# Embed dictionary entries (used for semantic search)
dict_emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=data_dictionary
).data

# Embed the user query (what we are trying to match)
q_emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding


def cosine(a, b):
    # measures similarity between two embedding vectors
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# Rank dictionary items by relevance to the query
best = sorted(
    zip(data_dictionary, dict_emb),  # pair each term with its embedding
    key=lambda x: cosine(q_emb, x[1].embedding),  # compute similarity
    reverse=True  # most relevant first
)[:2]  # take top results

# Build context for the LLM from retrieved items
context = "\n".join([text for text, _ in best])


# Send retrieved context + query to the LLM
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": query}
    ]
)

print(resp.choices[0].message.content)
```

---

## Possible Extensions of LLM Usage

Given more time, we would expand into additional routing and moderation logic.

### Project Optimizations

- Benchmark LLM answers and validity
- Add time series modeling to improve predictions
- Expand the chatbot to answer broader student data questions

### Reliability & Guardrails

LLMs are probabilistic systems, so production reliability matters.

- Detect and redirect off-topic queries
- Handle sensitive data with escalation paths
- Reject unsupported requests
- Use retries, timeouts, and output validation
- Log prompts for observability
- Add hallucination checks and input sanitization

**Principle:** The goal is not to rely solely on the model, but to build systems that remain reliable even when it makes mistakes.

---

## Conclusion

Not every problem needs an agent.

Use:

- A single prompt for simple classification or extraction
- A pipeline for fixed workflows
- An agent only when adaptive multi-step reasoning is required

Start with the simplest working approach before adding complexity. The path is straightforward:

1. Start with a simple LLM API call
2. Add structured outputs
3. Add tool calling
4. Add RAG
5. Introduce LLMs only when necessary

You do not need deep ML expertise to build useful AI systems. You need strong engineering fundamentals, clear abstractions, and careful system design.
