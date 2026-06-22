---
title: Building Trustworthy Scientific Python Workflows in Pharma
authors:
  - name: Tarun Gandrathi
    affiliation: ZS Associates, Hyderabad, India
    email: tarun.gandrathi@zs.com
  - name: Subbiah Sethuraman
    affiliation: ZS Associates
  - name: Rahul Sahu
    affiliation: ZS Associates
keywords: [Agentic AI, Knowledge Graph, Neo4j, Blueprint-driven generation, RAG, LangGraph, Regulatory authoring, Pharma, Production deployment]
abstract: |
  Pharmaceutical regulatory authoring requires generating large, structured, and evidence-backed documents under strict constraints of consistency, traceability, and reviewability. This paper presents a production-grade architecture for blueprint-driven regulatory document authoring in pharmaceutical R&D. Rather than treating document generation as a single-pass language model task, the system resolves a document blueprint in advance to specify section structure, dependencies, evidence requirements, retrieval constraints, and validation rules. An agentic orchestration framework decomposes the blueprint into parallel section-level tasks executed by specialist agents responsible for retrieval, drafting, critique, and replanning. Evidence grounding is provided through a graph database platform with semantic retrieval, exposed via governed tool contracts aligned with the Model Context Protocol (MCP). Production deployment across five regulated document types demonstrates the viability of this approach for enterprise pharmaceutical authoring.
---

## 1. Introduction

Pharmaceutical R&D organizations generate thousands of regulated documents each year, including Clinical Study Reports (CSRs), Informed Consent Forms (ICFs), and Chemistry, Manufacturing and Controls (CMC) documents. These documents require structured evidence synthesis, cross-section consistency, and traceable support for all generated content.

Recent advances in large language models (LLMs) and agentic systems have enabled innovative approaches to regulatory authoring and retrieval-grounded workflows. However, most published systems concentrate on discrete tasks — such as question answering, compliance monitoring, or scientific assistance — rather than end-to-end document generation.

A meaningful gap persists between prototype agent systems and production-grade regulatory authoring. Regulated documents demand section-level consistency, sentence-level provenance, and operational scalability, yet many existing approaches offer limited support for quality measurement, cost governance, and robust execution at scale.

This paper presents a blueprint-driven regulatory authoring architecture that integrates structured document planning, graph-grounded retrieval, agentic orchestration, and enterprise governance controls.

## 2. Problem Statement

Regulatory authoring presents four substantive challenges that single-pass language model generation does not adequately address: structural inconsistency, limited traceability in flat retrieval systems, the operational constraints of sequential generation workflows, and the absence of end-to-end authoring systems with blueprint-driven orchestration.

- **Structural Inconsistency:** Without a governing blueprint, models are prone to generating contradictory or duplicated content across sections of large documents.
- **Flat Retrieval Lacks Traceability:** Vector-based search surfaces relevant content but cannot establish which source section supports a specific claim, limiting auditability.
- **Sequential Generation Does Not Scale:** At approximately 25 language model calls across 60 sections, sequential generation imposes unacceptable latency for enterprise regulatory workflows.
- **Published Systems Are QA Tools, Not Authoring Systems:** The majority of published research addresses question answering or compliance monitoring rather than end-to-end, blueprint-driven document authoring.

Taken together, these limitations underscore the gap between prototype LLM applications and production-grade regulatory authoring. The central challenge addressed in this work is how to advance from exploratory prototype systems to production-ready regulatory document generation.

## 3. Blueprint-Driven Authoring

Blueprint-driven authoring reframes document generation as a planning problem before it becomes a writing task. Rather than directing a language model to produce a complete document in a single pass, the system first creates or retrieves a structured blueprint that defines the document hierarchy, section scope, inter-section dependencies, evidence requirements, retrieval constraints, and validation rules. This blueprint serves as the authoritative control layer for the entire workflow, ensuring that document structure is fully resolved before any section is drafted.

By decoupling planning from generation, blueprint-driven authoring makes document construction more controlled and reproducible. Each section is treated as a discrete unit with clearly defined requirements, which reduces the incidence of missing, duplicated, or contradictory content and allows retrieval, drafting, validation, and review to operate against a shared, consistent plan. In this sense, the blueprint is not merely a formatting template; it is a persistent planning artifact that coordinates how content is produced and verified across the full document.

This architecture is fundamentally distinct from a conventional LLM combined with retrieval-augmented generation (RAG). Retrieval continues to supply relevant evidence, but the blueprint governs what must be written, what evidence is required, how sections relate to one another, and how outputs are validated. Rather than relying solely on prompts to infer structure during generation, the system converts open-ended text generation into controlled, section-aware authoring — an approach far better suited to the demands of large, regulated documents.

## 4. System Architecture

:::{figure} figures/architecture.png
:label: fig-architecture
System architecture showing the five layers: user entry, agentic orchestration, MCP tool layer, knowledge layer, and production execution layer.
:::

The system architecture converts a regulatory authoring request into a governed, section-level generation workflow. Rather than generating a complete document in a single language model call, the system first resolves document structure through a blueprint, which subsequently drives retrieval, drafting, critique, validation, and final assembly.

@fig-architecture illustrates the architecture across integrated layers spanning agentic orchestration, knowledge grounding, blueprint planning, and production execution.

The system described here is a production deployment in a proprietary pharmaceutical environment. Implementation code is not publicly available, but the architectural patterns and design decisions are described in full for reproducibility.

### 4.1 Agentic Orchestration Layer

The agentic orchestration layer manages section-level document generation end to end. A supervisor component within the orchestration framework decomposes the document goal into a dependency-aware task graph. Specialist agents handle retrieval, writing, critique, validation, replanning, and formatting — making the workflow more controllable and localizing failures for easier diagnosis and remediation.

### 4.2 Knowledge Layer with Graph-Grounded Retrieval

The knowledge layer grounds generation in verifiable source evidence. Documents are parsed into a graph database platform comprising sections, entities, and evidence nodes connected by typed relationships. This structure enables relational retrieval and end-to-end provenance that flat vector search cannot support. A semantic search engine augments the graph layer with passage-level retrieval to broaden evidence coverage.

### 4.3 Blueprint Layer

The blueprint layer defines document structure, section dependencies, evidence requirements, retrieval filters, and validation rules before any drafting commences. Each section is encoded as a contract, reducing the risk of missing or duplicated content and enabling safe parallel generation. The blueprint functions as the workflow's authoritative control layer throughout the authoring lifecycle.

### 4.4 Governed Tool Contract Layer

Agents interact with the knowledge layer through a governed tool contract layer aligned with the Model Context Protocol (MCP). This layer provides standardized interfaces for section retrieval, semantic retrieval, table retrieval, and image retrieval, while enforcing governance over what agents may query, how queries are executed, and what evidence is returned.

### 4.5 Production Execution and Observability Layer

The production layer enables reliable, enterprise-scale execution through asynchronous workers, job queues, retry logic, idempotent task design, and circuit breakers. Distributed tracing, token-cost tracking, and automated quality evaluation provide end-to-end observability. Parallel section execution respects blueprint-defined dependencies, and validated outputs are assembled with embedded links to source evidence and generation traces.

Overall, the architecture shifts regulatory authoring from prompt-centric generation to blueprint-driven document construction. The blueprint defines what must be written; the knowledge layer grounds what may be asserted; the agentic layer governs how sections are produced; and the production layer ensures the workflow is scalable, observable, and auditable.

## 5. End-to-End Workflow

The system generates a regulatory document through a blueprint-driven, section-level workflow rather than a single-pass language model response. The workflow proceeds through eight discrete stages:

1. **User Submits the Document Request:** The user selects the target document type, uploads source materials, and provides authoring instructions through the authoring interface.
2. **Backend Validates and Initializes the Job:** The API gateway validates user access, initializes the document generation job, and places it on the asynchronous execution queue.
3. **Blueprint Is Resolved:** The system resolves the document into a structured blueprint defining required sections, subsections, inter-section dependencies, evidence requirements, retrieval filters, and validation rules.
4. **Orchestrator Creates Section-Level Tasks:** The agentic orchestration framework decomposes the blueprint into section-level tasks and determines which sections may execute in parallel based on dependency rules.
5. **Evidence Is Retrieved for Each Section:** For each section, the context retriever gathers relevant evidence through governed tool contracts using graph-grounded and semantic retrieval.
6. **Section Draft Is Generated:** The writer agent drafts each section using the retrieved evidence and the section-specific contract defined in the blueprint.
7. **Draft Is Critiqued and Validated:** The critique agent assesses whether each section adheres to the required structure, employs appropriate evidence, and satisfies validation rules. Where gaps are identified, the system may retrieve additional evidence, replan, or regenerate the section.
8. **Final Document Is Assembled for Review:** Validated sections are assembled into a complete draft document. The final output preserves links to the blueprint, source evidence, generation traces, and validation results, rendering it ready for human review and refinement.

## 6. Results

The system was evaluated through production deployment and load-tested authoring workflows across multiple regulatory and medical document types. The evaluation objective extended beyond fluent text generation to assess whether blueprint-driven orchestration could reliably support large, structured documents with traceable evidence and review-ready outputs.

The deployment demonstrated that regulatory authoring scales more effectively when document generation is treated as a governed workflow rather than a one-shot prompting task. The combination of blueprint-first planning, graph-grounded retrieval, asynchronous execution, and structured human review provides a practical and validated path from prototype document generation to production-scale authoring.

@tbl-results summarizes the principal metrics from the platform design specification.

```{list-table} Results and impact metrics from the platform design specification.
:label: tbl-results
:header-rows: 1
* - Metric
  - Value
  - Description
* - Authoring time reduction
  - 50%+
  - Target reduction in authoring time vs. the manual baseline
* - Sections per document
  - ~60
  - Sections per document at load-tested scale
* - Maximum parallel section tasks
  - 256
  - 8 workers × 4 cores × batch 8
* - Document types deployed
  - 5+
  - ICF, SOA, CSR, CMC, HAI
```

**Note:** All four figures are drawn from the platform design specification. The 50% figure is a design target versus the manual baseline, not a post-deployment measured outcome.

:::{figure} figures/results.png
:label: fig-results
Results and impact metrics: target authoring-time reduction, sections per document, maximum parallel section tasks, and document types deployed.
:::

## 7. Lessons Learnt

Advancing from prototype to production required the system to generate structured, traceable, and review-ready content across many sections simultaneously, while rigorously controlling cost, latency, access, and failure modes — demands that far exceed what prototype viability entails. Several consequential lessons emerged from deployment.

- **Governed tool contracts make every interaction auditable.** All agent-to-data interactions pass through typed MCP tool contracts, which means that every tool call and retrieval is logged and auditable. In a regulated environment this governance proved as important as the generated text itself, because reviewers must be able to reconstruct exactly which evidence an agent accessed and how it was queried.
- **Blueprint quality matters more than model selection.** Investing in rigorous blueprint design measurably reduced missing, duplicate, and mis-scoped content across documents exceeding 60 sections. Across functionally similar document types, careful blueprint engineering delivered a higher return on investment than switching between comparable language models.
- **Graph grounding enables provenance that flat retrieval cannot.** Parsing source material into a Neo4j graph enabled sentence-level provenance that flat vector search cannot provide. Because individual claims could be traced through typed relationships back to specific source sections, reviewers were able to verify statements one sentence at a time rather than accepting whole passages on trust.
- **Critique agents dominate token cost.** The critique agents account for approximately 60% of total token spend per document, far exceeding the cost of the writer agents. This makes batching validation steps a substantially higher-value optimisation than further tuning of writer prompts.
- **Average token usage hides large variance.** The average of roughly 100K tokens per section conceals a wide distribution, ranging from about 20K tokens for short sections to over 200K tokens for dense synthesis sections. This variance materially affects both cost estimation and latency SLA design, since planning around the mean understates the load created by the heaviest sections.

## 8. Future Work

The production architecture creates the foundation for continuous quality because each generated section is linked to its blueprint contract, retrieved evidence, validation trace, and reviewer actions. The next step is to use these signals not only for auditability, but also for ongoing improvement of blueprint design, retrieval quality, and validation rules.

First, the knowledge layer can be extended to multimodal evidence. Future work includes representing clinical imaging metadata, assay result tables, and other non-text evidence as structured graph nodes. This would allow the system to reason across text, tables, and metadata through the same governed retrieval layer.

Second, agent-to-agent integration can expand specialized capabilities. The document platform can delegate specific subtasks to external specialized agents, such as molecular property retrieval services or domain-specific analysis agents, while still preserving the blueprint-driven control layer.

Third, reviewer feedback can become a quality signal. Corrections and reviewer actions can be captured and fed back into blueprint finalization, retrieval filters, and validation checks. Over time, this creates a feedback loop where production usage improves future document generation quality.

Together, these directions move the system from a production authoring architecture toward a continuously improving quality framework, where blueprints, retrieval, validation, and human review evolve together across deployment cycles.

## 9. Conclusion

This work demonstrates that production-grade regulatory authoring requires considerably more than connecting a language model to a corpus of source documents. At pharma scale, the core challenge is generating structured, evidence-backed, review-ready content while preserving document coherence, provenance, governance, and operational reliability.

The principal contribution is a **blueprint-driven authoring architecture** in which the document blueprint serves as the authoritative control layer for generation. The blueprint defines what must be written; the knowledge layer grounds what may be asserted; the agentic workflow determines how sections are produced; and the production layer ensures the process is scalable, observable, and auditable.

By combining blueprint-first planning, graph-based knowledge grounding, governed tool contracts, asynchronous orchestration, and structured human review, the system provides a validated path from prototype LLM drafting to production-scale regulatory document generation. It transforms the authoring process from prompt-centric text generation to controlled document construction — making AI-assisted authoring substantively more suitable for regulated biomedical environments.

## References

1. Boiko D, et al. Autonomous chemical research with large language models. *Nature*. 2023;624:570–578.
2. Bran A, et al. ChemCrow: Augmenting large-language models with chemistry tools. *Nature Machine Intelligence*. 2024.
3. Anthropic. Model Context Protocol specification. 2024. [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
4. Agarwal B, et al. RAGulating Compliance: Multi-Agent Knowledge Graph for Regulatory QA. arXiv:2508.09893. 2025.
5. Es S, et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024.
6. IntuitionLabs. Agentic AI for Pharma Regulatory Document Automation. 2025. intuitionlabs.ai

_All claims have been verified by the authors against source documents. The authors take full responsibility for the accuracy of all content._
