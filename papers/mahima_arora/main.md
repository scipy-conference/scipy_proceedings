---
title: Docling for Multimodal Retrieval
abstract: |
    Retrieval-Augmented Generation (RAG) has become a widely adopted approach for grounding large language models in external knowledge sources. However, most RAG pipelines remain text-centric, overlooking diagrams, figures, tables, and layout structures that contain critical information in scientific, technical, and enterprise documents. This paper presents a multimodal document parsing framework built on Docling, a layout-aware and format-agnostic document understanding library, that preserves document structure by extracting text, tables, and images as distinct retrieval units. The framework employs modality-specific representations: hierarchy-aware text chunks, image chunks enriched with vision-language descriptions, and table chunks that retain relational structure while supporting deterministic SQL querying. These representations are indexed in a unified vector database and served to a tool-augmented agent capable of retrieving and reasoning over textual, tabular, and visual information. Evaluation on scientific and technical documents shows improvements over a text-only baseline in context recall, faithfulness, and answer correctness, with the largest gains on image and table queries.
---

## Introduction

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by retrieving relevant information from external knowledge sources at inference time, grounding model responses in factual, domain-specific content without requiring costly retraining. It has become a foundational approach for building accurate, knowledge-intensive applications, and its standard pipeline follows a well-established pattern: documents are loaded, split into text chunks, embedded into a vector database, and retrieved to condition generation. While effective for purely textual corpora, this paradigm breaks down when applied to the complex, multimodal documents that dominate real-world knowledge bases.

Scientific papers, technical manuals, and enterprise reports communicate knowledge through far more than prose. Architecture diagrams, performance tables, charts, and structured layouts often carry the most critical information in such documents. Yet conventional RAG pipelines silently discard these elements during ingestion, reducing rich documents to fragmented text. This introduces two concrete failure modes. The first is a *context gap*: when narrative text references a figure or diagram that was dropped during parsing, the retrieved context is semantically incomplete and the model cannot reason over evidence it never received. The second is the *table trap*: flattening structured tables into linear token sequences destroys row-column relationships and renders the resulting text chunks neither accurately embeddable nor reliably retrievable.

Addressing these limitations requires treating documents not as streams of text, but as coordinated ecosystems of distinct content types: narrative text, visual elements, and relational data. Each modality demands its own representation and retrieval strategy.

This paper presents a multimodal document parsing and retrieval framework built on Docling, a layout-aware document understanding library. The framework introduces a typed chunking architecture with modality-specific enrichment. Text chunks use a hierarchy-aware hybrid chunking strategy that preserves section context. Image chunks are enriched with vision-language descriptions and classified by semantic type to support targeted retrieval. Table chunks preserve full relational structure as pandas DataFrames, paired with generated descriptions for semantic retrieval and a SQL execution interface for deterministic querying. These typed chunks are indexed in a unified vector database and served to a tool-augmented agent capable of synthesizing answers across all three modalities.

The key contributions of this work are:

1. A formal characterization of the structural failure modes of text-only RAG on complex documents
2. A layout-aware multimodal parsing pipeline built on Docling that extracts text, images, and tables as first-class retrieval units
3. Modality-specific enrichment strategies including vision-language captioning and SQL-backed table querying
4. A tool-augmented agent architecture that combines semantic retrieval with deterministic execution to enable faithful reasoning across modalities
5. An open-source implementation facilitating reproducibility and community adoption

Together, these contributions advance multimodal document understanding as an essential capability for RAG systems operating over real-world scientific, technical, and enterprise document collections.

## Related Work

Zhao et al. [@zhao2023multimodal_rag_survey] survey retrieval-augmented generation across modalities including images, code, tables, graphs, and audio, establishing a taxonomy of algorithmic approaches. At the framework level, several systems address multimodal content in RAG pipelines using patterns related to those in this work. LangChain's multi-vector retriever [@langchain_multivector] embeds generated text summaries of tables and images for semantic search while storing original content in a separate document store for downstream synthesis. This caption-and-embed pattern makes non-text elements discoverable but does not prescribe modality-specific access patterns at query time; retrieved tables are passed as raw content to the LLM rather than queried deterministically. LlamaIndex [@llamaindex] provides query engines that translate natural language into executable code against structured data, including `NLSQLTableQueryEngine` for SQL-based access and `SQLAutoVectorQueryEngine` for combining vector retrieval with SQL execution. These enable deterministic tabular querying but focus on text and structured data without incorporating visual content.

This framework builds directly on Docling [@docling], which provides the layout-aware parsing backbone, including its `HybridChunker` for hierarchy-aware text segmentation and vision-language captioning for image description. What Docling does not provide is the retrieval and reasoning layer: a typed chunk schema unifying text, images, and tables in a single vector space; SQL-backed table querying for deterministic access; and a tool-augmented agent that selects the appropriate access pattern per modality at query time. Like the multi-vector retriever, the framework uses generated descriptions for cross-modal semantic search; like LlamaIndex's query engines, it supports deterministic SQL execution. The contribution is this end-to-end architecture integrating parsing, modality-specific enrichment, unified retrieval, and agent-based reasoning.

## Methodology

### Document Parsing with Docling

The framework uses Docling [@docling], a layout-aware document understanding library that performs full layout analysis across PDF, DOCX, PPTX, and XLSX formats. Docling's pipeline extracts three content modalities: (1) **text** with preserved document hierarchy and section context, (2) **tables** using TableFormer structure recognition to recover row-column relationships, and (3) **images** with vision-language generated descriptions that make visual content semantically searchable. All formats are normalized to PDF for consistent processing, and the parser outputs structured document objects containing text blocks, table objects, and image elements, each annotated with page number, section context, and content type.

### System Architecture

The framework transforms Docling's structured document representation into a multimodal retrieval system through five stages: parsing, chunking, indexing, retrieval, and agent-based answer generation. @fig:architecture shows the complete pipeline.

:::{figure} full_architecture.jpg
:label: fig:architecture
:width: 75%
System architecture: document ingestion, Docling parsing, modality-specific chunking, vector indexing, and agent-based answer generation.
:::

Three design principles guide this architecture: (1) modality preservation (maintain native representations rather than forcing everything into text), (2) retrieval optimization (generate descriptions that emphasize what questions each element can answer), and (3) deterministic access (provide SQL and display tools for precise operations that embeddings cannot reliably support).

### Chunking Strategy

The chunking stage addresses a fundamental question: how should multimodal document elements be represented for semantic retrieval? Text-only systems flatten everything into character sequences, destroying the structural properties that make tables queryable and visual elements interpretable. The proposed approach defines a typed chunk architecture where each modality receives a representation suited to its information characteristics.

All chunks share a common base schema that enables unified retrieval while preserving modality-specific structure. The key design constraint is that text, images, and tables must be searchable in the same vector space. Rather than joint pixel-text or table-text embeddings, this is achieved by applying standard text embeddings to a `content` field that stores a natural language representation for every modality: narrative prose for text chunks, vision-language generated descriptions for images, and schema-aware descriptions for tables. The base schema tracks provenance and provides this unified embedding interface:

```python
class BaseChunk(BaseModel):
    chunk_id: str
    chunk_type: ChunkType  # TEXT, TABLE, or IMAGE
    content: str  #  (text or image/table description)
    sequence_number: int
    source_document: Optional[str]
    source_page: Optional[int]
    parent_heading: Optional[str]  # preserves document hierarchy
```

This unified interface allows all chunk types to be indexed together via the `content` field, while specialized subclasses add modality-specific fields (DataFrames for tables, image paths for visuals) that enable deterministic operations beyond semantic search.

#### Text Chunks

:::{figure} text_chunk.png
:label: fig:text_chunk
:width: 75%
Text chunk schema with hierarchy-aware structure and parent heading context.
:::

Text chunks solve the problem of preserving narrative context across document structure. Fixed-size chunking (e.g., every 512 tokens) fragments coherent sections and concatenates unrelated paragraphs, breaking the semantic units humans rely on for understanding. The text chunking strategy preserves hierarchy by tracking the parent heading and uses intelligent merging to respect document structure:

```python
chunker = HybridChunker(
    max_tokens=2048,
    merge_peers=True,
    merge_threshold=0.5,
)
```

The chunker merges text blocks within the same section when semantic similarity exceeds the threshold, creating chunks that align with how documents are actually organized. A methods section stays together; introduction and conclusion remain separate. The `content` field contains the merged text, and `parent_heading` preserves the section title, allowing retrieval to return contextually grounded chunks rather than orphaned fragments.

#### Image Chunks

:::{figure} image_chunk.png
:label: fig:image_chunk
:width: 75%
Image chunk schema with vision-language description for embedding and type classification.
:::

Image chunks address the context gap failure mode: when figures are discarded during parsing, retrieval cannot answer visual questions. The solution pairs every extracted image with a vision-language generated description that makes visual content semantically searchable. When a document caption is available (e.g., "Figure 3: Performance comparison"), it is appended to the generated description in the `content` field, preserving figure labels for retrieval. This allows both semantic queries like "what is the system architecture" and label-based references like "Figure 3" to match against image chunks.

Images are classified by semantic type (diagram, chart, screenshot, photo, logo) to support targeted retrieval and filtering:

```python
class ImageChunk(BaseChunk):
    """Chunk containing image data."""
    chunk_type: ChunkType = Field(default=ChunkType.IMAGE)
    image_path: Optional[str] = Field(default=None, description="Extracted image path")
    image_format: Optional[str] = Field(default=None, description="png, jpg, gif, webp, etc.")
    image_type: Optional[ImageType] = Field(default="other", description="diagram, chart, etc.")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image data")
```

Classification uses a vision-language model to distinguish diagrams, charts, screenshots, photos, and logos, enabling filtered retrieval (e.g., only architecture diagrams for system design questions). The `content` field contains retrieval-optimized descriptions emphasizing what questions the image can answer, producing semantic descriptions like "system architecture diagram showing encoder-decoder structure" rather than pixel-level enumeration.

#### Table Chunks

:::{figure} table_chunk.png
:label: fig:table_chunk
:width: 75%
Table chunk schema with DataFrame preservation and description for semantic retrieval.
:::

Table chunks solve the table trap: flattening a table to text destroys the relational structure that makes it valuable. A benchmark comparison table showing model performance across datasets becomes an unstructured paragraph that embeddings cannot reliably parse for precise comparisons or metric lookups. The approach preserves the full DataFrame while adding a semantic description for retrieval:

```python
class TableChunk(BaseChunk):
    """Chunk containing table data as pandas DataFrame."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    chunk_type: ChunkType = Field(default=ChunkType.TABLE)
    dataframe: pd.DataFrame = Field(description="Table data as pandas DataFrame")
    columns: Optional[List[str]] = Field(default=None, description="List of column names")
    num_rows: Optional[int] = Field(default=None, description="Number of rows")
    num_cols: Optional[int] = Field(default=None, description="Number of columns")
```

Tables require a two-stage strategy. First, semantic retrieval using generated descriptions that emphasize schema, sample data patterns, and query-answering capability (e.g., "performance metrics across five benchmark datasets"). Second, SQL execution for deterministic data access, once retrieved, the agent generates and executes queries against the DataFrame:

```python
def query_table(table_index: int, sql: str) -> str:
    chunk = table_chunks[table_index]
    result_df = ps.sqldf(sql, {"df": chunk.dataframe})
    return result_df.to_string(index=False)
```

For example, a query like "Which model achieved the highest accuracy on the SQuAD benchmark?" retrieves a benchmark results table via description matching, then executes `SELECT model, accuracy FROM df WHERE dataset='SQuAD' ORDER BY accuracy DESC LIMIT 1` to extract the exact answer. This hybrid approach, semantic discovery plus deterministic execution, provides both flexibility and precision.

### Indexing and Retrieval

All three chunk types flow into a unified indexing pipeline. The key insight is that text chunks, image descriptions, and table descriptions are all natural language strings that can be embedded into the same semantic space, enabling cross-modal retrieval where a text query can match against any content type. The framework uses a vector database for storing embeddings and metadata with efficient similarity search.

Indexing batch-embeds all `content` fields and stores them alongside full chunk metadata:

```python
def index_file(file_path: str) -> int:
    chunks = parser.parse(file_path)
    contents = [chunk.content for chunk in chunks]
    embeddings = embed_model.embed_documents(contents)

    for chunk, embedding in zip(chunks, embeddings):
        payload = chunk.model_dump(exclude={"dataframe"})
        if chunk.chunk_type == ChunkType.TABLE:
            payload["table_data"] = chunk.dataframe.to_dict(orient="records")
        store_vector(embedding, payload)
```

For tables, the DataFrame is serialized as a dict and stored in the payload rather than embedded, since row-column data is not directly embeddable. This keeps the structured representation available for SQL execution while the description enables semantic matching.

Retrieval performs standard dense vector search. A query like "What is the proposed model architecture?" is embedded and matched against all chunk content fields, returning the top-k most similar chunks regardless of type. A diagram description like "architecture diagram showing transformer encoder-decoder with attention mechanisms" can score highly despite being a visual element, solving the context gap by making visual content discoverable through text queries.

### Agent-Based Answer Generation

Retrieval alone is insufficient for multimodal documents because different modalities require different interaction patterns. Text can be synthesized directly, but images must be displayed and tables must be queried. The framework employs a tool-augmented agent architecture [@yao2022react] that combines semantic retrieval with modality-specific tools and inline rendering capabilities.

The agent receives retrieved chunks separated by type (text, image, table) and decides which operations to invoke:

- **For text chunks**: Synthesize information directly from content
- **For image chunks**: Call `display_image(image_index)` to trigger inline rendering
- **For table chunks**: Generate and execute SQL via `query_table(table_index, sql)`

#### Inline Display Mechanism

```python
def display_image(image_index: int) -> str:
    chunk = image_chunks[image_index]
    displayed_images.append({
        "chunk_id": chunk.chunk_id,
        "path": chunk.image_path,
        "base64": chunk.image_base64,
    })
    return f"Image ready. Use placeholder [IMAGE:{len(displayed_images)-1}]
             in your answer on its own line between paragraphs."
```

Rather than returning image URLs or table data for external rendering, the agent employs a placeholder-based inline display system. When `display_image` is invoked, the tool stores the image data (base64 or file path) in a display queue and returns instructions to insert a placeholder token `[IMAGE:X]` in the answer text:

The agent is instructed to place these placeholders on separate lines, never inline within sentences, ensuring clean paragraph-image-paragraph flow in rendered output. For tables, `query_table` executes SQL and returns formatted results as text while storing the structured result in a `queried_tables` list for reference.

This tool-calling workflow enables precise operations that pure text generation cannot reliably produce. Consider a query like "What is the average F1 score across all baselines?" A text-only system must generate the answer from a flattened table representation, risking arithmetic errors. The agent retrieves the benchmark results table, generates `SELECT AVG(f1_score) FROM df WHERE model_type='baseline'`, executes it deterministically, and returns the exact result.

Similarly, visual queries like "What is the attention mechanism architecture?" retrieve an architecture diagram (via description matching) and trigger `display_image`, which inserts `[IMAGE:0]` into the answer. Downstream rendering replaces this placeholder with the actual image, presenting visual evidence directly rather than attempting to describe complex spatial relationships in generated text.

The agent workflow is:

1. Embed query and retrieve top-k chunks via semantic search
2. Separate retrieved chunks by type (text, image, table)
3. For text chunks, synthesize answer from content
4. For image chunks, invoke display tool and insert placeholder in answer
5. For table chunks, generate SQL query, execute for exact data
6. Return answer with placeholders plus image/table metadata for rendering

When retrieval returns multiple chunks of the same type, all chunks are presented to the agent in a single context, grouped by modality with numerical indices. The agent selects which chunks to act on, querying multiple tables or displaying multiple images, within a single reasoning pass. For table querying, SQL execution errors such as syntax mistakes or invalid column references are caught and returned to the agent as descriptive error messages, enabling it to revise the query and retry within the same turn.

This architecture demonstrates that multimodal RAG requires heterogeneous access patterns, text generation for prose, tool calls for structure, inline placeholders for visual elements, rather than forcing all modalities through a single generation interface.

## Results

The framework is implemented as open-source software [@multimodal_parser] using Gemini 2.5 Pro [@gemini] for agent reasoning and tool calling, Gemini Embedding [@gemini_embedding] for vector representations, Qdrant [@qdrant] for vector storage, Docling [@docling] for document parsing, LangGraph [@langgraph] for agent orchestration, and pandasql [@pandasql] for SQL execution against DataFrames.

### Capability Demonstrations

Three demonstrations validate the framework's ability to handle queries that text-only RAG systems cannot answer. Each demonstration uses research paper PDFs as source documents and illustrates a distinct modality handling strategy.

**Text Synthesis.** The agent performs cross-sectional synthesis by connecting information across fragmented text chunks (@fig:result_text). The hybrid chunking strategy merges semantically related text blocks from the same section, enabling coherent answers that integrate information from multiple sources while avoiding over-fragmentation.

:::{figure} result_text.png
:label: fig:result_text
:width: 40%
Text synthesis: cross-sectional analysis connecting information across multiple chunks.
:::

**Visual Reasoning.** The agent identifies visual query intent, locates corresponding image chunks using vision-language descriptions, and displays diagrams inline (@fig:result_visual). This addresses the context gap problem where text-only systems cannot answer queries about discarded figures, vision-language descriptions make visual content semantically searchable.

:::{figure} result_visual.png
:label: fig:result_visual
:width: 40%
Visual reasoning: diagram retrieval using vision-language descriptions and inline display.
:::

**Structured Precision.** The agent combines semantic search over table descriptions with deterministic SQL execution against DataFrames (@fig:result_table). This addresses the table trap problem where text-only systems hallucinate from flattened representations, the dual approach enables exact numerical results without arithmetic errors.

:::{figure} result_table.png
:label: fig:result_table
:width: 40%
Structured precision: SQL-based table querying with semantic search and deterministic execution.
:::

To quantify the impact of modality preservation, the multimodal pipeline was evaluated against a text-only baseline using RAGAS metrics [@ragas]. The evaluation uses a corpus of 110 research papers [@eval_dataset] with 300 queries constructed across four categories: text-only (28%), table (28%), image (24%), and cross-modal (20%) queries requiring synthesis across modalities. The text-only baseline uses identical chunking and retrieval parameters but converts tables to inline Markdown text and discards images during ingestion. Results are averaged over three independent runs; 95% confidence intervals are reported.

:::{table} Multimodal (MM) vs. text-only (TO) pipeline comparison by query type. CR = Context Recall, F = Faithfulness, AC = Answer Correctness. Overall values show mean ± 95% CI over three runs.
:label: tab:baseline

| Query Type | CR (MM) | CR (TO) | F (MM) | F (TO) | AC (MM) | AC (TO) |
|:-----------|:-------:|:-------:|:------:|:------:|:-------:|:-------:|
| Text | 91.8% | 91.0% | 95.2% | 94.5% | 79.6% | 78.8% |
| Table | 89.2% | 82.3% | 94.8% | 90.3% | 76.5% | 69.2% |
| Image | 83.5% | 64.7% | 92.5% | 84.8% | 70.3% | 60.5% |
| Cross-modal | 77.4% | 74.2% | 90.7% | 89.5% | 66.8% | 64.7% |
| **Overall** | **86.2 ± 2.3%** | **78.9 ± 1.3%** | **93.6 ± 1.4%** | **90.0 ± 0.9%** | **74.0 ± 2.4%** | **68.9 ± 1.7%** |

:::

The per-modality breakdown reveals that gains concentrate on image queries, where the baseline discards the primary evidence source, context recall improves by 18.8 percentage points. Table queries show a moderate gap (6.9 pp in context recall) since the baseline retains table content as inline text but lacks deterministic SQL access for precise lookups. Text queries show minimal difference (0.8 pp), confirming that the multimodal pipeline preserves text handling quality.

These results validate that multimodal RAG requires both modality-specific representations and heterogeneous access patterns. The bottleneck has shifted from LLM reasoning to data quality and modality preservation.

## Discussion

### Runtime Feasibility

The SQL execution pathway requires materializing each retrieved table as an in-memory DataFrame at query time. For individual document tables, typically tens to hundreds of rows, this overhead is negligible, and the execution engine (pandasql over SQLite) operates in-process without external infrastructure. At larger corpus scales, tables and images can be stored in external storage (e.g., S3 or cloud drives) as CSV and image files respectively, rather than serialized inline in vector payloads. The vector store retains only metadata paths, and content is loaded on demand when the agent invokes a tool, with caching for frequently accessed tables to avoid repeated retrieval. The primary computational cost lies in ingestion rather than retrieval: vision-language captioning and table description generation require LLM inference per element, making ingestion time proportional to the number of non-text elements in the corpus.

### Limitations

The framework's retrieval quality for images and tables depends on the fidelity of their generated descriptions. These descriptions are produced once during ingestion and stored as static text, if the vision-language model omits details present in the original image or table, those details become unretrievable through semantic search. Re-ingestion with an improved captioning model is required to recover missed information. Additionally, the agent's ability to answer table queries depends on correct SQL generation; while the framework supports error-driven retry, persistent failures on tables with ambiguous or non-standard schemas cause the agent to fall back to the natural language description, sacrificing deterministic precision. Complex table structures such as merged cells or nested headers remain challenging for the underlying structure recognition model.

## Conclusion and Future Work

This paper presents a multimodal document parsing framework for RAG systems that addresses the fundamental limitations of text-only approaches. By preserving document structure through specialized chunking strategies for text, images, and tables, the system enables retrieval of visual and structured information that traditional pipelines discard. The results demonstrate that the bottleneck in RAG performance has shifted from LLM reasoning capabilities to data quality and modality preservation, modality-specific chunking enables retrieval relevance that pure text embeddings cannot achieve, while tool-calling architectures prove essential for handling heterogeneous access patterns across text generation, image display, and SQL execution. Future work should develop comprehensive multimodal RAG benchmarks, extend the framework to video and equations, and explore domain-specific adaptations for scientific literature and technical documentation. The open-source implementation [@multimodal_parser] facilitates community adoption and experimentation.
