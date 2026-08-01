---
# Keep this title identical to the one in `myst.yml`
title: "ArcadeDB in Python: An In-Process Multi-Model Database"
abstract: |
  Scientific Python workflows increasingly mix three kinds of data access: transactional
  reads and writes over records, traversals over relationships, and similarity search over
  vector embeddings. In practice these are split across separate systems (a relational or
  document store, a graph engine, and a vector index), glued together with custom ETL and
  kept consistent by hand. This fragmentation adds operational cost and undermines
  reproducible, local experimentation. We present `arcadedb-embedded`, Python bindings that
  bring an embedded, multi-model database into the in-process scientific-Python
  ecosystem. A single `pip install` provides documents, a property graph, and HNSW vector
  search in one process, with no server to deploy. The contribution is not the database
  engine itself (ArcadeDB, a mature Apache-2.0 Java engine, which we credit) but its
  enablement from Python: a JPype binding over a bundled per-platform Java runtime, a
  NumPy/pandas-friendly API, and the in-process workflows this unlocks. We demonstrate this
  with a single hybrid retrieval workflow that combines vector search, SQL filtering, and graph
  traversal over one dataset in one transaction, a composition no single Python-embeddable
  alternative expresses today. We also characterize the embedded-from-Python experience against
  specialist embedded peers (SQLite, DuckDB, LadybugDB, Chroma), including its costs.
---

## Introduction

A growing share of scientific and applied machine-learning work is data-shaped in three
different ways at once. A retrieval-augmented question-answering system stores documents and
their metadata (records), models who-answered-what or what-cites-what (relationships), and
retrieves passages by semantic similarity (vectors). A bioinformatics or recommender pipeline
has the same character. Python is the lingua franca for all of it, but no single
Python-embeddable data store covers all three access patterns. The usual response is to
assemble a stack: SQLite or DuckDB for records, NetworkX or a graph database for
relationships, and a vector index such as FAISS, hnswlib, or Chroma for embeddings
[@duckdb2019; @sqlite; @networkx2008; @faiss2019; @hnswlib; @chroma].

Assembling specialists is a reasonable default, and for many projects it is the right one.
But it carries recurring costs that fall hardest on exactly the local, exploratory,
reproducibility-sensitive work that characterizes scientific Python. Data must be copied and
kept in sync across systems with different consistency models. A result that joins a graph
traversal to a vector neighborhood requires moving identifiers and rows between processes by
hand. And reproducing an experiment means standing up, version-pinning, and configuring
several services rather than installing one package. None of this is fundamental to the
science. It is accidental complexity introduced by the system boundaries.

This paper explores the alternative that becomes available when an embedded *multi-model*
database is reachable from Python in-process: one engine, one file-backed database, one
transaction, that natively stores documents, a property graph, and vector indexes together.
The engine we use is ArcadeDB [@arcadedb], a mature open-source (Apache-2.0) Java engine. It
keeps three data models over one set of records: documents, a property graph, and vectors.
Documents are queried with its SQL dialect, the graph with OpenCypher, and vectors through an
HNSW index. Workload type is a separate axis: both transactional (OLTP) and analytical (OLAP)
queries run over the same data. ArcadeDB is ACID and transaction-oriented, and recent versions add Graph
Analytical Views (GAV) that accelerate graph analytics without abandoning transactional
guarantees.

The engine, however, is Java. Our contribution is what makes it usable from the
scientific-Python ecosystem in-process: **`arcadedb-embedded`**, a Python binding built on
JPype [@jpype; @arcadedbpython] that runs the engine inside the Python process over a
*bundled* per-platform Java runtime, exposes a Pythonic, NumPy/pandas-friendly API, and ships as
ordinary wheels (`pip install arcadedb-embedded`, no Java installation, no server). The focus
throughout is enablement. The claim is not that the engine is the fastest at any one task (it
is not, and we show where it is not), but that having documents, graph, and vectors *in one
Python process* removes the system boundaries that fragment these workflows.

Concretely, this paper contributes:

1. **An in-process multi-model database for Python.** We describe the binding design (an
   in-process JVM via JPype, transaction and lifecycle management, and NumPy/pandas interop)
   and the packaging story: platform-agnostic Java bytecode plus a per-platform
   bundled JRE yields wheels for four platforms × five Python versions, installable with no
   Java present.
2. **A hybrid in-process workflow as the central demonstration.** A single retrieval pipeline
   composes vector search → SQL filtering → graph traversal over one dataset in one process
   and one transaction. We argue, with a capability matrix, that no single Python-embeddable
   alternative expresses this composition, and that the binding is what makes it possible from
   Python at all.
3. **An honest characterization of the embedded-from-Python experience.** Using a matched,
   resource-capped, reproducible benchmark suite, we compare the engine (*as reached from
   Python*) against specialist embedded peers across transactional, analytical, and vector
   workloads on a real-world Q&A corpus, reporting where it wins, where specialists win, and
   what the unified engine costs in memory, startup, latency tails, and disk.

This paper stays on the Python binding and the Python-side workflows and experience it enables,
not the engine's internals or algorithms.

## Background and Related Approaches

**Embedded data stores in Python.** The embedded database is a familiar shape in this community: it runs in the host process, persists to ordinary files, and needs no
server. SQLite is the archetype for records and transactions [@sqlite]. DuckDB brought a
columnar, analytics-first embedded engine to Python and is now widely used for in-process
analytical SQL [@duckdb2019]. For relationships, NetworkX [@networkx2008] provides in-memory
graph analysis, while LadybugDB [@ladybugdb], the maintained continuation of the Kùzu project
[@kuzu2023], is an embedded, analytics-oriented *graph database* with a Cypher interface and
vector indexing.
For vectors, hnswlib provides the reference HNSW implementation
[@hnsw2018; @hnswlib], FAISS provides exact and approximate similarity search [@faiss2019],
and Chroma packages HNSW behind a convenient Python API [@chroma]. Each is excellent within
its model. What none provides is *all three models in one embedded engine*. SQLite has no
graph or vector type, DuckDB has no transactional graph or Cypher, LadybugDB is graph-first
(graph and vectors, but no document/SQL OLTP store), and Chroma is vector-only.

**The composable-stack counterpoint.** A natural objection is that you do not need one engine.
You compose specialists. @datta2025composable articulates this "Python is all you need"
position: a composable, Python-native data stack assembled from best-of-breed components. We
take that argument seriously, and for warehouse-scale and team settings it is compelling. Our
claim is narrower and complementary: for *local, in-process, reproducible* workflows that
genuinely need all three access patterns over the *same* data, a single embedded multi-model
engine removes data movement and cross-system consistency work that the composable stack
leaves to the user. The two views answer different questions. Assemble for scale and
flexibility, or unify for locality and reproducibility.

**Related work.** Scientific-Python projects have built on databases before. aPhyloGeo-Covid
[@li2023aphylogeo] implements a reproducible phylogeographic workflow on Neo4j [@neo4j], in
client–server mode rather than in-process. Separately, bringing non-Python engines into the
ecosystem through bindings is well established, usually by wrapping native C/C++/Wasm code
(e.g. scikit-build-core [@schreiner2024skbuild]). This work combines those threads: a
*bundled-JVM*, multi-model engine that exposes documents, a property graph, and vectors
together, reachable in-process from Python through a single `pip install`. For the evaluation
we adopt the lightweight, comparative style of recent SciPy tool papers
[@alted2023blosc2; @sahara2025dft], with a few matched tables rather than an exhaustive
bake-off.

## Architecture: An Embedded Java Engine with Python Bindings

`arcadedb-embedded` runs the ArcadeDB engine *inside* the Python interpreter process. There
is no socket, no server lifecycle, and no separate process to manage. Opening a database is a
function call, and queries are method calls that return Python objects.

**In-process JVM via JPype.** On first use, the binding starts an embedded JVM inside the
Python process using JPype [@jpype], which bridges CPython and the JVM in a single address
space and lets Python call Java methods directly. The engine's Java objects are exposed
through a thin Pythonic facade, so application code never touches JPype directly. Because the
JVM is in-process, there is no inter-process serialization on the query path: result rows
cross the in-process boundary, not a network socket. The JVM is configured from Python via a
keyword argument at database creation, notably the heap size, which matters for large vector
indexes (discussed later):

```python
import arcadedb_embedded as arcadedb

# One in-process engine; persists to a local directory. No server.
with arcadedb.create_database("kb", jvm_kwargs={"heap_size": "4g"}) as db:
    db.command("sql", "CREATE DOCUMENT TYPE Post")
    db.command("sql", "CREATE PROPERTY Post.id LONG")
    db.command("sql", "CREATE INDEX ON Post (id) UNIQUE_HASH")
    with db.transaction():
        db.command("sql", "INSERT INTO Post SET id = :id, score = :s",
                   {"id": 1, "s": 42})
    rows = db.query("sql", "SELECT id, score FROM Post WHERE id = 1").to_list()
```

**One handle for all three models.** The same `db` object reaches documents (SQL), the property
graph (OpenCypher), and vectors (an HNSW index, searched with the `vectorNeighbors` function).
The query method's first argument selects the query language, so every model is reached through
one connection, one transaction scope, and one set of identifiers. Workload type is orthogonal:
the same query can be a transactional point operation (OLTP) or an analytical scan (OLAP). This
single-handle design is what later lets a hybrid workflow thread results from one model directly
into the next without leaving the process.

**NumPy and pandas.** Scientific Python speaks arrays and frames, and the binding fits in.
Embeddings are passed as `float32` NumPy arrays through a helper,
`arcadedb.to_java_float_array(vec)`, that hands the buffer to the engine without a Python-side
element-by-element copy; this is the path the vector and hybrid workloads use. Query results
come back as ordinary lists of dict-like rows that drop straight into a `pandas.DataFrame`, and
our workflows use pandas to load and prepare the source data. Bulk graph construction is
available through a `graph_batch` context manager that amortizes edge creation. Together these
keep the engine feeling native to a NumPy/pandas workflow rather than like a foreign service.

**Cross-platform packaging: the bundled-JRE wheel.** Distribution is an easily overlooked but
important part of the contribution. Java bytecode is platform-agnostic, so the
engine's `.jar` files are identical everywhere. The only platform-specific dependency is the
Java runtime itself. `arcadedb-embedded` therefore *bundles a per-platform JRE* inside the
wheel: the build produces wheels for four platforms (Linux x86-64, Linux ARM64, macOS Apple
Silicon, and Windows x86-64), each carrying the matching JRE, across Python 3.10–3.14, for
twenty wheels in total. The user runs `pip install arcadedb-embedded` and gets a working
multi-model database with **no Java installation and no `JAVA_HOME`**, and with no server to
deploy or operate. The JVM is an implementation detail sealed inside the package
([](#fig-arch)). The cost is wheel size, about 67 MB per wheel (Linux x86-64; the other
platforms are within a few MB), dominated by the runtime. We keep it small by trimming the JRE
with `jlink` to only the modules the engine needs and by excluding JARs the engine does not
require. That is the price of "no Java to install," paid once at install time.

About 8 MB of the wheel is an optional in-process HTTP server with a web UI, inert unless the
program calls `create_server()`: 12 JARs totalling 7.65 MB, plus 0.8 MB of JRE modules `jlink`
pulls in only for them, and about 10 ms of one-time JVM startup for the longer classpath, with
no measurable effect on resident memory. The wheels benchmarked here were built without it and
are about 62 MB; embedded behaviour is identical either way.

:::{figure} figures/architecture.png
:label: fig-arch
:width: 90%
The `arcadedb-embedded` architecture. A single Python process hosts an in-process JVM (via
JPype) running the ArcadeDB engine over one local, file-backed database. The Python API
exposes documents/SQL, a property graph (OpenCypher), and HNSW vector search through
one handle, with NumPy/pandas interop. The JVM and a per-platform JRE are bundled in the
wheel, so installation is a plain `pip install` with no Java present.
:::

## One Runtime, Three Models, and a Hybrid Workflow

This is a Python-enablement paper, so its focus is what the binding *unlocks*: the
ability to express a workflow that spans documents, graph, and vectors over one dataset, in
one process, in one transaction, from Python. We first show each model briefly, then compose
them into the unified workflow this paper centers on. The data is a public Stack Exchange Q&A
corpus: Cross Validated (`stats.stackexchange.com`), the statistics and machine-learning site
(CC BY-SA [@crossvalidated]). It is a natural knowledge-base / semantic-search setting, where
questions and answers are records, who-asked and who-answered form a graph, and question text
carries embeddings.

### Documents, via SQL

The document model is created and queried with SQL. Schema creation and ACID writes are issued
as SQL from Python. On the workload axis, the engine is strongest at transactional access:
indexed point reads, inserts, and updates are its strong suit. The
snippet in the Architecture section already shows the shape, with `CREATE ... TYPE`, a
`UNIQUE_HASH` index for fast point lookups, and parameterized `INSERT` inside
`db.transaction()`. Records can be document-typed (schema-flexible) or property-typed for
indexing, and both live in the same database as the graph and vectors below.

### The property graph, via Cypher

Vertices and edges are first-class. Once `Question`, `Answer`, and `Userx` vertex types and
`HAS_ANSWER` / `AUTHORED_BY` edge types exist, relationship queries use OpenCypher through the
same handle:

```python
db.query("opencypher",
    "MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)-[:AUTHORED_BY]->(u:Userx) "
    "WHERE q.id = $qid RETURN a.id, a.score, u.reputation "
    "ORDER BY a.score DESC", {"qid": 3}).to_list()
```

For graph *analytics* over the same data, a Graph Analytical View can be built once
(`CREATE GRAPH ANALYTICAL VIEW ...`) and polled until ready. Subsequent analytical traversals
run against the accelerated view, while transactional writes continue against the base graph.
This is the "strong transactional core, *and* graph OLAP via GAV" story we quantify in the comparison.

### Vectors, via an HNSW index

Embeddings are stored as a float-array property and indexed with an HNSW (`LSM_VECTOR`) index.
Search is a SQL function over the index. Parameters (`dimensions`, `similarity`,
`maxConnections`, `beamWidth` = `ef_construction`) are set on the index, and the
query supplies `ef_search`. One mapping deserves care: `maxConnections` is the
per-layer out-degree of the underlying Vamana graph, not hnswlib's $M$, and
hnswlib builds its base layer at $2M$. Setting both to the same number therefore
compares a half-degree ArcadeDB graph against a full-degree hnswlib one. We set
`maxConnections` $= 2M = 32$ to match the two by effect rather than by name:

```python
db.command("sql", "CREATE INDEX ON Question (embedding) LSM_VECTOR "
    'METADATA { "dimensions": 384, "similarity": "COSINE", '
    '"maxConnections": 32, "beamWidth": 100 }')

hits = db.query("sql",
    "SELECT id, score, distance FROM "
    "(SELECT expand(vectorNeighbors(?, ?, ?, ?))) ORDER BY distance",
    "Question[embedding]", arcadedb.to_java_float_array(seed), 200, 100).to_list()
```

### The hybrid workflow

The interesting case is composing all three over the same data, in one process, with no data leaving the
engine. Given a popular "seed" question, we answer: *find questions semantically similar to
this one, keep the well-scored ones, and return their best answers together with the
reputation of who wrote them.* That is vector search, then a relational filter, then a graph
traversal: three data models and three query languages (a vector function in SQL, plain SQL, OpenCypher), one database, one transaction ([](#fig-hybrid)).

```python
# 1) VECTOR — questions semantically similar to a seed
cands = db.query("sql",
    "SELECT id, score, distance FROM "
    "(SELECT expand(vectorNeighbors(?, ?, ?, ?))) ORDER BY distance",
    "Question[embedding]", seed, 200, 100).to_list()

# 2) SQL — keep well-scored candidates (ids flow straight from step 1)
ids = "[" + ",".join(str(int(c["id"])) for c in cands) + "]"
filt = db.query("sql",
    f"SELECT id, title, score FROM Question WHERE id IN {ids} "
    f"AND score >= 1 ORDER BY score DESC LIMIT 50").to_list()

# 3) CYPHER — traverse to answers + answerers' reputation
fids = "[" + ",".join(str(int(r["id"])) for r in filt) + "]"
hits = db.query("cypher",
    f"MATCH (q:Question)-[:HAS_ANSWER]->(ans:Answer)"
    f"-[:AUTHORED_BY]->(usr:Userx) "
    f"WHERE q.id IN {fids} "
    f"RETURN q.id AS qid, ans.id AS aid, ans.score AS ascore, "
    f"usr.reputation AS rep ORDER BY ascore DESC LIMIT 10").to_list()
```

The seams are what matter. The vector step's output ids feed the SQL `WHERE ... IN` directly,
and the SQL step's surviving ids feed the graph traversal directly, with no serialization, no
copying rows between processes, no second system to keep consistent, and no ETL. Over the
complete set of Cross Validated questions and answers (all 213,761 questions and 208,986
answers, with the 108,101 users linked to them), the end-to-end workflow runs warm in
**≈16 ms** (vector ≈10 ms, SQL ≈5 ms, Cypher ≈1.4 ms; median over 20 reps after 5
warmups, range 15–17 ms). The graph is stored with ArcadeDB's default bidirectional
edges, the Cypher traversal is accelerated by a Graph Analytical View (measured at 2.9×
on the analytical suite, see below), and the same
traversal through ArcadeDB's native SQL `MATCH` surface answers in ≈3 ms: both surfaces
run the same traversal over the same storage, within about 2× of each other, with neither
a translation layer bolted onto the other. (Preparing this workflow surfaced two
engine issues that we reported upstream and that were each fixed within days — a Cypher
planner gap that made this traversal ≈143 ms, and a vector-index maintenance bug that
inflated the vector step to ≈100 ms and the bulk load by ≈45× — an instance of the
release cadence noted at the end of the benchmark section.) All of it runs in a single
process after a one-time bulk load. The
three steps pass 200 vector candidates to the SQL filter, 50 survivors to the graph traversal,
and return the top 10 answers. Timings were measured on the same host and 8-core cap as the
comparison tables below.

What makes this a *contribution*, not merely a convenience, is that no single
Python-embeddable alternative can express it. [](#tbl-capability) lays out the capability
matrix. SQLite has no graph or vector model, DuckDB has no transactional graph or Cypher,
LadybugDB has graph and vectors but no document/SQL OLTP store, and Chroma is vector-only. A
composable stack can of course *reproduce* the result, but only by running separate systems (a
document/relational store, a graph engine, and a vector index), materializing
intermediate results across process boundaries, and writing the glue to move and reconcile
identifiers between them. Here it is one `pip install`, one process, one transaction. It is the unified alternative to the composable-stack position of @datta2025composable.

:::{figure} figures/hybrid_workflow.png
:label: fig-hybrid
:width: 45%
The hybrid retrieval workflow. A seed question's embedding drives an HNSW vector search, the
resulting ids flow into a SQL filter, and the surviving ids flow into a graph traversal that
reaches answers and their authors. All three steps run against one in-process database in a
single transaction, with identifiers passing directly between steps and no ETL or cross-system
movement.
:::

:::{table} Capability matrix for Python-embeddable data stores. Only the multi-model engine covers documents/OLTP, property-graph traversal, and vector search in one process, which is what the hybrid workflow requires (✓ native, ~ partial/limited, ✗ absent).
:label: tbl-capability
| Capability | SQLite | DuckDB | LadybugDB | Chroma | ArcadeDB |
|---|:--:|:--:|:--:|:--:|:--:|
| Documents / records | ✓ | ✓ | ✗ | ✗ | ✓ |
| Transactional SQL (OLTP) | ✓ | ~ | ✗ | ✗ | ✓ |
| Analytical SQL (OLAP) | ~ | ✓ | ✗ | ✗ | ✓ |
| Property graph + traversal | ✗ | ✗ | ✓ | ✗ | ✓ |
| Cypher | ✗ | ✗ | ✓ | ✗ | ✓ |
| HNSW vector search | ✗ | ✗ | ✓ | ✓ | ✓ |
| All three in one engine | ✗ | ✗ | ✗ | ✗ | ✓ |
| In-process, no server | ✓ | ✓ | ✓ | ✓ | ✓ |
:::

## Comparison: The Embedded-from-Python Experience

The hybrid workflow shows what the binding *enables*. This section is a supporting
credibility check: *is the unified engine competitive enough, per model, to be a practical
choice, and what does it cost?* It is deliberately not a database bake-off. We compare
ArcadeDB, *as reached from Python*, against the strongest
Python-embeddable specialist in each lane: SQLite and DuckDB (tabular), LadybugDB (graph), and
Chroma (vector).

**Protocol.** Each (lane, backend, workload) cell runs in its own pinned Docker container, one
at a time, restricted to 8 CPU cores (`--cpuset-cpus 0-7`) and a memory cap, repeated 5 times.
Engines run their shipped defaults except where stated: SQLite is configured WAL +
`synchronous=NORMAL` (its documented recommendation; the durability implications are stated
with the tabular results), and ArcadeDB's JVM heap is pinned per tier.
We report the median with the full [min–max] range: database benchmark repetitions are
right-skewed (GC pauses, page-cache state), so the median resists outliers while the range
exposes them, following established guidance for performance reporting
[@raasveldt2018fair; @hoefler2015scientific]. A host-side sidecar samples each container's cgroup memory
and CPU and reads the kernel peak. The data is the **Cross Validated** (`stats.stackexchange.com`)
public data dump [@crossvalidated], a statistics and machine-learning Q&A corpus and the most
SciPy-relevant Stack Exchange site, comprising 425,735 posts, 345,754 users, and 1,242,391
text embeddings. Vector lanes use **matched HNSW parameters across engines**, with graph degree
matched by effect rather than by name (Chroma $M = 16$, so a base layer of 32; ArcadeDB
`maxConnections` $= 32$), plus `ef_construction` $= 100$ and `ef_search` $= 100$, and report
recall@10 against an exact ground truth. Graph OLAP for ArcadeDB uses a GAV, and tabular OLAP for ArcadeDB uses secondary
indexes. All versions, image digests, host details, and per-run memory time-series are
captured in a manifest for reproducibility. Runs were executed on a single host: a 12th-gen
Intel Core i9-12900HK (20 logical cores, of which 8 were exposed to each container via
`--cpuset-cpus 0-7`), 61 GiB usable RAM, a Samsung 980 PRO 2 TB NVMe SSD (PCIe 4.0) holding
the databases and datasets, Linux kernel 7.0.0 (x86-64), and Docker 29.5.3. Engine and
competitor versions were pinned per lane, since the lanes were measured as the engine fixes
this work produced landed: ArcadeDB (`arcadedb-embedded`) 26.8.1.dev2 for the graph lane,
26.8.1.dev3 for the tabular lane and the hybrid workflow of the previous section, and
26.8.1.dev20 for the vector lane, which was re-measured last at matched graph degree. Re-running
the graph lane on 26.8.1.dev20 reproduces its published numbers within run-to-run spread (OLAP
796.4 ms vs 796.3, GAV build 1.40 s vs 1.43, OLTP 3,762 ops/s vs 3,929), so the version spread
is a reporting detail rather than a confound. Those ArcadeDB versions are the
pins in `experiments/build_images.sh` and are authoritative; the `lib_version` column in
`results/runs.csv` reads `26.8.1.dev0` for every ArcadeDB row, because
`arcadedb_embedded.__version__` was baked at build time and did not track the wheel until
26.8.1.dev21, which is later than every wheel used here. DuckDB 1.5.4, SQLite
3.46.1, LadybugDB (`ladybug`) 0.18.1, Chroma 1.5.9. Embeddings are 384-dimensional
(`all-MiniLM-L6-v2`).

**Workloads.** OLTP is a mixed point-operation workload issued by id and reported as sustained
throughput (ops/s): 5,000 operations for the tabular lane (60% reads, 20% updates, 10%
inserts, 10% deletes) and 2,000 for the graph lane (50% point lookups, 35% one-hop traversals,
15% vertex inserts). OLAP is a fixed suite of analytical queries, each timed individually (mean
of 7 runs) and summed: five `GROUP BY`/aggregate/top-N queries over the posts table for the
tabular lane, and four traversal-and-aggregation queries for the graph lane (e.g. top
contributors by post count, and multi-hop path counts). The vector lane runs 1,000 held-out
nearest-neighbor queries and reports recall@10 against an exact brute-force ground truth. The
exact query text for every lane is in the public benchmark suite.

**Tabular ([](#tbl-tabular)).** Transactional throughput comparisons are durability-sensitive,
so we state the contracts first. SQLite runs WAL + `synchronous=NORMAL` — its own
documentation's recommendation, fsyncing at checkpoints rather than per commit — and ArcadeDB
runs its default asynchronous WAL flush; both are bounded-loss contracts. DuckDB fsyncs per
commit and exposes no relaxation, so it is the one fully-durable engine in this table. At this
matched-relaxed operating point the in-process C library dominates the mixed point workload:
SQLite sustains ≈87,000 ops/s to ArcadeDB's ≈5,800 (≈15×), while ArcadeDB in turn runs ≈26×
DuckDB's fully-durable ≈219. Under the *strict* pairing — per-commit fsync for both, measured
as an ablation (`arcadedb.txWalFlush=2` vs `synchronous=FULL`) — the two converge to the
disk's fsync floor: ≈242 vs ≈187 ops/s. An earlier version of this benchmark ran SQLite at
library defaults (rollback journal, `synchronous=FULL`) against ArcadeDB's async default,
which inflated ArcadeDB's apparent advantage to 24–31×; we consider the corrected numbers the
honest ones and flag the asymmetry so others avoid it. On analytical SQL the specialists win
decisively: DuckDB's columnar engine answers the analytics suite in ≈9 ms versus ArcadeDB's
≈1,300 ms. The summary is unglamorous and useful: for single-model point work an embedded
C library is untouchable; ArcadeDB's transactional throughput is ample for application
workloads and comes attached to the graph and vector models that the rest of this paper is
about.

:::{table} Tabular lane (Cross Validated corpus): SQLite, DuckDB, ArcadeDB. OLTP is a mixed point read/insert/update workload (ops/s, higher is better). OLAP is an analytical aggregation suite (ms, lower is better). Values are median [min–max] over 5 reps. Durability contracts: SQLite WAL+NORMAL and ArcadeDB async WAL (both bounded-loss); DuckDB per-commit fsync (fully durable). At matched per-commit fsync (ablation), ArcadeDB ≈242 ops/s vs SQLite ≈187. On-disk DB size is deterministic across reps (no range). Peak = container memory, DB = on-disk size after load (MiB).
:label: tbl-tabular
| Backend | OLTP ops/s | OLAP ms | Ingest s | Peak MiB | DB MiB |
|---|--:|--:|--:|--:|--:|
| SQLite | 87,150 [59,138–88,591] | 292.8 [284.4–293.9] | 0.33 [0.32–0.36] | 299 [277–304] | 20.1 |
| DuckDB | 219 [199–226] | 9.3 [9.3–9.6] | 0.38 [0.37–0.39] | 301 [296–308] | 17.3 |
| ArcadeDB | 5,786 [5,067–5,892] | 1,299.6 [1,276.3–1,317.6] | 13.98 [13.76–14.42] | 852 [785–950] | 38.3 |
:::

**Graph ([](#tbl-graph)).** The durability lens matters here too: LadybugDB commits with
full per-commit durability by default (we measured its single-transaction writes at the same
≈110/s fsync floor as everyone else's strict mode) and exposes no relaxation knob. At the
engines' respective defaults ArcadeDB runs ≈7.5× LadybugDB's mixed-OLTP throughput (≈3,900
vs ≈525 ops/s) — but at ArcadeDB's matched-strict ablation the suite converges to near
parity (≈539 vs ≈525), so the headline gap is a difference in default durability contracts
at least as much as in engines. Where ArcadeDB's advantage is contract-independent is
per-operation read latency: its point and 1-hop reads beat LadybugDB's at both reported
percentiles ([](#tbl-latency)), by 2.8–9.0× across both operations. The 1-hop
*maximum* inverts (74.7 ms against 11.0), which is one worst-case observation rather than
a percentile, but we report it because it is the shape a JVM engine gives you: better
typical latency, a longer worst case. On graph analytics the analytics-oriented LadybugDB wins
(≈66 ms vs ≈800 ms). The GAV
is worth it on its own terms: ablating it on this corpus (N=5 per arm) takes the analytical
suite from ≈165 ms to ≈476 ms at the median, so the view is worth **2.9×** for a one-time
≈1.4 s build. It narrows rather than closes the gap to a dedicated analytical
graph engine. The costs are space and build memory. We build with ArcadeDB's default
*bidirectional* edges, which store adjacency pointers on both endpoints so traversals run
either way and the analytical planner can start from either end; this is the out-of-the-box
behavior and the fair one to measure, but it roughly doubles the on-disk graph (≈1,774 vs a
one-way ≈800 MiB) and raises peak build memory (a JVM growing its heap under a generous cap,
against LadybugDB's ≈684 MiB C++ footprint). The on-disk graph is ≈43× LadybugDB's columnar
store (≈1,774 vs ≈41 MiB). Again, the summary is complementarity: transactional graph writes and
point traversals here, heavy graph analytics on a specialist.

:::{table} Graph lane (Cross Validated corpus): LadybugDB, ArcadeDB. OLTP is neighborhood/traversal point ops (ops/s). OLAP is a multi-query analytical suite (ms). ArcadeDB OLAP uses a Graph Analytical View (one-time build shown); ablating it raises the suite median from ≈165 ms to ≈476 ms, so the view is worth 2.9× (N=5 per arm). Values are median [min–max] over 5 reps. Durability contracts: LadybugDB fsyncs per commit (no relaxation knob); ArcadeDB shown at its async default — at matched per-commit fsync (ablation) its suite throughput is ≈539 ops/s, near parity with LadybugDB. On-disk DB size is deterministic across reps (no range). Peak = container memory, DB = on-disk size (MiB).
:label: tbl-graph
| Backend | OLTP ops/s | OLAP ms | GAV build s | Peak MiB | DB MiB |
|---|--:|--:|--:|--:|--:|
| LadybugDB | 525 [467–532] | 65.7 [64.9–66.9] | — | 684 [675–688] | 41.4 |
| ArcadeDB | 3,929 [3,466–4,422] | 796.3 [781.3–845.6] | 1.43 [1.38–1.66] | 11,458 [10,302–11,663] | 1,774.1 |
:::

**Vector ([](#tbl-vector)).** With graph degree matched by effect rather than by name
(`maxConnections` $= 2M$), ArcadeDB is *competitive while being multi-model*, and the
comparison lands differently than a name-matched one would. Recall@10 is **higher** than
Chroma's (0.979 vs 0.973), so ArcadeDB is the more exact of the two at this operating point,
not the more approximate. The costs are build time (≈546 s vs ≈318 s for 1.24 M vectors) and
query latency, ≈3.4× higher (≈3.9 ms vs ≈1.2 ms) but still single-digit milliseconds. The
notable result is memory: ArcadeDB's *peak* memory is **lower** than Chroma's (≈15.2 GiB vs
≈24.6 GiB, a 38% reduction), because the engine keeps vectors on disk rather than holding the
entire set resident in RAM as the pure-Python HNSW path does. The trade is deliberate: you
give up some query latency relative to a dedicated vector store and get vectors that live in
the same engine as your documents and graph.

The degree correction is worth stating plainly because it moved a conclusion. Our earlier
name-matched configuration reported recall 0.951 against Chroma's 0.971 and read as a small
quality deficit. It was an artifact of comparing a degree-16 graph against a degree-32 one.
At matched degree the deficit reverses, and the build-time cost of the denser graph (≈353 s
to ≈546 s) is the price of that recall. The mapping is now documented upstream.

:::{table} Vector lane (Cross Validated corpus, 1,242,391 vectors): Chroma, ArcadeDB, matched HNSW graph degree (Chroma $M=16$, ArcadeDB `maxConnections`=32, both `ef_construction`=100, `ef_search`=100). Build = insert+index (s). Query = mean latency per query within a rep (ms). recall@10 vs exact ground truth. Values are median [min–max] over 5 reps. On-disk DB size is deterministic across reps (no range). Peak = container memory, DB = on-disk size (MiB).
:label: tbl-vector
| Backend | Build s | Query ms | recall@10 | Peak MiB | DB MiB |
|---|--:|--:|--:|--:|--:|
| Chroma | 317.9 [317.2–320.1] | 1.16 [1.15–1.19] | 0.973 [0.972–0.974] | 25,176 [25,168–25,186] | 2,208 |
| ArcadeDB | 545.5 [542.0–547.5] | 3.93 [3.77–4.00] | 0.979 [0.977–0.979] | 15,593 [14,841–17,480] | 2,857 |
:::

Beyond the headline throughput and latency numbers, the benchmark suite isolates each
lifecycle phase (import, JVM init, open, schema, ingest, index build, close), and two
cross-cutting results bear on concerns a JVM-backed binding raises. First, **JVM startup is
negligible**: isolated JVM initialization is ≈0.16 s and database open ≈0.12 s, a one-time,
sub-second cost amortized over any real session, and in fact *smaller* than Chroma's Python
import alone (≈0.37 s). The bulk of an ArcadeDB vector build is the HNSW index phase (≈514 s of
the ≈546 s total), not startup. Second, ArcadeDB's **typical latencies are excellent**
([](#tbl-latency)): graph point and 1-hop p99 (0.45 ms, 0.56 ms) beat LadybugDB's (1.24 ms,
4.34 ms), and tabular read p99 (0.17 ms) beats DuckDB's (1.89 ms) — though not WAL-mode
SQLite's memory-mapped reads (0.007 ms), which nothing in this table touches. But the JVM
shows a **tail**: occasional max latencies of tens of milliseconds (e.g. a 34–75 ms outlier
under GC), the cost
of a managed runtime. For interactive and batch analytics this tail is irrelevant. For hard
real-time serving it matters.

:::{table} Per-operation query latency on the Cross Validated corpus (ms, median of per-rep summaries over 5 reps): median (p50), tail (p99), and worst single operation (max).
:label: tbl-latency
| Lane / op | Backend | p50 | p99 | max |
|---|---|--:|--:|--:|
| vector query | Chroma | 1.16 | 1.36 | 1.6 |
| vector query | ArcadeDB | 3.79 | 6.93 | 8.8 |
| tabular read | SQLite | 0.004 | 0.007 | 0.1 |
| tabular read | DuckDB | 0.93 | 1.89 | 2.9 |
| tabular read | ArcadeDB | 0.06 | 0.17 | 33.9 |
| graph point | LadybugDB | 0.41 | 1.24 | 1.8 |
| graph point | ArcadeDB | 0.14 | 0.45 | 1.5 |
| graph hop | LadybugDB | 1.44 | 4.34 | 11.0 |
| graph hop | ArcadeDB | 0.16 | 0.56 | 74.7 |
:::

**Memory is the cost.** On the transactional workload ArcadeDB's footprint is larger than the
lean C-based specialists (≈852 MiB vs ≈299–301 MiB), the cost of a running JVM and a
general-purpose engine, and on the graph build it is larger still (the bidirectional property
graph plus a heap growing under a generous cap, discussed above). The vector lane is the
exception that proves the rule: its disk-backed index makes it *more* memory-frugal than an
all-in-RAM vector library at scale. We report these plainly so practitioners can decide. The unified, in-process engine is
not free, and for memory-constrained single-purpose tasks a specialist may be the better pick.

Taken together, the comparison supports a measured claim — more measured than our own first
draft of it. ArcadeDB-from-Python has *excellent point-operation latencies* (graph reads beat
the graph specialist at p50 and p99, with a longer worst case; tabular reads beat DuckDB), *ample transactional
throughput under either durability contract* (converging with the specialists at the fsync
floor when strict), is *competitive on vector search at matched graph degree, trading query
latency for slightly higher recall and 38% lower peak memory*, is *outclassed by specialists on heavy analytics and by in-process C on raw
single-model point throughput*, and is *clear about its memory cost*. None of these numbers
alone justifies a multi-model engine; the case is the previous section's: three models, one
process, one transaction — with per-model performance that is good enough to keep everything
in one place.

These results are a snapshot of the versions pinned in the protocol. ArcadeDB is in active
development and releases on a roughly monthly cadence (its version scheme is year-and-month),
and `arcadedb-embedded` packages each upstream release, so the gaps reported here (analytical
workloads in particular) are a moving target rather than a fixed property of the engine or the
binding.

## Discussion: Tradeoffs and When to Use

**When this fits.** The natural setting is local, in-process, reproducible work that genuinely
touches more than one data model over the same data: retrieval-augmented prototypes that need
documents, graph, and embeddings; exploratory analyses where standing up three services is
friction; teaching and reproducible artifacts where "one `pip install`" matters; and
transactional, record- or graph-oriented applications that also want vector search beside the
data. The advantages are operational (no server, one dependency), reproducibility
(one pinned package, one file-backed database), and composition (cross-model workflows with no
ETL).

**When it does not.** This is a single-node embedded engine, not a distributed warehouse. The
case for reaching past it is narrow, and specific to *one* model and workload dominating at
scale: large columnar/tabular analytical scans belong in DuckDB or a warehouse, and pure
high-QPS vector serving belongs in a dedicated vector store whose all-in-RAM index buys the
lowest latency. Note what this list does *not* include: moderate analytics, graph analytics
(GAV makes these workable), and vector search at scale are all well within reach. Short of a
single lane dominating, the unified engine is the right default whenever a workflow needs
several models over the same data in one process.

**Costs to plan for.** Four, briefly. *Memory*: a JVM plus a general-purpose engine carries a
higher baseline than C-based specialists, and large vector indexes need an enlarged heap (set
from Python via `jvm_kwargs`). *Startup*: the in-process JVM adds a one-time ≈0.3 s to the
first operation (≈0.16 s init plus ≈0.13 s open), amortized over the session and negligible
outside very short scripts. *Concurrency*: JPype calls cross the CPython↔JVM boundary under the
GIL, so Python-driven parallel query throughput is constrained as with any native extension,
though process-level and engine-internal parallelism remain. *Binding overhead*: the engine
itself runs at Java speed from Python (the JPype call is a direct method invocation); the
measurable cost is materializing results into Python objects, which the binding's bulk paths
keep small — ≈1.1× a pure-Java baseline for vector search and ≈1.6× for full-table scans on
this host — so the boundary tax is paid per batch, not per row. *Packaging*: a `jlink`-trimmed JRE
still makes each wheel ≈67 MB, paid once at install.

**Maturity and scope.** Beyond the three models shown, the engine and binding cover more than
this paper exercises (additional data types and query surfaces, batch import paths, and an
async execution option), and the project ships an extensive example and test suite. We
deliberately keep this paper to the document/graph/vector core and the hybrid workflow that
motivates it.

## Conclusion

Scientific Python increasingly needs records, relationships, and vectors over the same data,
and today that usually means three systems and the glue between them. We have shown that an
embedded, multi-model engine, ArcadeDB, can be made to behave like a native
Python package, `arcadedb-embedded`, via an in-process JVM over a bundled per-platform JRE,
and that doing so collapses a multi-system workflow into one `pip install`, one process, and
one transaction. The mental model we hope readers take away is simple: *when a local workflow
needs more than one data model over the same data, you can have one embedded engine instead of
a stack.* Our hybrid workflow shows this concretely (vector → SQL → graph with no ETL),
and our matched comparison lays out the tradeoffs: strong on the transactional core,
competitive on vectors at matched graph degree, outclassed by specialists on heavy analytics, and
carrying a real but bounded memory cost. The contribution is the Python enablement, not the
engine. We credit ArcadeDB and its authors for the latter. The binding (Apache-2.0), the full
benchmark suite, the datasets, and the run manifest are openly available, so every result here
is reproducible end to end.

- Repository: <https://github.com/humemai/arcadedb-embedded-python>
- Documentation: <https://docs.humem.ai/arcadedb/>

## Acknowledgements

We thank the ArcadeDB open-source contributors for the engine that this work makes available to
Python, and the JPype maintainers for the CPython–JVM bridge that the binding builds on.

Portions of this work were assisted using generative AI tools (Claude and GPT-based
assistants) for drafting and refining language and for editorial suggestions. All outputs were
reviewed, verified, and revised by the author, who takes full responsibility for the accuracy
and integrity of the final content.
