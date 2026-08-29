---
# Keep this title identical to the one in `myst.yml`
title: "ArcadeDB in Python: An In-Process Multi-Model Database"
abstract: |
  Some Python workflows need transactional reads and writes over records, traversals over
  relationships, and similarity search over vector embeddings. In practice these are split across separate systems (a relational or
  document store, a graph engine, and a vector index), and the data is copied between
  them and kept consistent by application code. ArcadeDB is an Apache-2.0 multi-model
  database engine, written in Java and so out of reach of a Python program that wants to
  stay in one process. We present `arcadedb-embedded`, a Python package that runs the
  engine inside the Python interpreter through JPype, over a Java runtime bundled in the
  wheel. A single `pip install` provides documents, a property graph, and vector
  search with no Java installation, and results come back as NumPy arrays, pandas
  DataFrames, or Arrow tables. We demonstrate a hybrid retrieval workflow that chains
  vector search, SQL filtering, and graph traversal over one dataset. We benchmark the engine from Python against SQLite, DuckDB,
  LadybugDB, and Chroma, and report its costs.
---

## Introduction

A retrieval-augmented question-answering system stores documents and their metadata
(records), models who-answered-what or what-cites-what (relationships), and retrieves
passages by semantic similarity (vectors), all over the same corpus. Python is popular for these workloads, but no single
Python-embeddable data store covers all three access patterns. The usual response is to assemble
a stack of SQLite or DuckDB for records, NetworkX or a graph database for
relationships, and a vector index such as FAISS, hnswlib, or Chroma for embeddings
[@duckdb2019; @sqlite; @networkx2008; @faisslib; @hnswlib; @chroma].

A stack of specialists is a reasonable default, and @datta2025composable makes the case
for assembling one from interoperable Python libraries. It has costs when all three access
patterns apply to the same data. Each system holds its own copy, with its own consistency
model, so every write has to reach all of them. A query that spans models passes identifiers from one library to another. Reproducing
an experiment means pinning several packages instead of one.

This paper describes the alternative, one embedded multi-model database that holds
documents, a property graph, and vector indexes. The engine is ArcadeDB [@arcadedb], an
Apache-2.0 Java engine that keeps all three models over one set of records. Documents are queried with
its SQL dialect, the graph with OpenCypher, and vectors through a graph index of the
Hierarchical Navigable Small World (HNSW) family. Transactional (online transaction processing, OLTP) and analytical (online analytical
processing, OLAP) queries run over the same data. ArcadeDB is ACID (atomic, consistent,
isolated, durable), and recent versions add Graph Analytical Views (GAV) that speed up
graph analytics while the base graph stays transactional.

`arcadedb-embedded` binds the Java engine through JPype [@jpype; @arcadedbpython] and
ships as wheels with a bundled Java runtime. The benchmarks report the tasks on which the
engine is slower than the specialists.

Concretely, this paper contributes:

1. **An in-process multi-model database for Python.** The binding design covers an in-process Java
   Virtual Machine (JVM) via JPype, transaction and lifecycle management, and
   NumPy/pandas/Arrow interop. Platform-agnostic Java bytecode plus a per-platform
   bundled Java Runtime Environment (JRE) yields wheels for four platforms and five Python
   versions, installable with no Java present.
2. **A hybrid workflow.** A single retrieval pipeline composes vector search, SQL
   filtering and graph traversal over one dataset in one process. A capability matrix
   shows that no single Python-embeddable alternative expresses this composition.
3. **A benchmark against embedded specialists.** A reproducible benchmark suite compares
   the engine, from Python, against embedded specialists on a real-world Q&A
   corpus, across transactional, analytical, and vector workloads. We report which lanes
   each side wins and what the unified engine costs in memory, startup, latency tails,
      disk, and result transport into Python. Engine internals and algorithms are out of
   scope.

## Background

**Embedded versus server databases.** A server database, such as PostgreSQL with pgvector
[@pgvector], Neo4j [@neo4j], or Milvus [@milvus2021], runs as its own process, and Python
reaches it over a socket. An embedded database is a full database engine
delivered as a library. It runs inside the host process and persists to ordinary files,
which fits local, single-machine work. This paper
is about that setting, so from here on it compares `arcadedb-embedded` only with other
Python-embeddable stores.

**Embedded data stores in Python.** SQLite is the standard choice for records and
transactions [@sqlite]. DuckDB brought a columnar, analytics-first embedded engine to
Python [@duckdb2019; @duckdb]. For relationships, NetworkX [@networkx2008] provides in-memory
graph analysis. LadybugDB [@ladybugdb] is an embedded graph database with a Cypher
interface and a vector index. For vectors, hnswlib provides the reference HNSW implementation [@hnsw2018;
@hnswlib], FAISS provides exact and approximate similarity search [@faisslib], and
Chroma packages HNSW behind a Python API [@chroma]. None of them provides all three
models in one embedded engine ([](#tbl-capability)).

:::{table} Capability matrix for the Python-embeddable data stores compared here; all five run in the host process (✓ native, ~ partial or through an official extension, ✗ absent). "All three" means SQL over records, a property graph, and vector search. DuckDB's HNSW index is an experimental core extension; ArcadeDB's index is JVector (Vamana graphs with an HNSW-style hierarchy). SQLite has vector search only through extensions, and neither is a graph index: `sqlite-vec` is brute force and sqlite.org's `vec1` uses IVFADC. LadybugDB stores records as typed node tables and Chroma as document-plus-metadata entries. ArcadeDB's SQL has `GROUP BY` and aggregates but no joins and no `OVER`/`PARTITION BY` window clause.
:label: tbl-capability
| Capability | SQLite | DuckDB | LadybugDB | Chroma | ArcadeDB |
|---|:--:|:--:|:--:|:--:|:--:|
| Documents / records | ✓ | ✓ | ~ | ~ | ✓ |
| Transactional SQL (OLTP) | ✓ | ~ | ✗ | ✗ | ✓ |
| Analytical SQL (OLAP) | ~ | ✓ | ✗ | ✗ | ~ |
| Property graph + traversal | ✗ | ✗ | ✓ | ✗ | ✓ |
| Cypher | ✗ | ✗ | ✓ | ✗ | ✓ |
| Graph approximate-nearest-neighbor index (HNSW family) | ✗ | ~ | ✓ | ✓ | ✓ |
| All three in one engine | ✗ | ✗ | ✗ | ✗ | ✓ |
:::



(sec-arch)=
## Architecture

`arcadedb-embedded` runs the ArcadeDB engine *inside* the Python interpreter process.
Opening a database is a function call, and queries are method calls that return Python
objects.

**In-process JVM via JPype.** On first use, the binding starts an embedded JVM inside the
Python process using JPype [@jpype]. JVM settings like heap size are configured directly
from Python, and CPython and the JVM share a single address space, so data can be passed
around without needing network calls or inter-process serialization. The engine's Java
objects are exposed through a Pythonic layer, so applications never need to interact with
JPype directly.

```python
import arcadedb_embedded as arcadedb

# One in-process engine; persists to a local directory.
with arcadedb.create_database("kb", jvm_kwargs={"heap_size": "4g"}) as db:
    db.command("sql", "CREATE DOCUMENT TYPE Post")
    db.command("sql", "CREATE PROPERTY Post.id LONG")
    db.command("sql", "CREATE INDEX ON Post (id) UNIQUE_HASH")
    with db.transaction():
        db.command("sql", "INSERT INTO Post SET id = :id, score = :s",
                   {"id": 1, "s": 42})
    rows = db.query("sql", "SELECT id, score FROM Post WHERE id = 1").to_list()
```

**One handle for all three models.** The same `db` object queries documents (SQL), the
property graph (OpenCypher), and vectors (a graph index, searched with the
`vectorNeighbors` function). The first argument of `query` selects the language. All three
models share one transaction scope and one set of record identifiers, so the hybrid
workflow below can pass results from one model into the next without leaving the process.

**NumPy, pandas, and Arrow.** Embeddings are passed as `float32` NumPy arrays through
`arcadedb.to_java_float_array(vec)`, which hands the buffer to the engine without a
Python-side element-by-element copy. Results come back through four methods on the result
set. `to_list()` gives a list of dicts, `to_dataframe()` a `pandas.DataFrame`,
`to_columns()` a dict of NumPy arrays, and `to_arrow()` a `pyarrow.Table`. The benchmark suite linked at the end of the paper uses these same calls.

```python
import numpy as np

# db as above; embeddings is an (n, 384) float32 array, one row per Question;
# the Question type and its vector index are created in the next section.
seed = np.asarray(embeddings[0], dtype=np.float32)
# vectorNeighbors(index, query vector, k, ef_search)
hits = db.query("sql",
    "SELECT id, distance FROM (SELECT expand(vectorNeighbors(?, ?, ?, ?)))",
    "Question[embedding]", arcadedb.to_java_float_array(seed), 10, 100)
df = hits.to_dataframe()                 # pandas.DataFrame

posts = db.query("sql", "SELECT id, score, title FROM Question")
cols = posts.to_columns()                # {"id": ndarray, "score": ndarray, "title": list}
table = db.query("sql", "SELECT id, score, title FROM Question").to_arrow()
                                         # pyarrow.Table; nullable INTEGER stays int64
```

pandas and pyarrow are optional dependencies, imported only when the corresponding
method is called. NumPy has no missing value for integers, so with `to_columns()` a nullable `INTEGER` column
becomes `float64` with `NaN` holes, which loses the type and any precision above $2^{53}$.
`to_arrow()` reads the same buffer and keeps its null bitmap, so the column stays `int64`.
Both read one packed columnar buffer from the engine, and the Arrow path needs no extra
Java code in the wheel. [](#sec-transport) measures the four export paths.

**Cross-platform packaging.** Bringing an engine written in another language into Python
usually means compiling a native C/C++ extension, with tooling such as scikit-build-core
[@schreiner2024skbuild]. `arcadedb-embedded` ships a JVM instead. Java bytecode is
platform-agnostic, so the engine's `.jar` files are identical everywhere and the only
platform-specific part is the Java runtime.
`arcadedb-embedded` bundles a per-platform JRE inside the wheel, trimmed with `jlink`
[@jep282] to the modules the engine needs. The build produces wheels for Linux x86-64,
Linux ARM64, macOS Apple Silicon, and Windows x86-64, for Python 3.10 to 3.14, twenty wheels
in total. `pip install arcadedb-embedded` needs no Java installation and no `JAVA_HOME`
([](#fig-arch)). Each wheel is about
70 MB, roughly four times a NumPy wheel, downloaded once at install. Every number in this
paper was measured on the wheel published on PyPI as `arcadedb-embedded` 26.8.1.

:::{figure} figures/architecture.png
:label: fig-arch
:width: 90%
The `arcadedb-embedded` architecture. A single Python process hosts a JVM (via JPype)
running the ArcadeDB engine over one local, file-backed database. The Python API
exposes documents/SQL, a property graph (OpenCypher), and vector search through
one handle, with NumPy/pandas interop. The JVM and a per-platform JRE are bundled in the
wheel, so installation is a plain `pip install` with no Java present.
:::

## Three models in one process

Each model is shown on its own below, then all three are combined in one workflow. The
data is the Cross Validated
(`stats.stackexchange.com`) Stack Exchange dump, a statistics and machine-learning Q&A site
(CC BY-SA [@crossvalidated]). Questions and answers are records, who-asked and who-answered
form a graph, and each question text has an embedding.

### Documents

The document model is created and queried with SQL, and writes run inside ACID
transactions. The snippet in [](#sec-arch) creates a type and a `UNIQUE_HASH` index for
point lookups, then runs a parameterized `INSERT` inside `db.transaction()`. Records can
be document-typed (schema-flexible) or property-typed for indexing, and both are stored in
the same database as the graph and vectors below.

Loading a corpus row by row pays the language-boundary cost once per row, so bulk ingest
goes through `insert_many`, which serializes a batch to one string and loops over it on
the Java side. As with the graph below, the columns a type does not need are never read
into Python:

```python
import pandas as pd

data = "datasets/prepared/stats.stackexchange.com"   # written by the suite's prepare.py

# read only the columns the type needs; the post body stays in the file
posts = pd.read_parquet(f"{data}/posts.parquet", columns=["id", "score", "title"])
db.insert_many("Post", posts.to_dict("records"), commit_every=10_000)
```

### Property graph

Vertex and edge types are declared with SQL. OpenCypher creates them implicitly on first
write, as in Neo4j, but we declare them so that properties can be typed and indexed. Bulk
loading goes through a `graph_batch` context manager. `create_vertices` and `new_edges` each cross into the JVM once for a
whole list and the batch commits on flush, instead of one crossing and one transaction per
element. `batch_size` is the number of edges buffered before an automatic flush, and
`create_vertices` returns the record ids, so edges reuse them instead of querying them
back. Reading the corpus column by column keeps the same discipline on the pandas side.
Relationship queries then use OpenCypher through the same handle:

```python
import pandas as pd

for t in ("Question", "Answer", "Userx"):
    db.command("sql", f"CREATE VERTEX TYPE {t}")
db.command("sql", "CREATE EDGE TYPE HAS_ANSWER")
db.command("sql", "CREATE EDGE TYPE AUTHORED_BY")

data = "datasets/prepared/stats.stackexchange.com"

# one column-selected read per table; only these columns cross into Python
questions = pd.read_parquet(f"{data}/questions.parquet", columns=["id", "score"])
answers   = pd.read_parquet(f"{data}/answers.parquet", columns=["id", "score", "author"])
users     = pd.read_parquet(f"{data}/users.parquet", columns=["id", "reputation"])
answer_author = answers.pop("author").tolist()

with db.graph_batch(batch_size=100_000) as batch:
    qrids = batch.create_vertices("Question", questions.to_dict("records"))
    arids = batch.create_vertices("Answer", answers.to_dict("records"))
    urids = batch.create_vertices("Userx", users.to_dict("records"))
    batch.new_edges(qrids, "HAS_ANSWER", arids)   # RIDs reused, one call per edge type
    batch.new_edges(arids, "AUTHORED_BY", [urids[i] for i in answer_author])

db.query("opencypher",
    "MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)-[:AUTHORED_BY]->(u:Userx) "
    "WHERE q.id = $qid RETURN a.id, a.score, u.reputation "
    "ORDER BY a.score DESC",
    {"qid": 3}).to_list()          # Cypher binds $name from a dict; SQL uses positional ?
```

For graph *analytics* over the same data, a GAV is created
(`CREATE GRAPH ANALYTICAL VIEW ... VERTEX TYPES (...) EDGE TYPES (...)`). The build runs
in the background, so the statement returns before the view is usable and
`SELECT status FROM schema:graphAnalyticalViews` is read until it reports `READY`. The view is an in-memory snapshot of the selected types, holding topology in compressed
sparse row form, two flat arrays per direction so that a vertex's neighbors are a
contiguous range rather than a pointer chase, and selected properties in typed columns.
Each committed transaction produces an immutable delta that reads merge over the snapshot,
which is how analytical traversals stay consistent with transactional writes to the base
graph. Only the
view's *definition* is persisted, so the view is rebuilt by scanning the graph each time
the database is opened, and the build cost reported below is paid on every open.

### Vectors

Embeddings are stored as a float-array property and indexed with an `LSM_VECTOR` index.
The index is a JVector graph index, built from Vamana graphs in a multi-layer hierarchy
comparable to HNSW. Search is a SQL function over the index. Parameters (`dimensions`, `similarity`,
`maxConnections`, `beamWidth` = `ef_construction`) are set on the index, and the query
supplies `ef_search`. `maxConnections` is the per-layer out-degree, while hnswlib's $M$
is half of it, because hnswlib builds its base layer at degree $2M$ and upper layers at $M$. Setting both to the same number compares a half-degree ArcadeDB
graph against a full-degree hnswlib one, so we set `maxConnections` $= 2M = 32$:

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

Given a popular "seed" question, we answer one query: *find questions similar to this
one, keep the well-scored ones, and return their best answers with the reputation of who
wrote them.* The steps are a vector search, a SQL filter, and a Cypher traversal
([](#fig-hybrid)). They pass 200 vector candidates to the SQL filter, 50 that pass the filter to
the graph traversal, and return the top 10 answers. The `IN` lists are built as text to match
the timed script, and `IN :ids` with a list parameter also works. The first two steps are kept
apart so that each stage can be timed separately. An application does not have to split
them, because `vectorNeighbors` returns whole records, so the filter can sit on the vector search
in one statement, `SELECT id, title, score FROM (SELECT expand(vectorNeighbors(...)))
WHERE score >= 1`, and only the graph traversal then needs a second call.

:::{figure} figures/hybrid_workflow.png
:label: fig-hybrid
:width: 45%
The hybrid retrieval workflow. A seed question's embedding drives a vector search, the
resulting ids flow into a SQL filter, and the surviving ids flow into a graph traversal that
reaches answers and their authors. All three steps run against one database, with identifiers passing directly between
steps.
:::

```python
# 1) VECTOR: questions semantically similar to a seed
cands = db.query("sql",
    "SELECT id, score, distance FROM "
    "(SELECT expand(vectorNeighbors(?, ?, ?, ?))) ORDER BY distance",
    "Question[embedding]", arcadedb.to_java_float_array(seed), 200, 100).to_list()

# 2) SQL: keep well-scored candidates (ids flow straight from step 1)
ids = "[" + ",".join(str(int(c["id"])) for c in cands) + "]"
filt = db.query("sql",
    f"SELECT id, title, score FROM Question WHERE id IN {ids} "
    f"AND score >= 1 ORDER BY score DESC LIMIT 50").to_list()

# 3) CYPHER: traverse to answers + answerers' reputation
fids = "[" + ",".join(str(int(r["id"])) for r in filt) + "]"
hits = db.query("opencypher",
    f"MATCH (q:Question)-[:HAS_ANSWER]->(ans:Answer)"
    f"-[:AUTHORED_BY]->(usr:Userx) "
    f"WHERE q.id IN {fids} "
    f"RETURN q.id AS qid, ans.id AS aid, ans.score AS ascore, "
    f"usr.reputation AS rep ORDER BY ascore DESC LIMIT 10").to_list()
```

Over the complete Cross Validated corpus (213,761 questions, 208,986
answers, 108,101 linked users), the workflow runs warm in ≈13 ms (median over 20 reps
after 5 warmups), of which vector ≈5.9 ms, SQL ≈5.0 ms and Cypher ≈2.0 ms. Nineteen of the twenty reps
fall between 11.4 and 16.2 ms. The twentieth, 31.8 ms, includes one SQL step that stalled
to 24 ms, the managed-runtime tail in [](#tbl-latency), and is included in the median.
Timings were measured on the same host and 8-core cap as the comparison tables below.

A GAV accelerates the Cypher traversal (measured at 2.4× on the analytical suite, see
below). The same traversal written in ArcadeDB's SQL `MATCH` answers in ≈3 ms, and both languages run the
same traversal over the same storage.

## Benchmarks

Is each model competitive enough for practical use, and what does it cost? We compare
ArcadeDB, *as reached from Python*, against the strongest Python-embeddable
specialist in each lane, namely SQLite and DuckDB (tabular), LadybugDB (graph), and
Chroma (vector).

**Protocol.** Each (lane, backend, workload) cell runs in its own pinned Docker container, one
at a time, restricted to 8 CPU cores (`--cpuset-cpus 0-7`) and a memory cap, repeated 5 times.
Engines run their shipped defaults except where stated. SQLite is configured WAL +
`synchronous=NORMAL`, which its own documentation recommends and whose durability
implications are stated with the tabular results, and ArcadeDB's JVM heap is capped at 16 GiB for this corpus,
a ceiling set by the vector lane, which does not complete at 4 GiB. The same ceiling
applies to every lane, and it is a ceiling rather than a reservation, since under it the
tabular lane peaks below 2 GiB.
Repetitions are right-skewed (GC pauses,
page-cache state), so we report the median rather than the mean [@raasveldt2018fair;
@hoefler2015scientific], and give the full [min–max] range with it. A host-side monitor
process samples each container's cgroup memory and CPU and reads the kernel peak.

The data is the Cross Validated (`stats.stackexchange.com`)
public data dump [@crossvalidated], a statistics and machine-learning Q&A corpus, comprising 425,735 posts, 345,754 users, and 1,242,391
text embeddings. Vector-index parameters are matched across engines (degree as in the
Vectors section, `ef_construction` $= 100$, `ef_search` $= 100$). We report recall@10 against an
exact ground truth. Graph OLAP for ArcadeDB uses a GAV. There is no columnar equivalent for document types,
so before the tabular OLAP arm we build a `NOTUNIQUE` index on each column the suite groups
or filters by (`post_type`, `owner_user_id`, `score`), which lets the engine answer from an
index rather than scanning the type. The index build is timed separately and excluded
from the reported query times. Every published row records the engine version, the container image digest, the cpuset and
the memory cap it ran under, and the two ablations additionally ship their run manifest and
environment file (link at the end of the paper).

All runs were on a single host, a 12th-gen Intel Core i9-12900HK with 20 logical cores, of
which 8 reached each container via `--cpuset-cpus 0-7`. It has 61 GiB usable RAM and a
Samsung 980 PRO 2 TB NVMe SSD (PCIe 4.0) holding the databases and datasets, on Linux
kernel 7.0.0 (x86-64) and Docker 29.5.3. The versions are `arcadedb-embedded` 26.8.1 (the
PyPI wheel), DuckDB 1.5.4, LadybugDB (`ladybug`) 0.18.1, Chroma 1.5.9, and SQLite 3.46.1,
the version Python's `sqlite3` module binds in the `python:3.12-slim` image the containers
are built from. ArcadeDB releases roughly monthly and `arcadedb-embedded` packages each
release, so these numbers describe 26.8.1 rather than a fixed ceiling. Embeddings are
384-dimensional and computed in the binding's repository with `all-MiniLM-L6-v2` over each
question's title and de-HTML'd body, L2-normalized, stored as `float32`.

**Workloads.** OLTP is a mixed point-operation workload issued by id and reported as
sustained throughput (ops/s). The tabular lane issues 5,000 operations (60% reads, 20%
updates, 10% inserts, 10% deletes) and the graph lane 2,000 (50% point lookups, 35% one-hop
traversals, 15% vertex inserts). OLAP is a fixed suite of analytical queries. Each query is timed
seven times and averaged, the averages are summed into a suite time, and that suite time is
reported as the median over the five repetitions, like every other cell. The tabular suite is five `GROUP BY`/aggregate/top-N
queries over the posts table. The graph suite is four traversal-and-aggregation queries
(e.g. top contributors by post count, and multi-hop path counts). The vector lane runs 1,000 held-out
nearest-neighbor queries and reports recall@10 against an exact brute-force ground truth. The
exact query text for every lane is in the public benchmark suite.

**Tabular ([](#tbl-tabular)).** Transactional throughput depends on the durability
contract. SQLite runs WAL + `synchronous=NORMAL`, which its own
documentation recommends and which fsyncs only at checkpoints. ArcadeDB runs its default
asynchronous WAL flush. Both are bounded-loss contracts. DuckDB fsyncs per commit and
exposes no relaxation, so it is the one fully-durable engine in this table. With both engines at relaxed durability, SQLite sustains ≈87,000 ops/s to ArcadeDB's
≈6,400 (≈14×) on the mixed point workload, while ArcadeDB in turn runs ≈29× DuckDB's
fully-durable ≈219. Under the *strict* pairing, per-commit fsync for both
(`arcadedb.txWalFlush=2`, and SQLite at its library defaults of a rollback journal with
`synchronous=FULL`), the two converge on the disk's fsync floor, ≈262 against ≈187 ops/s.
The SQLite strict figure is in the suite's append log rather than the
curated per-run table. Comparing each engine at its own default would mostly compare
durability settings, so both pairings are reported. On analytical SQL DuckDB answers the
suite in ≈9 ms versus ArcadeDB's ≈1,400 ms.

:::{table} Tabular lane (Cross Validated corpus): SQLite, DuckDB, ArcadeDB. OLTP is a mixed point read/insert/update workload (ops/s, higher better), OLAP an analytical aggregation suite (ms, lower better). Durability: SQLite WAL+NORMAL and ArcadeDB async WAL are bounded-loss, DuckDB fsyncs per commit. Median [min–max] over 5 reps. Peak = container memory, DB = on-disk size at the end of the OLTP rep (MiB).
:label: tbl-tabular
| Backend | OLTP ops/s | OLAP ms | Ingest s | Peak MiB | DB MiB |
|---|--:|--:|--:|--:|--:|
| SQLite | 87,150 [59,138–88,591] | 292.8 [284.4–293.9] | 0.33 [0.32–0.36] | 299 [277–304] | 20.1 |
| DuckDB | 219 [199–226] | 9.3 [9.3–9.6] | 0.38 [0.37–0.39] | 301 [296–308] | 17.3 |
| ArcadeDB | 6,416 [6,091–6,826] | 1,406.6 [1,382.1–1,432.4] | 17.52 [17.15–17.95] | 803 [763–846] | 38.5 |
:::

**Graph ([](#tbl-graph)).** LadybugDB commits with full per-commit durability by default
(each write commits on its own, at ≈135 writes/s or ≈7.4 ms per commit) and has no
relaxation knob. At their respective defaults ArcadeDB runs ≈6.1× LadybugDB's mixed-OLTP throughput (≈3,200
against ≈525 ops/s), but with ArcadeDB also at per-commit fsync the gap closes to 1.1×
(≈595 against ≈525), so the headline figure is mostly the durability setting. Reads are not
affected by that setting, and point and 1-hop reads beat LadybugDB's at both percentiles by
1.7–6.7× ([](#tbl-latency)). ArcadeDB's single worst
1-hop operation (90.3 ms) is above LadybugDB's (11.0 ms).

:::{table} Graph lane (Cross Validated corpus): LadybugDB, ArcadeDB. OLTP is neighborhood/traversal point ops (ops/s), OLAP a multi-query analytical suite (ms). ArcadeDB OLAP uses a GAV, whose build is shown separately and repeats on every database open. Durability: LadybugDB fsyncs per commit with no relaxation knob; ArcadeDB is at its async default. Median [min–max] over 5 reps. Peak = container memory, DB = on-disk size at the end of the OLTP rep (MiB).
:label: tbl-graph
| Backend | OLTP ops/s | OLAP ms | GAV build s | Peak MiB | DB MiB |
|---|--:|--:|--:|--:|--:|
| LadybugDB | 525 [467–532] | 65.7 [64.9–66.9] | n/a | 684 [675–688] | 41.4 |
| ArcadeDB | 3,212 [3,034–3,629] | 839.6 [794.7–875.1] | 1.49 [1.39–1.77] | 11,037 [10,561–11,190] | 1,840.8 |
:::

On graph analytics LadybugDB wins (≈66 ms vs ≈840 ms). Ablating the GAV (N=5 per arm, same engine, host, and harness as the table) takes
the suite from ≈840 ms to ≈2,023 ms, so the view is worth 2.4× per run against a ≈1.5 s
build. The OLTP arm is unchanged (≈3,185 against ≈3,212 ops/s), so the view affects only
the analytical queries. Because the build repeats on every open, one run of the suite in a session is slightly
slower with the view (≈2.3 s against ≈2.0 s) and two or more runs are faster. With the view, ArcadeDB is still ≈13× slower than
LadybugDB on this suite.

Bidirectional edges cost disk space and build memory. We build with ArcadeDB's default
*bidirectional* edges. These store adjacency pointers on both endpoints, so traversals run either way and
the analytical planner can start from either end. Storing both directions roughly doubles
the on-disk graph (≈1,841 against ≈800 MiB one-way), and peak build memory is ≈11 GiB
against LadybugDB's ≈684 MiB. The on-disk graph is ≈44× LadybugDB's columnar store
(≈1,841 vs ≈41 MiB).

**Vector ([](#tbl-vector)).** At matched graph degree (`maxConnections` $= 2M$), recall@10
is higher than Chroma's (0.980 vs 0.972). Build time is longer (≈546 s vs ≈319 s for
1.24 M vectors) and query latency ≈3.5× higher (≈4.2 ms vs ≈1.2 ms), still single-digit
milliseconds. Peak memory is lower than Chroma's (≈21.1 GiB vs ≈24.6 GiB, 14% less), because
the engine keeps vectors on disk while Chroma's hnswlib index holds the whole set in RAM.

:::{table} Vector lane (Cross Validated corpus, 1,242,391 vectors): Chroma, ArcadeDB at matched HNSW degree (Chroma $M=16$, ArcadeDB `maxConnections`=32, both `ef_construction`=100, `ef_search`=100). Build = insert+index (s), Query = mean latency per query in a rep (ms), recall@10 vs exact ground truth. Median [min–max] over 5 reps. Peak = container memory, DB = on-disk size (MiB).
:label: tbl-vector
| Backend | Build s | Query ms | recall@10 | Peak MiB | DB MiB |
|---|--:|--:|--:|--:|--:|
| Chroma | 319.1 [315.7–320.6] | 1.17 [1.15–1.17] | 0.972 [0.970–0.972] | 25,212 [25,173–27,002] | 2,208 |
| ArcadeDB | 546.1 [541.3–550.6] | 4.15 [3.86–4.17] | 0.980 [0.978–0.981] | 21,650 [19,680–21,791] | 2,856 |
:::

The suite also times each lifecycle phase (import, JVM init, open, schema, ingest, index
build, close). JVM initialization is ≈0.18 s and database open ≈0.12 s, one-time costs
below Chroma's Python import alone (≈0.37 s). The bulk of an ArcadeDB vector build is the
index-build phase (≈514 s of the ≈546 s total).

Graph point and 1-hop p99 (0.74 ms, 0.81 ms) beat LadybugDB's (1.24 ms, 4.34 ms), and
tabular read p99 (0.17 ms) beats DuckDB's (1.89 ms) ([](#tbl-latency)). SQLite's
memory-mapped reads at 0.007 ms are far ahead of everything in the table. The tail is
longer, though. ArcadeDB's worst single operation runs 35–146 ms across the three lanes
against 0.05–11 ms for the specialists, which matters under a hard per-request bound but
not under a p99 target.

:::{table} Per-operation query latency on the Cross Validated corpus (ms, median of per-rep summaries over 5 reps): median (p50), tail (p99), and worst single operation (max).
:label: tbl-latency
| Lane / op | Backend | p50 | p99 | max |
|---|---|--:|--:|--:|
| tabular read | SQLite | 0.004 | 0.007 | 0.05 |
| tabular read | DuckDB | 0.92 | 1.89 | 2.9 |
| tabular read | ArcadeDB | 0.06 | 0.17 | 34.9 |
| graph point | LadybugDB | 0.41 | 1.24 | 1.8 |
| graph point | ArcadeDB | 0.18 | 0.74 | 1.5 |
| graph hop | LadybugDB | 1.44 | 4.34 | 11.0 |
| graph hop | ArcadeDB | 0.21 | 0.81 | 90.3 |
| vector query | Chroma | 1.18 | 1.36 | 1.4 |
| vector query | ArcadeDB | 3.86 | 7.11 | 145.9 |
:::

**Memory.** On the transactional workload ArcadeDB's footprint (≈803 MiB) is larger than
SQLite's and DuckDB's (≈299–301 MiB), the difference being the running JVM and the
general-purpose engine. On the graph build it is larger still, from the bidirectional
property graph plus a heap growing under a generous cap. On the vector lane the
disk-backed index uses less memory than Chroma's all-in-RAM index.



(sec-transport)=
### Result transport into Python

The same engine runs the same query in all four arms, and only the way results cross
into Python differs. `iter_dicts()` converts one row at a time, `to_json_list()`
serializes one JSON string per batch for Python to parse, `to_columns()` reads a packed
columnar buffer with `numpy.frombuffer`, and `to_arrow()` wraps that same buffer as a
`pyarrow.Table`.

:::{table} Result transport, 200k-row document type, four columns (two integer, one double, one string). Median of 7 timed passes after 2 warmups, milliseconds, lower is better. The engine executes the same query in every arm.
:label: tbl-transport
| Rows | `iter_dicts` | `to_json_list` | `to_columns` | `to_arrow` |
|---|--:|--:|--:|--:|
| 10 | 2.04 | 1.70 | 0.82 | 0.59 |
| 100 | 2.66 | 1.10 | 0.93 | 0.79 |
| 1,000 | 20.12 | 2.97 | 2.28 | 1.49 |
| 10,000 | 197.29 | 25.87 | 15.87 | 14.20 |
| 100,000 | 1,978.27 | 278.22 | 135.51 | 109.03 |
:::

At 100k rows ([](#tbl-transport)) `iter_dicts` is 18× slower than `to_arrow` (1,978
against 109 ms), and the whole difference is on the Python side. At 10 rows every arm
finishes under 2.1 ms, so the choice only starts to matter as results grow.

`to_arrow()` is also the fastest arm at every size, 0.65–0.90× of `to_columns()`, because
the NumPy path decodes the string column into one Python `str` per row while Arrow wraps
the offsets-plus-blob layout the buffer already has. A purely numeric result would narrow
this.

Types also differ, as noted in [](#sec-arch). `to_arrow()` keeps nullable integer columns
typed. Querying a 3,000-row table in which every third reading is absent:

| | dtype | integer? | missing |
|---|---|---|---|
| `to_columns()` | `float64` | no | 1,000 `NaN` |
| `to_arrow()` | `int64` | yes | 1,000 nulls |

A Python user cannot change how the engine works, but they do choose the transport, and on
a large result that choice costs more than the engine does. That is why it is measured
here.

## When to use it

**Where it fits.** Local, single-machine work that touches more than one data model over
the same data, such as retrieval-augmented prototypes that need documents, graph, and embeddings,
and record- or graph-oriented applications that also want vector search beside the data.

**Costs to plan for.** Memory is the baseline above plus an enlarged heap for large vector
indexes, set from Python via `jvm_kwargs`. Startup is a one-time ≈0.3 s on the first
operation. JPype calls hold the global interpreter lock (GIL), so Python-driven
parallelism is limited as with any native extension, though engine-internal parallelism
is unaffected. Python results cost ≈1.3×
(vector search) and ≈1.6× (full-table scan) a pure-Java baseline on the same engine, the
difference being result materialization. Each wheel is ≈70 MB, downloaded once.

## Conclusion

Applications that need records, relationships, and vectors over the same data usually run
three systems and application code to keep them consistent. ArcadeDB can be packaged as a
Python wheel with a bundled JVM, and one engine then provides documents, graph, and
vectors from one process. The hybrid workflow chains vector search, SQL filtering, and
graph traversal without data leaving the engine. The comparison puts each specialist ahead on
its own ground, leaves the engine competitive on the rest, and prices the difference mainly
in memory. The ArcadeDB project wrote the engine. This paper contributes the
binding, the packaging, and the in-process workflow. The binding (Apache-2.0), the
benchmark suite, and the curated per-run results are openly available.

- Repository: <https://github.com/humemai/arcadedb-embedded-python>
- Documentation: <https://docs.humem.ai/arcadedb/>
- Benchmark suite and curated per-run results:
  <https://github.com/humemai/arcadedb-embedded-python/tree/main/benchmarks/python-bindings>

## Acknowledgements

We thank the ArcadeDB open-source contributors for the engine that this work makes available to
Python, and the JPype maintainers for the CPython–JVM bridge that the binding builds on.

Claude (Anthropic) was used to draft and revise text, and to check the paper's numbers and
citations against the benchmark artifacts. The author reviewed and verified all of it and is
responsible for the final content.
