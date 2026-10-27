# Retrieval: build and query the lay-health index

The retrieval layer gives the agents grounded, citable lay-language text. The
pipeline has three steps:

```
raw MedlinePlus XML ──► data/processed/retrieval_chunks.jsonl ──► Qdrant collection ──► retrieve()
   (you download)        scripts/build_retrieval_corpus.py        scripts/build_retrieval_index.py
```

Run every command from the repo root with the venv active, using `python -m`
so the `retrieval` package imports correctly.

## 1. Put the raw MedlinePlus files in place

Download from the MedlinePlus XML page (https://medlineplus.gov/xml.html):

```
data/raw/medlineplus/
├── definitions/
│   ├── fitnessdefinitions.xml
│   ├── generalhealthdefinitions.xml
│   ├── mineralsdefinitions.xml
│   ├── nutritiondefinitions.xml
│   └── vitaminsdefinitions.xml
└── mplus_topics_compressed_YYYY-MM-DD.zip   # or the uncompressed mplus_topics_YYYY-MM-DD.xml
```

- **Definitions:** all five glossary files are required. The file names must
  match `GLOSSARY_SOURCES` in `scripts/build_retrieval_corpus.py`.
- **Health Topics:** use the English "Health Topic XML" file. Either the `.xml`
  or the compressed `.zip` works.

`data/raw/` is git-ignored. Don't commit these files.

## 2. Build `retrieval_chunks.jsonl`

```bash
python -m scripts.build_retrieval_corpus \
  --definitions-dir data/raw/medlineplus/definitions \
  --health-topics data/raw/medlineplus/mplus_topics_compressed_YYYY-MM-DD.zip
```

This parses, cleans and chunks every document, validates the result (no
duplicate IDs, empty chunks, leftover HTML or oversized chunks), and writes
`data/processed/retrieval_chunks.jsonl`. Each line is one `RetrievalChunk`
(`retrieval/schema.py`). The build is deterministic, so chunk IDs stay the
same across rebuilds.

## 3. Build the Qdrant index

Your `.env` needs:

```
GEMINI_API_KEY=...      # https://aistudio.google.com/apikey
QDRANT_URL=...          # cluster URL from cloud.qdrant.io, e.g. https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=...      # database API key for that cluster
```

For a local Qdrant instead, run `docker run -p 6333:6333 qdrant/qdrant`, set
`QDRANT_URL=http://localhost:6333` and leave `QDRANT_API_KEY` empty.

**Don't create the collection in the Qdrant web UI.** The code creates it
the first time it connects, with the settings the pipeline expects:

- one unnamed dense vector with 768 dimensions and cosine distance
- the embedding model recorded in the collection metadata
- keyword payload indexes on `source_type` and `source`

A collection made in the UI, for example with named, hybrid or 3072-dimension
vectors or with multitenancy, will be rejected.

```bash
python -m scripts.build_retrieval_index
```

- The script embeds each chunk with `gemini-embedding-2` (768 dimensions) and
  upserts it into the shared collection `faithfulmed_lay_health_gemini2_768_v1`.
  Each point's payload holds the chunk text and all provenance fields.
  Point IDs are UUIDs derived from the chunk IDs.
- The build **resumes**: IDs already in the collection are skipped, so an
  interrupted or rate-limited run can simply be restarted. Pass
  `--rebuild-existing` to re-embed everything, for example after you change
  the embedding text format.
- The collection is shared, so one person builds it and everyone else only
  queries it. Teammates who only query can use a **read-only** API key. If
  you change the model or dimensions, use a new `--collection` name. The
  index refuses to open a collection whose model, dimensions or distance
  metric don't match.

## 4. Query with `retrieve()`

```python
from retrieval.retriever import retrieve

results = retrieve("bilateral lower-extremity edema", k=5)

for r in results:
    print(f"{r.similarity:.3f}  {r.title} [{r.section}]  {r.url}")
    print(r.text[:200])
```

Each `RetrievalResult` has:

| field | meaning |
|---|---|
| `text` | the chunk text (what the agents should ground on) |
| `title` | document title, e.g. `Edema` |
| `source` | publisher, e.g. `MedlinePlus` |
| `source_type` | `health_topic` or `glossary_definition` |
| `url` | canonical page to cite |
| `section` | section inside the document (`full_summary`, `Symptoms`, `definition`, …) |
| `id`, `parent_id`, `chunk_index` | stable chunk and document identifiers |
| `similarity` / `distance` | cosine similarity from Qdrant (higher is closer), and `1 - similarity` |

- Results are ordered most similar first. `r.to_dict()` gives a JSON-ready dict.
- To search only one source type, pass it:
  `retrieve("vitamin D", k=3, source_type="glossary_definition")`.
- The first call reads `.env` and connects; later calls reuse the same
  client. In tests, construct `Retriever(index, embedding_client)` with an
  in-memory Qdrant index and a fake embedder (see `tests/test_retriever.py`).

## 5. Evaluate retrieval

`data/eval/retrieval_gold_v1.jsonl` is a small, hand-labeled set of lay and
clinical queries. Each line names the document title(s) that answer the query:

```json
{"query": "bilateral lower-extremity edema", "relevant_titles": ["Edema"], "notes": "clinical phrasing"}
```

```bash
python -m scripts.evaluate_retrieval \
  --corpus data/processed/retrieval_chunks.jsonl   # optional: flags labeled titles not in the corpus
```

- **Output:** Recall@1, Recall@3 and Recall@5 (the share of queries with a
  relevant document in the top k), MRR over the top 10, and the queries that
  missed.
- **Options:** `--source-type` restricts the search to one source type, and
  `--output results.json` saves ranks for each query.
- **Changing the gold set:** relevance is judged per document, not per chunk,
  so the labels survive re-chunking. Once results have been reported against a
  version, don't edit it. Add `retrieval_gold_v2.jsonl` instead.

## Tests

```bash
python -m unittest discover -s tests -t .
```

All tests run offline. The Gemini client is replaced with a fake, and
Qdrant runs in memory (`QdrantClient(":memory:")`), so the tests exercise
real Qdrant behaviour without a server.

## Never commit

| what | where | why |
|---|---|---|
| API keys | `.env` | secrets. Already git-ignored |
| Raw datasets | `data/raw/` | large and re-downloadable. Some have licenses that forbid redistribution |
| Processed corpus | `data/processed/retrieval_chunks.jsonl` | generated and reproducible from step 2 |
| Local vector stores | `qdrant_storage/`, `chroma/`, `chroma_db/`, `.chroma/`, `*.faiss` | generated. The shared index lives in Qdrant Cloud |
| Generated embeddings | `embeddings/`, `*.npy`, `*.npz` | generated and large |

Everything under `data/` is ignored except `data/README.md`, the MedAESQA
methods spreadsheet and `data/eval/`, which holds the small, frozen gold sets.
Run `git status` before every commit to confirm that none of the above shows up.
