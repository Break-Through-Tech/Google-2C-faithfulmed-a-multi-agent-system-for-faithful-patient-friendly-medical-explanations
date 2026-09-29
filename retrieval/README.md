# Retrieval: build and query the lay-health index

The retrieval layer gives the agents grounded, citable lay-language text. The
pipeline has three steps:

```
raw MedlinePlus XML ──► data/processed/retrieval_chunks.jsonl ──► Chroma Cloud collection ──► retrieve()
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

## 3. Build the Chroma Cloud index

Your `.env` needs:

```
GEMINI_API_KEY=...      # https://aistudio.google.com/apikey
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
```

```bash
python -m scripts.build_retrieval_index
```

- The script embeds each chunk with `gemini-embedding-2` (768 dimensions) and
  upserts it into the shared collection `faithfulmed_lay_health_gemini2_768_v1`,
  which uses cosine distance.
- The build **resumes**: IDs already in the collection are skipped, so an
  interrupted or rate-limited run can simply be restarted. Pass
  `--rebuild-existing` to re-embed everything, for example after you change
  the embedding text format.
- The collection is shared, so one person builds it and everyone else only
  queries it. If you change the model or dimensions, use a new
  `--collection` name. The index refuses to open a collection whose model,
  dimensions or distance metric don't match.

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
| `distance` / `similarity` | cosine distance, and `1 - distance` (higher is closer) |

- Results are ordered most similar first. `r.to_dict()` gives a JSON-ready dict.
- To search only one source type, pass it:
  `retrieve("vitamin D", k=3, source_type="glossary_definition")`.
- The first call reads `.env` and connects; later calls reuse the same
  client. In tests, construct `Retriever(index, embedding_client)` with
  fakes instead (see `tests/test_retriever.py`).

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

All tests run offline because the Gemini and Chroma clients are replaced with
fakes.

## Never commit

| what | where | why |
|---|---|---|
| API keys | `.env` | secrets. Already git-ignored |
| Raw datasets | `data/raw/` | large and re-downloadable. Some have licenses that forbid redistribution |
| Processed corpus | `data/processed/retrieval_chunks.jsonl` | generated and reproducible from step 2 |
| Chroma DB / vector indexes | `chroma/`, `chroma_db/`, `.chroma/`, `*.faiss` | generated. The real index lives in Chroma Cloud |
| Generated embeddings | `embeddings/`, `*.npy`, `*.npz` | generated and large |

Everything under `data/` is ignored except `data/README.md`, the MedAESQA
methods spreadsheet and `data/eval/`, which holds the small, frozen gold sets.
Run `git status` before every commit to confirm that none of the above shows up.
