# Design

## 1. Architecture and key decisions

This project is a retrieval-augmented generation (RAG) document lookup system for clinical-operations information. The design prioritizes grounded answers, source traceability, deterministic refusal behavior, and a small operational footprint suitable for the assignment's corpus and time box.

### End-to-end architecture

```text
PDF / source document
        │
        ▼
┌─────────────────┐
│ Parse document   │
└────────┬────────┘
         ▼
    Chunk text
         │
         ▼
  Generate embedding
         │
         ▼
      Pinecone
         │
    ┌────┴─────────────┐
    │                  │
  vector            metadata
    │                  │
    │          ┌───────┴──────────┐
    │          │ document_id      │
    │          │ title            │
    │          │ publisher        │
    │          │ source_url       │
    │          │ chunk_id / id    │
    │          └──────────────────┘
    │
    ▼
 Retrieval
    │
    ▼
 Gemini LLM
    │
    ▼
 [S1] [S2] inline citations
    │
    ▼
 citations.py
    │
    ▼
 Actual source information
```

The ingestion implementation is document-format aware: for the current corpus, HTML is parsed with a DOM-based extractor, while the logical pipeline remains parse → chunk → embed → index. The same architecture can accept PDF or other document parsers without changing the retrieval/generation stages.

### Parsing and chunking

I use a DOM-based HTML extraction approach rather than relying on ad-hoc string splitting. After an early implementation, I added Beautiful Soup because it made the parser substantially simpler and more reliable when extracting meaningful document text from HTML. This also separates document parsing concerns from downstream chunking.

Chunks use a fixed target size with overlap (`max_chunk_tokens=700`, `chunk_overlap_tokens=100`). The tradeoff is deliberate: smaller chunks improve retrieval precision and reduce irrelevant context, while overlap protects against important information being split across chunk boundaries. I rejected both extremely small chunks, which increase index size and lose context, and very large chunks, which make retrieval less precise and increase generation context unnecessarily.

One current technical limitation is that the embedding stage processes chunks one at a time. The original implementation effectively does:

```python
embeddings = [embedder.embed_document(c.metadata.text) for c in chunks]
```

which creates one embedding request per chunk. This is inefficient under API request limits. A better implementation would batch chunks into a single embedding request where supported and pass the required output dimensionality through consistently, with explicit dimension-mismatch validation before upserting into Pinecone. This is a known optimization/technical-debt item rather than a design goal.

### Embedding model and vector store

The embedding model is `gemini-embedding-001`, with a configured 1536-dimensional output so the vectors match the Pinecone index. Pinecone was chosen as the vector store because it provides managed similarity search without requiring custom infrastructure for vector indexing, persistence, or scaling.

The tradeoff is operational simplicity versus API/vendor dependency. A local vector database could reduce external dependencies for development, but Pinecone was a better fit for the assignment because retrieval remains reproducible and closer to a production-managed architecture.

### Retrieval

The system generates one query embedding and retrieves the top 6 chunks (`retrieval_top_k=6`). The highest-scoring chunk must meet a minimum similarity threshold of `0.55`; otherwise the system returns a deterministic `unanswerable` response instead of asking the LLM to guess.

This is an intentional tradeoff. A lower threshold could improve recall but would increase irrelevant context and hallucination risk. A higher threshold would improve precision but could incorrectly label corpus-supported questions as unanswerable. For this small corpus, a conservative threshold is appropriate.

The eval harness measures expected-source hit rate and reciprocal rank (MRR), with multi-document questions requiring all expected documents for a full hit. Partial hits are tracked separately.

### Guardrails

The application has two independent gates before generation:

1. Personal-medical-advice detection. High-confidence regex patterns handle obvious individualized medication/symptom questions, while an LLM classifier is used only for ambiguous questions that contain both personal-context and medical-decision signals. The classifier is constrained to a structured boolean output.
2. Retrieval sufficiency. Weak retrieval results produce a deterministic `unanswerable` response and skip generation.

This was chosen over relying solely on a prompt instruction such as “do not give medical advice.” A hard pre-generation gate makes refusal behavior deterministic and testable.

### Prompt and citation structure

The generation prompt combines the user's question with the retrieved chunks and instructs Gemini to answer from the provided documents. The model is expected to use inline source tags such as `[S1]` and `[S2]` when supporting claims.

The backend does not trust model-generated citation metadata. `citations.py` resolves the inline tags back to the retrieved chunks and emits backend-owned source information (`document_id`, title, section, and source path). This prevents the model from inventing a source reference that was not actually retrieved.

### Streaming API

The backend returns Server-Sent Events (SSE) with structured JSON events such as `retrieval`, `status`, `token`, `citations`, and `done`. The frontend parses those events and incrementally updates the assistant message.

A frontend bug initially expected SSE `event:` fields while the backend encoded the event type inside the JSON `data:` payload. Correcting the parser to match the actual wire format restored streaming responses in the UI.

## 2. Evaluation and failure analysis

The final eval suite contains exactly 15 cases: 8 standard factual questions, 2 multi-document synthesis questions, 3 unanswerable questions, and 2 personal-medical-advice refusal questions. Status correctness and retrieval metrics are evaluated across the full suite, while LLM-as-judge is intentionally limited to a representative subset to control API usage.

The most recent full run was not a clean final benchmark because the Gemini free tier exhausted its daily `gemini-3.6-flash` generation quota after the first few successful cases. Consequently, the report showed 15 total cases but only 3 successfully evaluated cases and 12 evaluation errors. Those quota failures are infrastructure/evaluation failures, not application-quality failures, and the harness now records them separately instead of counting them as status failures.

The cases that were successfully evaluated demonstrated that the retrieval ID mapping was corrected: q01 retrieved `cdc-standard-precautions` at rank 1 and q02 retrieved `cdc-hand-hygiene` at rank 1. q16 also correctly returned `unanswerable`.

The eval also exposed several concrete issues:

- **q03–q08, q11–q12, q14–q15, q17–q18:** these were not successfully evaluated in the latest run because Gemini generation requests hit the daily free-tier quota. They should be rerun after quota availability is restored; they should not be interpreted as functional failures.
- **q12:** before the quota interruption, retrieval returned `cdc-standard-precautions`, `cdc-isolation-precautions`, and `cdc-core-infection-prevention`, but did not retrieve `cdc-hand-hygiene`, even though the expected sources were `cdc-standard-precautions` and `cdc-hand-hygiene`. This is a genuine retrieval-recall signal worth improving.
- **q01 / judge:** the first judge attempts returned prose such as “Here is the JSON...” instead of valid structured JSON. The judge was changed to use structured output and schema validation, but this should be verified again in a clean run.

Because the latest run was quota-constrained, the current report should not be presented as a final quality score. The next benchmark should be run after quota recovery and should confirm retrieval metrics, safety/refusal behavior, and the structured judge output without infrastructure errors.

## 3. What I would do with another week

### Scale to 10,000 documents

I would introduce a durable asynchronous ingestion pipeline with batching, parallelism, retry/backoff, incremental indexing, document versioning, and idempotent upserts. Embedding generation would be batched rather than one request per chunk. I would also add ingestion checkpoints so a large corpus can resume after failures instead of restarting from the beginning.

At larger scale, retrieval would likely need metadata prefilters, better duplicate handling, document-level diversification, and potentially reranking to avoid returning six chunks from the same document when evidence from multiple documents is more useful.

### Multi-tenancy

Every indexed vector would carry a tenant identifier and all retrieval operations would require a tenant-scoped filter. Authentication and authorization would determine the tenant before retrieval, preventing cross-tenant access. I would also separate per-tenant configuration and usage accounting from the shared application code.

### Cost controls

The first priority would be batching embeddings and reducing unnecessary LLM calls. The medical-advice classifier already avoids unnecessary model calls for ordinary questions; the next step would be caching query embeddings, caching stable retrieval results, setting hard token budgets, and using a smaller/cheaper model for narrow classification or judging tasks where appropriate. Usage would be tracked per request and tenant with explicit quotas.

### Latency budgets

I would define separate budgets for guardrails, embedding, retrieval, generation, and total request time. Retrieval and embedding would be instrumented independently so latency regressions can be located quickly. Streaming generation would remain in place so users receive useful tokens before the entire answer is complete, while slow upstream calls would have timeouts and retry policies rather than hanging indefinitely.

## 4. Known shortcuts and technical debt

Several shortcuts were consciously accepted for the assignment time box:

- Embeddings are still generated one chunk at a time instead of through a production-grade batch pipeline.
- The eval suite intentionally uses a small 15-case dataset rather than broad statistical coverage.
- The LLM judge evaluates a representative subset instead of every answerable case to control API usage.
- The application currently depends directly on Gemini and Pinecone rather than hiding all providers behind fully interchangeable production abstractions.
- The API has basic error handling and retry behavior but not a complete production observability stack with request tracing, metrics, dashboards, and persistent evaluation history.
- The corpus and ingestion workflow are designed for a small controlled document set rather than high-volume continuous ingestion.

These are known limitations rather than accidental omissions; they were accepted to prioritize a working end-to-end RAG system, deterministic safety behavior, evaluation infrastructure, and a usable UI within the assignment time box.
