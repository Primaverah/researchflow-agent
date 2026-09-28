# Embedding and Hybrid Retrieval Design

## Goal

Extend the bilingual same-language evaluation baseline with local multilingual
embedding retrieval and reciprocal-rank-fusion hybrid retrieval.

## Constraints

- Preserve the fixed Chinese/English corpus, cases, language isolation, and
  Recall@1, Recall@3, Recall@5, and MRR calculations.
- Do not add reranking, vector databases, LLMs, network search, or cross-language
  evaluation.
- All text and JSON persistence remains explicit UTF-8.

## Architecture

Define an `EmbeddingProvider` protocol that embeds a sequence of strings and
returns fixed-length vectors. A lazy local provider loads a configurable
multilingual sentence-transformers model only when embedding or hybrid
evaluation is requested. Missing optional dependencies or an unavailable model
raise a dedicated, actionable retrieval error.

`EmbeddingRetriever` uses the provider to embed the query and candidate
documents, ranks same-language candidates by cosine similarity, and preserves
stable path ordering for ties. `HybridRetriever` obtains BM25 and embedding
rankings and combines their ranks with RRF using a fixed, documented constant.

The evaluator accepts `keyword`, `bm25`, `embedding`, `hybrid`, or `all`; its
existing result shape and metric aggregation are unchanged. Unit tests inject a
fake provider, so tests need no model download or network access.

## CLI and documentation

`researchflow evaluate` accepts the two new selectors and optional model
configuration. The README documents the optional embedding dependency, lazy
loading, same-language-only scope, and the hybrid method.
