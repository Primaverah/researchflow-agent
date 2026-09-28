# Bilingual Retrieval Evaluation Design

## Goal

Provide an offline, deterministic baseline that compares the existing keyword
retriever with a lightweight BM25 retriever on original Chinese and English
queries. Each query is evaluated only against documents in its own language.

## Scope and constraints

- Keep all production inputs and outputs UTF-8.
- Add no dependencies, embeddings, LLMs, network calls, or evaluation framework.
- Preserve the existing research workflow and its keyword retriever behavior.
- Report Recall@1, Recall@3, Recall@5, and MRR for Chinese, English, and all
  queries combined.

## Architecture

Add an `evaluation` package with immutable evaluation cases, a compact original
two-language corpus, metric aggregation, and retriever adapters. The existing
`KeywordSearchBackend` is evaluated through the same `SearchBackend` protocol;
`Bm25SearchBackend` uses the document source and deterministic tokenization.
The BM25 implementation uses whitespace terms for English and per-character
terms for Chinese, so it remains dependency-free and supports the fixed corpus.

Evaluation filters the corpus by each case's language before ranking. A case has
a query, language, and one or more relevant document paths. Recall@k is the
fraction of cases whose relevant set appears in the first k hits; MRR is the
mean reciprocal rank of the first relevant hit. The overall result aggregates
all language cases, not an average of language averages.

## CLI and persistence

`researchflow evaluate --retriever keyword|bm25|all` evaluates the selected
retriever(s). It prints one JSON result object per selected retriever, including
per-language and overall metric blocks. `--output <path>` writes the complete
result as UTF-8 JSON; parent directories are created as necessary. Invalid
output paths or write failures return the existing CLI input/runtime error
style.

## Testing and documentation

Unit tests pin BM25 ranking and metric formulas, including rank cutoffs and
language isolation. CLI tests cover each selector value and UTF-8 JSON output.
The README gains a concise command example, metric definitions, and explicit
same-language-only limitation.
