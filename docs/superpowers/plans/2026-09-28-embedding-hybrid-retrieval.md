# Embedding and Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional multilingual embedding and RRF hybrid baselines to the existing bilingual evaluation CLI.

**Architecture:** An embedding protocol separates retrieval from model loading; a lazy sentence-transformers adapter supplies vectors only on demand. Evaluation continues to use the fixed corpus and metrics, while hybrid ranking fuses existing BM25 and embedding rankings with RRF.

**Tech Stack:** Python 3.11, sentence-transformers, PyTorch, Typer, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-28-embedding-hybrid-retrieval-design.md`

## Global Constraints

- Keep same-language-only evaluation and existing metrics unchanged.
- Use explicit UTF-8 for every new text or JSON read/write.
- No reranker, vector database, LLM, network search, or cross-language evaluation.
- Model loading must be lazy and failures must be actionable.

## Review Focus

- Fake providers must exercise all embedding tests without a downloaded model.
- A zero vector must not cause a cosine-similarity division error.
- Missing optional model dependencies must fail only when embedding is selected.
- Hybrid RRF ties must be deterministic by document path.
- `all` must retain keyword and BM25 results while adding embedding and hybrid.

---

### Task 1: Embedding provider and embedding retriever

**Files:**
- Create: `src/researchflow/evaluation/embeddings.py`
- Test: `tests/unit/test_embedding_retrieval.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: `EmbeddingProvider`, `SentenceTransformerProvider`, and `EmbeddingRetriever`.

- [ ] Write fake-provider cosine ranking, zero-vector, and lazy-import failure tests.
- [ ] Run targeted tests and confirm RED.
- [ ] Implement provider protocol, lazy configurable model adapter, and cosine retriever.
- [ ] Run targeted tests and confirm GREEN.
- [ ] Commit `feat: add optional embedding retriever`.

### Task 2: Hybrid evaluation baseline

**Files:**
- Create: `src/researchflow/evaluation/hybrid.py`
- Modify: `src/researchflow/evaluation/evaluator.py`, `src/researchflow/evaluation/__init__.py`
- Test: `tests/unit/test_embedding_retrieval.py`, `tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes: Task 1 embedding ranking and existing BM25 ranking.
- Produces: `HybridRetriever` and evaluator selectors `embedding`, `hybrid`, and `all`.

- [ ] Write failing RRF, same-language, and selector-expansion tests.
- [ ] Run targeted tests and confirm RED.
- [ ] Implement deterministic RRF and evaluator wiring with dependency injection for tests.
- [ ] Run targeted tests and confirm GREEN.
- [ ] Commit `feat: add hybrid retrieval evaluation`.

### Task 3: CLI and documentation

**Files:**
- Modify: `src/researchflow/cli.py`, `tests/test_cli.py`, `README.md`

**Interfaces:**
- Consumes: evaluator selectors from Task 2.
- Produces: `researchflow evaluate --retriever embedding|hybrid|all` and optional model option.

- [ ] Write failing CLI tests for new selectors, UTF-8 JSON, and missing-model errors.
- [ ] Implement CLI options and concise dependency/model documentation.
- [ ] Run full pytest, Ruff check, Ruff format check, and CLI smoke tests.
- [ ] Commit `feat: expose embedding and hybrid evaluation`.
