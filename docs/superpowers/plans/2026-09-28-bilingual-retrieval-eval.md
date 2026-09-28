# Bilingual Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic Chinese/English same-language retrieval evaluation for keyword and BM25 baselines.

**Architecture:** A new evaluation package owns the original corpus, cases, language filtering, metrics, and JSON-ready result objects. The existing keyword backend remains unchanged; a dependency-free BM25 backend shares the `SearchBackend` protocol. The CLI delegates evaluation and persists UTF-8 JSON when requested.

**Tech Stack:** Python 3.11, standard library, Typer, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-28-bilingual-retrieval-eval-design.md`

## Global Constraints

- Do not add dependencies, embeddings, LLMs, network calls, or external evaluation frameworks.
- Preserve the existing research workflow and keyword retriever behavior.
- All new text and JSON reads/writes use explicit UTF-8 encoding.
- Evaluate only within each query's declared language.
- Report Recall@1, Recall@3, Recall@5, and MRR for Chinese, English, and overall.

## Review Focus

- A Chinese query must never receive an English document even if its terms match.
- A relevant document found only after rank 5 must contribute zero to recall cutoffs but its reciprocal rank to MRR.
- Empty rankings must produce zero metrics without division errors.
- `all` must aggregate every case directly, not average language summaries.
- JSON output must preserve Chinese text and be readable with explicit UTF-8.

---

### Task 1: Evaluation dataset and metrics

**Files:**
- Create: `src/researchflow/evaluation/__init__.py`
- Create: `src/researchflow/evaluation/dataset.py`
- Create: `src/researchflow/evaluation/metrics.py`
- Test: `tests/unit/test_evaluation.py`

**Interfaces:**
- Produces: `EvaluationCase`, `EvaluationDocument`, `EVALUATION_CASES`, `EVALUATION_DOCUMENTS`, and `summarize_rankings(cases, rankings)`.

- [ ] **Step 1: Write failing metric and language-isolation tests.**
- [ ] **Step 2: Run `uv run pytest tests/unit/test_evaluation.py` and confirm failure.**
- [ ] **Step 3: Implement frozen dataset models, original bilingual corpus/cases, and metric aggregation.**
- [ ] **Step 4: Re-run the targeted test and confirm it passes.**
- [ ] **Step 5: Commit `feat: add bilingual evaluation dataset and metrics`.**

### Task 2: BM25 baseline and evaluator

**Files:**
- Create: `src/researchflow/evaluation/evaluator.py`
- Modify: `src/researchflow/tools/offline/backends.py`
- Modify: `src/researchflow/tools/offline/__init__.py`
- Test: `tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes: Task 1 cases/documents and `SearchBackend`.
- Produces: `Bm25SearchBackend` and `evaluate_retriever(name)` returning JSON-ready language and overall metrics.

- [ ] **Step 1: Write failing BM25 ranking and both-retriever evaluation tests.**
- [ ] **Step 2: Run the named tests and confirm failure.**
- [ ] **Step 3: Implement deterministic tokenization, BM25 scoring, same-language filtering, and evaluator dispatch.**
- [ ] **Step 4: Re-run targeted tests and confirm they pass.**
- [ ] **Step 5: Commit `feat: add keyword and BM25 evaluation baselines`.**

### Task 3: Evaluation CLI and documentation

**Files:**
- Modify: `src/researchflow/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `evaluate_retriever(name)` from Task 2.
- Produces: `researchflow evaluate --retriever keyword|bm25|all [--output PATH]`.

- [ ] **Step 1: Write failing CLI tests for selector output, UTF-8 JSON persistence, and invalid selector handling.**
- [ ] **Step 2: Run named CLI tests and confirm failure.**
- [ ] **Step 3: Add the command and concise README usage/metric/limitation documentation.**
- [ ] **Step 4: Run targeted CLI tests and confirm they pass.**
- [ ] **Step 5: Run full pytest and Ruff checks, then commit `feat: add bilingual retrieval evaluation CLI`.**
