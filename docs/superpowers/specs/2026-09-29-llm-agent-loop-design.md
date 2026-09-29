# LLM Agent Decision Loop Design

## Goal

Add an optional LLM-driven planner, selector, and summarizer that completes the
existing bounded local research workflow without changing its default rule mode.

## Architecture

LLM components consume the existing `AgentState`, `ToolContext`, and structured
LLM Provider. Their Pydantic response models constrain plan steps and selected
tool calls to the existing local search, read, and save tools. The runner keeps
its current maximum-step guard and executor trace behavior.

`rule` remains the default CLI mode. `llm` mode creates LLM components from the
configured provider; unavailable configuration, invalid output after one retry,
tool failures, or empty results fall back to the equivalent rule component or
end safely. Summaries receive only successful read results, so un-read sources
cannot enter the report.

## Observability and tests

Decision events record component name, model name, and token usage without raw
LLM content, headers, or credentials. Fake providers drive unit and end-to-end
tests for the complete local workflow, bounded calls, invalid output fallback,
and source filtering. No test accesses a real provider or network.
