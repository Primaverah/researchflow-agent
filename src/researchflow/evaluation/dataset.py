"""Small original same-language retrieval evaluation dataset."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """One UTF-8 document with an explicit language label."""

    path: str
    language: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A query and its relevant same-language document paths."""

    case_id: str
    query: str
    language: str
    relevant_paths: tuple[str, ...]


EVALUATION_DOCUMENTS = (
    EvaluationDocument(
        "zh-tool-calling.md",
        "zh",
        "工具调用安全",
        "工具调用需要参数校验、稳定名称和受限权限。",
    ),
    EvaluationDocument(
        "zh-tracing.md", "zh", "执行追踪", "JSONL Trace 保存每次工具执行的状态和耗时。"
    ),
    EvaluationDocument(
        "en-tool-calling.md",
        "en",
        "Tool calling safety",
        "Tool calling requires validated arguments, stable names, and safe access.",
    ),
    EvaluationDocument(
        "en-tracing.md",
        "en",
        "Execution tracing",
        "JSONL traces record tool execution status and duration.",
    ),
)

EVALUATION_CASES = (
    EvaluationCase("zh-tool", "工具调用参数校验", "zh", ("zh-tool-calling.md",)),
    EvaluationCase("zh-trace", "执行追踪 JSONL", "zh", ("zh-tracing.md",)),
    EvaluationCase(
        "en-tool", "tool calling argument validation", "en", ("en-tool-calling.md",)
    ),
    EvaluationCase("en-trace", "JSONL execution trace", "en", ("en-tracing.md",)),
)
