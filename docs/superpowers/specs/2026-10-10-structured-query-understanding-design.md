# 结构化问题理解与检索质量设计

## 背景与目标

ResearchFlow 已有持久会话、确定性指代改写、网页检索、来源读取与证据门控，但目前仍会出现两个根因不同的失败：

- 对话式问题的主体没有稳定地传入检索。例如“C 和 C++ 的区别是什么”后追问“它们的发展历史”，检索将“发展历史”当作独立主题，可能接受马克思主义或 GIS 页面。
- 时效性问题的候选与必需事实不匹配。例如“今年的诺贝尔奖获得者有哪些”得到 Microsoft 页面；在读取到不相关页面后又把“来源数量”误当成“证据充分”。

本次采用方案 B：在检索前增加**受约束的 LLM 结构化问题理解**，但不让 LLM 自由决定事实、跳过来源验证或取代可解释的排序与门控。目标是让检索围绕当前问题的主体、意图、所需证据和时效要求进行，并在模型不可用时安全降级。

成功标准：

1. 后续问题的检索查询保留明确主体；“它们的发展历史”必须关联 C 与 C++，不能将“发展历史”单独检索。
2. “今年的诺贝尔奖获得者有哪些”只接受同时支持当年、诺贝尔奖和获奖名单的来源；Microsoft 等无关页面不得成为证据。
3. 每轮最多生成少量互补查询，普通问题不无意义拆分；额外模型调用和查询数量都可配置、可观察、可限制。
4. LLM 结构化输出失败或超时时，使用既有确定性上下文改写继续运行，并在 Trace 中记录原因；不得把历史完整问题直接拼接到当前输入。
5. 没有通过主体与必需维度检查的正文时，仍然不得输出事实性总结。

## 非目标与约束

- 不改变 grounded 研究模式的原则：最终事实只来自成功读取且通过门控的正文。
- 不以关闭 SSRF、正文质量、引用验证或现有时效性检查来换取命中率。
- 不把完整对话、API 密钥、模型原始响应或网页正文写入诊断 Trace。
- 本轮不增加通用向量数据库或让 LLM 作为最终相关性裁判；先用结构化理解和确定性可解释规则解决明显偏题。
- 当前问题始终是回答目标；历史仅用于消歧，不能把上一轮的回答主题覆盖本轮意图。

## 总体数据流

```text
current_input + bounded conversation history
        │
        ├─ deterministic contextualizer ──> fallback QueryUnderstanding
        │
        └─ LLM query rewriter (when enabled and applicable)
                   │ strict JSON / schema validation
                   ▼
          QueryUnderstanding
          { resolved_subject, standalone_query, intent,
            required_facets, freshness, search_queries }
                   │
                   ▼
  parallel, capped search ──> stable candidate merge + relevance gates
                   │                         │
                   │                         └─ rejected reasons in Trace
                   ▼
  capped body reads ──> facet-aware evidence evaluation ──> answer current_input
```

`QueryUnderstanding` 成为会话层与研究图之间的显式边界。直接 `researchflow run` 没有历史，也使用同一对象，只是 `resolved_subject` 为空或来自当前输入。

## 结构化契约

新增不可变、可序列化的 `QueryUnderstanding` 模型，字段如下：

| 字段 | 含义与约束 |
| --- | --- |
| `current_input` | 原样保留的本轮用户问题；回答必须以此为目标。 |
| `resolved_subject` | 本轮所指的一个或多个明确主体；由当前输入或有限历史消歧得到。 |
| `standalone_query` | 可独立理解的检索表述；不得把整段旧问题拼接进去。 |
| `intent` | 有限枚举，如 `definition`、`comparison`、`history`、`list_current`、`age`。 |
| `required_facets` | 每个可验证答案所需的事实维度，例如 `C`、`C++`、`history`，或 `target_year`、`Nobel Prize`、`winners`。 |
| `freshness` | 是否需要当年/最近信息及目标年份；沿用现有时效意图规则。 |
| `search_queries` | 1–3 条互补查询，稳定去重；简单问题默认仅 1 条。 |
| `decision_source` | `llm` 或 `deterministic_fallback`，并携带不含敏感信息的失败类别。 |

模型调用使用 JSON Schema 校验，不允许模型返回搜索 URL、证据结论、任意工具调用或最终回答。JSON 不合法、字段不满足约束、主体被丢弃、超时或 provider 错误时，整个对象回退到确定性实现，不采纳部分模型输出。

## 会话与 LLM 改写策略

现有会话 contextualizer 继续负责维护上一轮明确主体并产生安全基线。新 `LLMQueryRewriter` 仅接收：本轮输入、最近有限条用户问题（不包含网页正文）、确定性主体与意图提示、当前日期和输出 schema。

调用条件：

- `agent-mode llm` 且 LLM provider 已配置；并且当前问题包含指代、过短省略表达、多个可能主体、当前/最近等时效表达，或确定性解析的主体置信度不足时调用。
- 自包含的首轮问题直接用确定性对象，避免为简单问题额外增加一次 LLM 往返。
- 相同 `session_id`、同一 `turn_id` 的重试复用已持久化的理解结果，避免恢复时重复调用；新轮次不跨会话复用。

示例：

```json
{
  "current_input": "它们的发展历史",
  "resolved_subject": ["C", "C++"],
  "standalone_query": "C 和 C++ 的发展历史",
  "intent": "history",
  "required_facets": ["C", "C++", "history"],
  "search_queries": ["C 和 C++ 的发展历史"]
}
```

`answer_target` 保持 `current_input`。因此 summarizer 回答“它们的发展历史”，而不是重复比较 C/C++ 或总结上一轮网页。

## 查询生成与候选筛选

研究图消费 `QueryUnderstanding.search_queries`，不再从裸 `state.query` 单独构造所有检索。

1. 简单意图只搜索 `standalone_query`。复杂问题按缺失的 `required_facets` 最多扩展为三条互补查询；每条查询都携带主体，时效问题携带目标年份。
2. 对多个检索响应进行稳定 URL 去重，保留最多 10 个候选，同时记录候选来自哪些查询。
3. 每个候选获得可解释分数：主体覆盖、意图/维度词命中、标题/摘要相关性、来源质量、时效匹配。网页仅出现主体名称不构成相关。
4. 主体覆盖是硬门槛：双主体问题默认要求两个主体都出现；未满足时写入 `missing_subject_coverage`。必需维度缺失写入 `missing_required_facet`；过时内容写入现有 `stale_for_current_query`。
5. 仅对通过候选门槛的来源读取正文，最多 5 个。若前几个失败且预算尚在，继续尝试未读的合格候选；若缺少某个维度，至多进行一次针对该维度的重新检索。

对于“今年的诺贝尔奖获得者有哪些”，候选必须覆盖目标年份、诺贝尔奖和获奖者/奖项名单的同义事实。官方诺贝尔奖完整名单可单独满足证据；非官方、年份不符或只含“Microsoft”的候选不能满足任何维度。

## 证据、回答与失败状态

`EvidenceStatus` 继续为 `SUFFICIENT`、`PARTIAL`、`INSUFFICIENT`，但评估输入升级为已读取正文的主体覆盖和 facet 覆盖，而不是来源计数。

- `SUFFICIENT`：每项 `required_facets` 均由至少一个成功读取、相关的正文支持；存在完整权威名单时允许一条来源。
- `PARTIAL`：仅输出有正文支撑的维度，并具体说明哪些 facet 未覆盖。
- `INSUFFICIENT`：一次有针对性的再检索耗尽或无合格候选后结束为 `insufficient_evidence`，不进入事实性 LLM 总结。

Summarizer 输入仅包含 `current_input`、意图、已接受的来源片段和 source ID；它不得看到候选摘要作为正文，也不得把未读取候选引用为来源。LLM 总结失败时，可走现有摘录式回退，但只有在有效正文存在时允许，并展示 `answer_policy` 和回退原因。

## 性能与可观测性

性能预算：

- LLM 改写仅在需要会话消歧/歧义处理时运行；小输出、严格超时和单轮缓存。
- 互补搜索上限为 3，并发执行；候选最多 10、正文最多 5。
- 将重检索限制为一次且仅针对未满足 facet，避免最大步骤耗尽的循环。
- 每个动作消耗明确图步骤；达到预算时状态为失败或证据不足，不能伪标记为 `completed`。

JSONL Trace 记录：`thread_id`、`turn_count`、`current_input`、历史消息数、`resolved_subject`、`standalone_query`、`intent`、`required_facets`、是否调用 LLM 改写及降级原因、生成的查询、候选分数与拒绝原因、source ID、evidence gaps、answer policy、重检索和步骤预算状态。网页正文、完整模型提示词、密钥和认证头不得写入 Trace。

Web UI 将继续区分“搜索候选（未读取正文）”“已读取正文”“拒绝/读取失败”，并在有重试或证据不足时展示简短、可操作的状态，而非把候选数量显示为证据充分。

## 测试与验收

先增加失败测试，再实现。至少覆盖：

1. C/C++ 后续“它们的发展历史”：结构化对象和所有检索查询均包含 C、C++ 与历史；马克思/GIS 候选因主体缺失被拒绝。
2. “今年的诺贝尔奖获得者有哪些”：查询包含目标年；Microsoft 候选被拒绝；官方完整名单可单源使状态为 `SUFFICIENT`。
3. 不合规 JSON、超时、provider 失败均回退到确定性 `QueryUnderstanding`，且 Trace 标记原因、不泄漏内容。
4. 简单首轮不调用 rewriter；歧义后续问题调用一次，恢复/重试不重复调用。
5. 候选、读取、重检索和步骤数均不超过配置上限；缺少证据时不生成无依据事实回答。
6. 现有会话隔离、SQLite 恢复、CLI、Web API/UI 与 grounded 引用测试保持通过。

验收时使用真实入口分别验证 C/C++ 对话和当年诺贝尔获奖者问题，记录实际查询、候选拒绝理由、成功读取数、来源、回答状态与耗时；若网络或 provider 不可用，明确报告限制而不将离线测试视为在线成功。
