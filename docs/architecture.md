# ResearchFlow Agent 架构设计

本文描述当前仓库中可执行实现的架构，而非产品愿景。默认运行模式是
**grounded research**：事实性结论必须来自成功读取并通过相关性检查的来源。

## 1. 技术栈与边界

| 层 | 实现 | 职责 |
| --- | --- | --- |
| 运行时 | Python 3.11 | 单进程同步 CLI 与编排运行时 |
| CLI | Typer | `run`、`chat`、`sessions` 等命令入口 |
| 数据模型 | Pydantic v2 | 状态、工具输入输出、持久化载荷的验证 |
| 会话图 | LangGraph | 多轮状态、`interrupt()`、`Command(resume=...)` |
| Checkpoint | `langgraph-checkpoint-sqlite` + SQLite | 按线程恢复持久化会话 |
| 工具执行 | 内部 `ToolExecutor` / `ToolRegistry` | 参数验证、调用、结构化追踪 |
| 本地检索 | 文档索引与文本读取工具 | 离线文档候选与正文证据 |
| Web 检索 | Tavily REST 适配器（显式 `--enable-web`） | 搜索候选，不等于正文证据 |
| Web 读取 | 标准库 HTTP 客户端与 HTML 提取 | 在 URL 安全策略内读取候选正文 |
| 可选 LLM | OpenAI-compatible provider | 结构化规划、选择与来源约束总结 |
| 质量工具 | pytest、Ruff、uv | 回归验证、静态检查、锁文件一致性 |

依赖版本和可选 LLM/embedding 依赖以 `pyproject.toml` 为准。默认安装不要求
LLM 或联网；Web 与 LLM 都是显式启用、运行时配置的能力。

## 2. 两条运行路径

普通单次研究从 `researchflow run` 进入 `_run_workflow()`：创建工具上下文、注册
本地工具，按选项加入 Tavily 与网页读取工具，再运行循环编排器或固定研究图。

会话研究从 `researchflow chat` 进入 `LangGraphSessionRunner`。每个 CLI 调用重新
创建 Runner 也可以，因为状态不在 Runner 内存中，而在
`<output-dir>/sessions/checkpoints.sqlite3`。同一 SQLite 文件可保存多个会话；
`session_id` 同时是 LangGraph 的 `thread_id`，因此状态按线程隔离。

会话内的数据流为：

```text
current_input
  -> contextualizer + conversation_history
  -> standalone_query / answer_target / answer_language
  -> GraphAgentRunner
  -> search candidates -> accepted full-text sources
  -> grounded summarizer -> Markdown report + JSONL traces
```

`standalone_query` 只用于检索和规划；`answer_target` 是要回答的实际问题；
`current_input` 保留最新用户输入。这个区分避免把“用中文回答”错误地当作新的
研究主题。

## 3. 检索、读取与证据

固定研究图的节点为：`initialize -> plan -> retrieve -> assess_evidence ->
read_sources -> synthesize -> verify -> save -> finish`。在没有候选或没有可用证据
时，图可有限次重新规划；读取失败时会在步骤预算内继续尝试未读候选，而不是反复
读取同一批链接。

1. 规划器根据 `standalone_query` 生成本地搜索与（启用后）Web 搜索操作。
2. 检索结果被稳定去重，最多保留 10 个候选。
3. `fetch_url` 只能读取本轮成功搜索返回的候选；默认最多读取 5 个高优先级候选。
4. 本地文档和成功抓取的网页正文均会经过相关性判断。标题出现主体名称本身不够；
   例如编译器问题需要 GCC、Clang、MSVC 或同类编译器证据，年龄问题需要出生信息。
5. 只有成功读取且通过相关性检查的 `ReadDocumentOutput` / `WebSource` 进入总结器。

搜索摘要是“候选信息”，不能标为已读正文。若所有正文读取失败，报告会明确说明
没有可验证的事实性答案，并分别列出未读搜索候选、搜索失败原因、正文读取失败原因。

### URL 安全

网页读取验证 URL 协议、主机名和 DNS 解析地址，拒绝非 HTTP(S)、本机、私网与其它
非全局地址。混合 DNS 结果也会拒绝，避免“先校验、后连接到不同地址”的 SSRF 绕过。
当前实现拒绝重定向；因此重定向来源会作为读取失败记录。安全的、逐跳重新验证且将
实际连接绑定到已验证地址的重定向支持仍是后续工作，不能把关闭 SSRF 检查作为替代。

## 4. 回答生成与引用

`ExtractiveSummarizer` 是离线确定性回退：它从已读取正文抽取相关文本，并列出来源。
`LLMSummarizer` 仅将已验证正文交给模型，要求结构化 JSON 返回 `summary` 与
`source_paths`。返回引用必须是输入证据集中的来源，否则该次 LLM 输出被拒绝并回退。

LLM 提示包含实际 `answer_target`，明确要求直接回答问题而不是逐页概括。会话层传入
语言策略：中文输入默认倾向 `zh`；“用中文回答”等纯语言指令会显式设为 `zh`，复用
上一轮研究目标。若 LLM 在要求中文时返回没有中文字符的摘要，该输出会被拒绝而不会
伪装成合格的中文回答。

证据状态使用 `SUFFICIENT`、`PARTIAL`、`INSUFFICIENT` 描述。当前固定图会在零有效
来源时生成可诊断的不足证据报告，不用模型内部知识补齐；`PARTIAL` 的分面级约束和
结论逐条 source-id 验证仍应继续加强（见“当前限制”）。

## 5. 多轮记忆与上下文改写

每个线程的 checkpoint 状态包括：

- `messages`：用户和助手消息，使用追加 reducer；只有历史压缩节点能显式替换旧消息。
- `summary`：长会话的压缩历史，用于保留记录，不拼接进检索词。
- `current_input`：最新原始用户输入。
- `resolved_subject`：上一轮已确定主体，如 `C和C++`、`成龙`、`RAG`。
- `intent` 与 `required_facets`：当前问题意图及需要的证据，例如编译器、代表作、出生
  日期/年龄或更新时间。
- `standalone_query`：消除代词后的检索用问题。
- `answer_target` 与 `answer_language`：回答的实际目标和持久化语言偏好。
- `turn_count`、`interrupt_status`、`response`：运行诊断和结果。

代词解析仅做完整 token 替换，覆盖“它、它们、他、他们、她、两者、该项目、这个游戏”等；
不会执行 `current_input[1:]` 或将上一轮完整问题直接拼接到当前输入。例如：

```text
C和C++有什么区别  -> resolved_subject = C和C++
它们都用什么编译器 -> standalone_query = C和C++都使用哪些常见编译器

什么是RAG？帮我查找3篇相关论文 -> answer_target = 原问题
用中文回答 -> 复用原 standalone_query，answer_language = zh
```

`SessionResearchRequest` 是会话层与研究工作流的结构化边界。它传递当前输入、独立
查询、回答目标、语言、历史、主体、意图与分面，避免下游只收到一个丢失语境的字符串。

## 6. 中断、恢复与一致性

空问题会在 `clarify` 节点调用 `interrupt()`，此点位于 research、工具调用、trace 写入
和 Markdown 写入之前，因此中断本身没有副作用。恢复使用
`Command(resume=answer)`，并与初始 `invoke()`、`get_state()` 使用相同的
`{"configurable": {"thread_id": session_id}}`。

SQLite 的 checkpointer 以 JSON 序列化持久化图状态；所有字段必须 JSON-safe，运行时
服务（HTTP、LLM、工具实例）不进入 checkpoint。重新创建 Runner 后可从同一数据库
恢复。会话目录旁的 `session_catalog` 只用于列出/删除精确的 session id；删除时按
`thread_id` 清理相关 checkpoint/writes 行，不影响其它会话。

## 7. 配置、输出与可观测性

`.env.local` 由项目配置加载器读取；外层单/双引号会被移除，避免把引号作为 API key
的一部分。环境变量优先级遵循加载器定义。密钥不写入 Pydantic 状态、Markdown、JSONL
或终端诊断。

每个运行在输出目录写入：

```text
notes/<run_id>.md          # UTF-8 Markdown 报告
traces/<run_id>.jsonl      # 每行一个工具/决策/图事件
sessions/checkpoints.sqlite3  # chat 状态
```

JSONL 记录工具状态、耗时、图节点、候选/拒绝原因和 LLM 决策元数据；大报告正文会被
redact，凭据和 HTTP 正文不进入 trace。Windows 控制台输出在目标编码无法表示字符时使用
替代字符，文件仍保持 UTF-8，避免 `UnicodeEncodeError` 使命令失败。

## 8. 当前限制与演进方向

- 网页重定向当前安全拒绝，尚未实现逐跳校验和连接绑定。
- 相关性规则已有意图特例，但尚未形成统一的可解释分数模型。
- `PARTIAL` 的按分面回答、逐条结论 source-id verifier 与完整会话诊断字段仍需继续完成。
- 离线抽取器不会翻译英文原文；LLM 不可用或未遵守输出约束时，报告仍会标明来源而不能
  承诺把所有证据翻译成人类指定语言。
- 搜索、LLM、网页可访问性是外部依赖；失败会写入报告和 trace，不能由模型知识伪造为
  检索证据。
