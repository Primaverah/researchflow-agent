# 本地 Web 控制台与渐进式 LangGraph 迁移设计

## 1. 目标与边界

本设计面向单个开发者在本机运行 ResearchFlow Agent。目标是在不改变 grounded
research 约束的前提下，提供一个可查看、可恢复、可诊断的 Web 控制台，并把当前自定义
研究状态机逐步迁移为可 checkpoint、可中断、可流式观察的 LangGraph 工作流。

成功标准：

- 用户能创建、查看、继续和删除本地会话，且同一 `session_id` 在进程重启后仍保留。
- 每次研究能在 UI 中展示计划、检索候选、正文读取、拒绝原因、证据状态、总结状态和最终
  报告；界面不能把搜索摘要显示为已读正文。
- 用户能看到实时事件和完成后的完整 Trace；断开浏览器不会中断后端运行。
- 用户能在需要澄清或显式暂停时恢复同一个 checkpoint，且恢复不重复已完成的工具调用。
- 核心研究图完成迁移后，证据门控、来源约束、步骤预算和现有 CLI 的行为保持兼容。

不在本轮产品范围内：云部署、登录、多人协作、共享会话、远程任务队列、浏览器直接持有
任何 API 密钥、移动端适配或替换 SQLite。

## 2. 总体架构

采用同机回环（loopback）部署：Python 后端拥有所有凭据、SQLite、LangGraph、工具执行和
文件输出；浏览器只呈现数据并发出用户操作。

```text
React + Vite 浏览器界面
  ├─ HTTP JSON：会话、运行、恢复、报告、Trace
  └─ SSE：运行事件订阅
          ↓
FastAPI 本地 API（127.0.0.1）
  ├─ Application service：会话和运行生命周期
  ├─ Event broker：按 run_id 缓冲并广播事件
  ├─ LangGraph 会话/研究图
  ├─ ToolExecutor / Web / LLM providers
  └─ SQLite：checkpoint、会话目录、运行元数据
          ↓
output/
  ├─ notes/<run_id>.md
  ├─ traces/<run_id>.jsonl
  └─ sessions/checkpoints.sqlite3
```

前端与 CLI 不直接调用彼此。两者都调用同一 application service；CLI 保留为自动化、脚本和
离线使用入口。这可避免 Web 和 CLI 形成两套不一致的恢复或证据逻辑。

## 3. 稳定领域契约

先定义与 UI 无关的 Python service 边界，再引入 HTTP：

```python
class ResearchService:
    def start_turn(self, request: StartTurn) -> RunSnapshot: ...
    def resume_turn(self, request: ResumeTurn) -> RunSnapshot: ...
    def get_run(self, run_id: str) -> RunSnapshot: ...
    def get_session(self, session_id: str) -> SessionSnapshot: ...
    def list_sessions(self) -> list[SessionListItem]: ...
    def delete_session(self, session_id: str) -> None: ...
```

`StartTurn` 包含 `session_id`、`message`、运行选项（是否 Web、agent mode、允许域名）和
输出目录；不得包含 API key。`RunSnapshot` 至少包含 `run_id`、`session_id`、状态、是否
等待恢复、澄清提示、报告路径、来源、证据状态、结束原因和最近事件序号。

领域状态为：

```text
created → running → waiting_for_input → running → completed
                   ↘ failed
running → paused → waiting_for_input
```

`completed` 仅表示流程以有效结束原因完成；零有效证据的结果必须标记为
`insufficient_evidence`，不能显示为“已回答”。`failed` 表示基础设施、配置或不可恢复执行
错误。`waiting_for_input` 表示 LangGraph interrupt 已持久化且尚未收到 `Command(resume=...)`。

## 4. HTTP API

所有 API 初期只监听 `127.0.0.1`。不允许通过 `0.0.0.0` 默认暴露；如未来增加局域网模式，
必须另行设计认证和 CSRF 边界。

| 方法与路径 | 作用 | 成功结果 |
| --- | --- | --- |
| `GET /api/health` | 本地后端存活检查 | 版本、SQLite 可用性、能力标志；不返回密钥 |
| `GET /api/sessions` | 会话列表 | session id、更新时间、最后运行摘要 |
| `GET /api/sessions/{id}` | 会话快照 | 消息、上下文摘要、等待恢复状态 |
| `DELETE /api/sessions/{id}` | 删除精确会话 | `204`；只删除对应 thread_id 数据 |
| `POST /api/sessions/{id}/turns` | 创建一轮研究 | `202` 和 `run_id` |
| `POST /api/sessions/{id}/resume` | 恢复已中断会话 | `202` 和原/新 `run_id` |
| `GET /api/runs/{id}` | 运行最终或中间快照 | `RunSnapshot` |
| `GET /api/runs/{id}/events` | SSE 事件订阅 | 有序事件流，支持 `Last-Event-ID` |
| `GET /api/runs/{id}/report` | 读取 UTF-8 Markdown 报告 | 报告模型或 Markdown 文本 |
| `GET /api/runs/{id}/trace` | 读取经过脱敏的 Trace | JSON 事件数组 |

请求重复发送的处理：前端为每个 `POST` 提供 `Idempotency-Key`；同一会话和键的重复请求
返回初次创建的 `run_id`，不再次执行工具。`resume` 只接受 `waiting_for_input` 状态，否则
返回 `409 Conflict`。同一 session 同时运行第二个 turn 也返回 `409`。

## 5. 事件协议与流式显示

SSE 是观察通道，不是状态真相。后端先在 SQLite/图状态完成关键状态转换，再发布事件；客户端
重连后用 `GET /api/runs/{id}` 补齐快照，并以 `Last-Event-ID` 读取仍在内存缓冲中的事件。
历史完整审计仍来自 JSONL Trace。

所有事件使用统一信封：

```json
{
  "event_id": 17,
  "run_id": "...",
  "session_id": "...",
  "timestamp": "2026-10-08T00:00:00Z",
  "type": "source_rejected",
  "data": {}
}
```

首期事件类型：

| 类型 | `data` 最小字段 | UI 含义 |
| --- | --- | --- |
| `run_started` | mode、web_enabled | 创建运行卡片 |
| `graph_node_started` / `graph_node_finished` | node、step_count | 更新流程时间线 |
| `search_completed` | query、candidate_count | 显示搜索候选数 |
| `candidate_selected` | source_id、title、url、score | 显示待读取来源 |
| `source_read` | source_id、title、url、content_length | 标记成功正文证据 |
| `source_rejected` | source_id、reason | 显示拒绝原因，不显示正文 |
| `evidence_assessed` | status、gaps、source_ids | 更新证据面板 |
| `generation_status` | mode、fallback_reason、error_type | 显示 LLM 或抽取式回退 |
| `interrupt_requested` | prompt、checkpoint_id | 显示恢复输入框 |
| `run_completed` / `run_failed` | end_reason、report_available | 展示最终结果或错误 |

事件不得包含 API key、完整网页正文、原始 LLM prompt、原始 LLM 输出或未脱敏异常。正文预览
只可由用户显式读取已接受来源的受限字段，且不得出现在 SSE 默认载荷中。

## 6. 前端页面与交互

首个可用版本只需要一个三栏工作台：

```text
会话列表          当前会话/报告                       运行证据与事件
────────          ─────────────                       ──────────────
新建会话          输入框、发送、恢复输入框             节点时间线
历史会话          用户/助手消息                         搜索候选
删除会话          Markdown 报告                         已读正文来源
                  报告生成状态                          拒绝/失败原因
                                                        EvidenceStatus
```

交互规则：

- “发送”成功后立即创建运行卡片并订阅 SSE；刷新页面后通过会话和运行快照恢复界面。
- `waiting_for_input` 时禁用普通发送，显示澄清问题和“恢复”按钮；恢复使用专用 API，不能
  把回答误提交为新 turn。
- 来源分为“搜索候选”“已读取正文”“拒绝/读取失败”三个区域，禁止合并展示。
- 最终报告必须显示 `llm_grounded`、`extractive_fallback` 或证据不足状态；无来源时不得用
  正常答案样式掩盖失败。
- UI 显示本地文件路径和来源链接，但 Markdown/HTML 渲染必须消毒，外链使用安全属性。

## 7. 中断与恢复设计

现有空白输入澄清只是最低限度的 human-in-the-loop。服务层将它扩大为显式生命周期：

1. 图需要用户输入时，先写入 `waiting_for_input` 和 interrupt 元数据，再返回 interrupt。
2. API 发布 `interrupt_requested`；前端展示提示，但不自动恢复。
3. 用户提交恢复内容时，service 用原 `session_id` 作为 `thread_id` 调用
   `Command(resume=answer)`。
4. 图从 checkpoint 的 interrupt 点继续；先前成功的工具调用不重复。
5. 恢复成功或失败均发布事件，并写入用户/助手消息和运行记录。

首期保留“空问题”澄清触发器；第二期可增加：用户显式暂停、检索歧义、域名访问确认。任何
interrupt 节点之前不得有不可重复的文件写入或工具副作用。取消运行不是强行终止 Python
线程：首期只允许在安全节点间记录 `cancel_requested` 并路由到结束节点。

## 8. 渐进式 LangGraph 迁移

当前 `LangGraphSessionRunner` 已是 `StateGraph + SqliteSaver`；`GraphAgentRunner` 仍是自定义
同步循环。迁移不应在引入前端时一次完成，而应保持 service 和事件契约稳定地分阶段替换：

1. **适配阶段**：为现有 `GraphAgentRunner` 发出标准事件；CLI 和 Web 共用 `ResearchService`。
2. **图骨架阶段**：以现有 `AgentGraphState` 定义 LangGraph `StateGraph`，只迁移
   `initialize → plan → retrieve → assess_evidence`，并与旧 runner 做相同输入的回归比较。
3. **工具阶段**：迁移 `read_sources`、replan、步骤预算和候选重试；每个工具结果成为可
   checkpoint 的状态增量。
4. **生成阶段**：迁移 `synthesize → verify → save → finish`，保留严格证据门控和
   source-id 校验。
5. **统一阶段**：会话图与研究子图在一个可组合的 LangGraph 调用链中运行；删除旧 runner
   前必须完成兼容性、恢复和无重复工具调用验证。

迁移期间不可将 HTTP client、SQLite connection、LLM client 或 ToolExecutor 放入 checkpoint
状态；这些由每次运行构建并通过依赖注入使用。所有状态更新是 JSON-safe 增量，`messages`
继续使用追加 reducer，历史压缩是唯一允许替换消息列表的节点。

## 9. 分阶段实施与验收

### 阶段 A：服务层与可观察前端

- 新增 `ResearchService`、运行元数据模型和 CLI 适配。
- 新增 FastAPI 本地 API、React/Vite 工作台、会话/报告/Trace 的只读视图。
- 先使用短轮询获取运行状态，不改变现有执行器。

验收：CLI 与 UI 对同一运行显示相同报告、来源、证据状态和结束原因；重启 API 后会话仍可
浏览；不新增网络监听到非 loopback 地址。

### 阶段 B：事件与实时进度

- 将现有 JSONL/图节点诊断映射为标准运行事件。
- 加入 SSE、重连、运行快照补齐和事件序号。

验收：网页中能实时区分搜索候选、已读正文、拒绝来源与 LLM 回退；断开/刷新浏览器不会重复
运行或丢失最终报告。

### 阶段 C：显式暂停恢复

- 新增运行状态表、`resume` API、前端恢复交互和 idempotency。
- 将当前 CLI 自动恢复改为可选便利模式；Web 默认等待用户明确恢复。

验收：创建新 Runner/新 API 进程后，使用相同 session id 可以恢复；同一恢复请求重复发送不
重复抓取网页、调用工具或写报告；不同 session 完全隔离。

### 阶段 D：研究图 LangGraph 化

- 分节点迁移 `GraphAgentRunner`，保留旧 runner 作为受控对照直至验收完成。
- 为节点路由、checkpoint 恢复、证据门控、取消和事件顺序增加集成测试。

验收：在相同离线 fixture 下，新旧路径的候选上限、读取上限、证据状态、引用约束和结束原因
一致；中断/恢复后不重复副作用；旧 runner 移除后 CLI、Web 和全量测试继续通过。

## 10. 测试与安全要求

- 单元测试：service 状态机、事件序列、idempotency、消息 reducer、来源展示分类、API 错误
  映射。
- 集成测试：SQLite 跨进程恢复、两个 session 隔离、SSE 重连、interrupt/resume、断网/LLM
  错误、无证据报告、工具副作用不重复。
- 端到端测试：本地浏览器从新会话到报告、从澄清到恢复、从刷新到继续观察。
- 回归测试：保留现有 CLI、离线模式、Web evidence gate、Windows GBK 控制台与 UTF-8 文件
  输出测试。
- 安全测试：API 仅 loopback、无 API key 响应/事件、Markdown 消毒、外链安全、URL SSRF
  防护不因前端接入而放宽。

## 11. 已知限制与决策记录

- SSE 是单向的；用户动作一律经 HTTP POST，避免把恢复/取消语义塞进事件通道。
- 首期使用本地 SQLite 和单进程运行，不引入 Redis、Celery、消息队列或账户系统。
- 浏览器关闭不取消运行；运行可在 API 进程存活期间继续，重启后只能从已经 checkpoint 的
  安全节点恢复。
- 当前网页重定向仍由安全层拒绝；前端只展示失败原因，不能绕过或关闭 SSRF 策略。
- “当前日期”等问题在 grounded research 模式仍需要工具可验证来源；前端不应把浏览器或
  服务器时钟伪装为已检索事实。
