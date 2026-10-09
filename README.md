# ResearchFlow Agent

ResearchFlow 是一个本地优先、带证据门槛的研究助手。它可以检索本地文档，可选地
搜索并安全读取网页；只有成功读取且通过相关性检查的正文才可用于事实性回答。

默认研究执行器是带 SQLite checkpoint 的 LangGraph 图。它与单独的多轮会话图共同
支持持久会话、澄清中断与恢复。项目仍是单机工具：服务默认只监听回环地址，不能作为
云端、多用户或生产级服务使用。

## 主要能力

- 本地 Markdown/Text 检索、可选 Tavily 网页候选和受 SSRF 保护的 HTML 正文读取。
- `SUFFICIENT`、`PARTIAL`、`INSUFFICIENT` 证据状态；没有成功读取的相关正文时不会
  把模型常识包装成研究结论。
- 可选 OpenAI-compatible LLM 进行规划、选择和基于来源的总结；失败时明确显示
  `extractive_fallback` 与脱敏原因。
- `researchflow chat` 的跨进程会话记忆：保留当前输入、主体、检索独立问题、意图、
  所需证据分面和语言偏好。
- 空问题会在工具或文件副作用之前中断；恢复采用相同 session/thread ID。
- JSONL 工具 trace、Markdown 报告、SQLite 运行快照/事件和本地 React Web 控制台。

详细的数据流、证据规则、记忆与恢复约束见
[架构文档](docs/architecture.md)，本地控制台的启动方法见
[操作文档](docs/local-web-console.md)。

## 安装

需要 Python 3.11、[uv](https://docs.astral.sh/uv/) 和 Node.js（仅 Web 控制台）。

```powershell
uv sync --frozen --extra llm --extra ui
npm --prefix frontend ci
```

`--extra llm` 与 `--extra ui` 允许使用可选的 LLM 和本地 API；纯离线命令只需
`uv sync --frozen`。

若运行“今年/最新 + 获奖名单/完整列表”类问题，默认至少需要两个不同的已读取正文来源。
只有显式配置的 HTTPS 官方完整名单页可使用单来源例外：

```toml
# researchflow.toml
[web]
official_domains = ["example-official.org"]
```

## 命令行快速开始

```powershell
uv run researchflow run "tool calling"
uv run researchflow chat "C和C++有什么区别" --session-id demo
uv run researchflow chat "它们都用什么编译器" --session-id demo
```

`run` 默认使用 `--orchestrator langgraph`。保留 `--orchestrator graph` 和
`--orchestrator loop` 仅用于兼容/诊断，不是默认路径。

要启用 Web 检索，设置 `RESEARCHFLOW_SEARCH_API_KEY` 后显式传入 `--enable-web`：

```powershell
uv run researchflow chat "村上春树是谁" --session-id author-demo --enable-web
```

要启用 LLM，总结和规划配置放在 `.env.local` 或进程环境中：
`RESEARCHFLOW_LLM_API_KEY`、`RESEARCHFLOW_LLM_BASE_URL`、
`RESEARCHFLOW_LLM_MODEL` 和可选的 `RESEARCHFLOW_LLM_TIMEOUT`。密钥不会写进
报告、JSONL 或控制台诊断。

```powershell
uv run researchflow chat "什么是 RAG？" --session-id rag-demo --agent-mode llm --enable-web
```

使用空输入时，`chat` 会请求澄清并在同一次命令中提示恢复答案。会话可列出和精确删除：

```powershell
uv run researchflow sessions list
uv run researchflow sessions delete demo
```

## 输出与诊断

默认输出根目录为 `output/`：

```text
output/notes/<run_id>.md                 # UTF-8 报告
output/traces/<run_id>.jsonl             # 工具调用与安全诊断 trace
output/sessions/checkpoints.sqlite3      # LangGraph 与应用运行快照
```

搜索摘要只作为候选信息；`已读取正文` 才是可用证据。网页抓取失败、低相关来源和证据
缺口都会在报告、trace 或本地控制台中明确呈现。Windows 终端不能编码的字符会被安全
替代，文件内容仍为 UTF-8。

## 验证

```powershell
uv lock --check
uv run pytest
uv run ruff check .
uv run ruff format --check .
npm --prefix frontend test -- --run
npm --prefix frontend run build
uv run researchflow --help
uv run researchflow serve --help
```

## 当前限制

- 执行是单进程同步的；SSE 用于本地观察与重放，不是分布式任务队列。
- Web 页面须为可公开访问的 HTML；私网/本机地址、非 HTML、超大页面和重定向目前会
  被安全拒绝。
- LLM、网页搜索和页面可访问性是外部依赖。失败时系统会记录原因，不会伪造已验证结论。
- 抽取式离线回退不会翻译外文来源；若 LLM 不可用或未满足语言/引用约束，报告会说明
  回退状态。

许可证见 [MIT License](LICENSE)。
