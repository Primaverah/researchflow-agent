# 本地 Web 控制台操作说明

ResearchFlow 的 Web 控制台是单用户、本机开发界面。后端只允许绑定
`127.0.0.1`、`::1` 或 `localhost`；不要把这个开发服务器暴露到局域网或公网。

## 启动

在仓库根目录安装依赖：

```powershell
uv sync --frozen --extra llm --extra ui
npm --prefix frontend ci
```

在第一个终端启动 API。`--enable-web` 是可选项，只有设置
`RESEARCHFLOW_SEARCH_API_KEY` 时才使用它；`--agent-mode llm` 同样是可选项。

```powershell
uv run researchflow serve --host 127.0.0.1 --port 8000 --enable-web --agent-mode llm
```

在第二个终端启动 Vite 开发服务器：

```powershell
npm --prefix frontend run dev -- --host 127.0.0.1
```

打开 Vite 打印的地址（通常是 <http://127.0.0.1:5173>）。开发服务器将相对
`/api` 请求代理到本地 API，因此浏览器不会保存 LLM 或搜索密钥。

生产静态构建只用于检查前端产物：

```powershell
npm --prefix frontend run build
```

## 使用流程

1. 点击“新建会话”，输入研究问题并发送；也可以从左侧选择已有会话。
2. 中间区域显示当前运行状态、回答或澄清提示。右侧严格区分“搜索候选”“已读取正文”
   和“拒绝或读取失败”。候选摘要不是已读正文，也不能单独支持事实性回答。
3. 为空或无法形成研究问题的输入会进入 `waiting_for_input`。填写澄清内容后，按钮会变为
   “继续研究”，请求以原 session ID 恢复，而不是重新执行中断前的工具调用。
4. 页面使用 SSE 接收运行事件，刷新、断连或完成后会回读持久化 run snapshot；因此刷新
   不会丢失已经保存的候选、已读来源或最终状态。

每次启动运行、恢复运行的请求都带独立 `Idempotency-Key`。重复提交同一键会返回已保存的
运行，而不会重复执行工作流。

## 文件与诊断

给定 `--output-dir output/web-console`，本地状态位于：

```text
output/web-console/notes/<run_id>.md
output/web-console/traces/<run_id>.jsonl
output/web-console/sessions/checkpoints.sqlite3
```

- Markdown 和 JSONL 始终是 UTF-8。
- JSONL 是低层工具与图节点审计记录；SQLite 保存面向控制台的运行快照、事件和
  LangGraph checkpoints。
- 运行不成功读取相关正文时，状态为 `insufficient_evidence`，而不是成功完成；诊断
  回答会列出候选、搜索失败或正文读取失败原因。
- `researchflow sessions list --output-dir ...` 可列出会话；`sessions delete <id>` 只删除
  精确匹配会话的 checkpoint，不会删除其他会话。

## 证据与安全边界

研究模式默认 grounded。LLM 只接收通过相关性判断的正文和其 source ID；不存在的引用、
无依据结论和空证据下的模型知识应被拒绝。LLM 不可用、结构化输出无效、引用不合法或
语言约束不满足时，报告会显示 `extractive_fallback` 和脱敏原因。

Web 读取只接受本轮搜索候选中的公共 HTTP(S) HTML 页面。URL、DNS 地址和内容类型都会
检查；本机、私网、混合 DNS、非 HTML、超大响应和重定向会被拒绝。当前版本刻意不自动
跟随重定向，因为安全的逐跳验证和连接绑定尚未实现。

该控制台不是登录系统、协作系统或任务队列。LLM、搜索提供方和目标网页失败属于外部
依赖故障，系统会展示诊断而非伪造来源验证的答案。
