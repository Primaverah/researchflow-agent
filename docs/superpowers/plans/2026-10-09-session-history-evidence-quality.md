# 会话历史与网页证据质量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让本地控制台可浏览持久会话历史，并以可解释的网页质量和来源数量门槛阻止低质量证据生成 grounded 回答。

**Architecture:** 网页读取层在正文解码后执行独立质量判断并返回稳定错误码；研究图把 query 分类为标准或时效性完整名单策略，再根据来源数量/配置的官方域名判定证据是否足够。应用层将每轮问题和证据策略持久化，React 控制台读取会话快照、切换历史 run，并以安全链接卡片呈现来源。

**Tech Stack:** Python 3.11、Pydantic v2、LangGraph/SQLite、FastAPI、pytest、React 19、TypeScript、Vitest、Testing Library、Ruff。

**Spec:** `docs/superpowers/specs/2026-10-09-session-history-evidence-quality-design.md`

## Global Constraints

- 保持单用户、回环地址和单进程同步范围；不实现云端、多用户或账户。
- 不关闭 URL/SSRF、混合 DNS、HTML 类型或大小限制检查。
- Markdown、JSONL、SQLite 文本保持 UTF-8；不记录 API 密钥、HTTP body 或完整网页正文。
- 网页来源仅在成功读取、相关性和质量判断均通过后传给 summarizer。
- 时效性完整名单默认需要两个不同 URL；仅配置 `web.official_domains` 的 HTTPS 完整名单页可单独通过。
- 前端不得注入网页 HTML；回答用纯文本，外部来源使用普通锚点。
- 每项行为修改先写失败测试；不推送、不合并。

## Review Focus

- HTTP 标头误报 `iso-8859-1` 的 GBK/GB18030 中文页：不能误拒绝真正可读内容；Task 1 覆盖。
- 只有一个配置官方域名但页面不是完整名单：不得享有单来源例外；Task 2 覆盖。
- 两个 URL 相同但 title 不同的来源：不得计为两条独立来源；Task 2 覆盖。
- 点击会话时网络失败或会话为空：保留可见错误/空态，不误显示上一个会话内容；Task 4 覆盖。
- 已完成 run 的 SSE 重放后切换历史 run：不得重新触发研究或覆盖手动选择的历史条目；Task 5 覆盖。

---

### Task 1: 网页正文乱码与低质量拒绝

**Files:**
- Modify: `src/researchflow/tools/web/http.py`
- Test: `tests/unit/web/test_http.py`

**Interfaces:**
- Produces: `assess_text_quality(title: str, content: str, *, expected_language: str = "") -> str | None`，`None` 表示合格，其他值为 `web_garbled_content` 或 `web_low_quality_content`。
- Consumes: 现有 `_decode_html()` 和 `_TextExtractor`，不改变 URL 验证器或请求连接行为。

- [ ] **Step 1: 写失败测试，覆盖乱码拒绝和可读 GB18030 保留**

```python
def test_fetch_rejects_mojibake_content_after_successful_decode():
    with pytest.raises(WebFetchError, match="garbled") as error:
        SafeHttpClient(opener=garbled_opener, resolver=public_resolver).fetch(URL)
    assert error.value.error_type == "web_garbled_content"

def test_fetch_accepts_readable_gb18030_content():
    page = SafeHttpClient(opener=gb18030_opener, resolver=public_resolver).fetch(URL)
    assert "诺贝尔奖" in page.content
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/web/test_http.py -k "mojibake or gb18030" -v`

Expected: mojibake 测试失败，因为当前只检查正文非空。

- [ ] **Step 3: 在 `http.py` 实现 `assess_text_quality` 并接入 `SafeHttpClient.fetch`**

在解析 HTML 后调用质量函数。通过 replacement/control/Latin-1 mojibake 模式、可见字符比例、
默认标题和正文长度组合判断；保留 `WebFetchError` 的稳定错误码并且不泄漏正文。

- [ ] **Step 4: 运行单元测试确认通过**

Run: `uv run pytest tests/unit/web/test_http.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/researchflow/tools/web/http.py tests/unit/web/test_http.py
git commit -m "fix: reject garbled web content"
```

### Task 2: 时效性名单来源策略与官方完整名单例外

**Files:**
- Create: `src/researchflow/agent/evidence_policy.py`
- Modify: `src/researchflow/agent/graph.py`
- Modify: `src/researchflow/agent/research_graph.py`
- Modify: `src/researchflow/cli.py`
- Test: `tests/unit/agent/test_evidence_policy.py`
- Test: `tests/unit/agent/test_graph.py`
- Test: `tests/integration/test_research_graph_parity.py`

**Interfaces:**
- Produces: `EvidencePolicy(name: Literal["standard", "current_complete_list"], required_source_count: int, official_domains: tuple[str, ...])` 与 `evaluate_evidence_policy(query, sources, *, official_domains) -> EvidenceAssessment`。
- Consumes: `GraphCandidate`、`WebSource`、`ReadDocumentOutput` 与 `researchflow.toml` 的 `[web] official_domains`。

- [ ] **Step 1: 写失败测试，定义策略和边界**

```python
def test_current_complete_list_requires_two_distinct_sources():
    assessment = evaluate_evidence_policy("今年诺贝尔奖获奖者名单", [one_source])
    assert assessment.sufficient is False
    assert assessment.required_source_count == 2

def test_configured_official_complete_list_allows_one_source():
    assessment = evaluate_evidence_policy(
        "今年诺贝尔奖获奖者名单", [complete_source],
        official_domains=("official.example",),
    )
    assert assessment.sufficient is True

def test_same_url_never_counts_twice():
    assert evaluate_evidence_policy("今年获奖名单", [source, duplicate]).sufficient is False
```

- [ ] **Step 2: 运行策略测试确认失败**

Run: `uv run pytest tests/unit/agent/test_evidence_policy.py -v`

Expected: FAIL，因为策略模块不存在。

- [ ] **Step 3: 实现 `evidence_policy.py` 和配置读取**

`current_complete_list` 识别“今年/最新”与“获奖者/名单/完整列表”组合；完整名单要求正文同时
覆盖时间词和名单/全部语义。仅 HTTPS 且 hostname 精确匹配传入配置域名的来源可走官方例外。
在 `cli._graph_options()` 或独立 `_web_options()` 校验 `official_domains` 是字符串列表，非法值
视为默认空元组。

- [ ] **Step 4: 将评估接入两种图执行器**

在 `GraphAgentRunner._assess_evidence()` 和 `LangGraphResearchRunner._assess()` 使用相同纯函数：
未足够时写入 `EvidenceStatus.PARTIAL`/`INSUFFICIENT`、缺口和策略诊断；候选仍存在时继续读取，
耗尽后才重新规划/诊断。只有 `assessment.sufficient` 时允许 synthesize 调用 summarizer。

- [ ] **Step 5: 运行图与策略测试确认通过**

Run: `uv run pytest tests/unit/agent/test_evidence_policy.py tests/unit/agent/test_graph.py tests/integration/test_research_graph_parity.py -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/researchflow/agent/evidence_policy.py src/researchflow/agent/graph.py src/researchflow/agent/research_graph.py src/researchflow/cli.py tests/unit/agent tests/integration/test_research_graph_parity.py
git commit -m "feat: require sufficient list evidence"
```

### Task 3: 持久化问题和策略诊断到运行快照

**Files:**
- Modify: `src/researchflow/application/models.py`
- Modify: `src/researchflow/application/service.py`
- Modify: `src/researchflow/application/run_store.py`
- Test: `tests/application/test_models.py`
- Test: `tests/application/test_service.py`
- Test: `tests/application/test_run_store.py`
- Test: `tests/api/test_routes.py`

**Interfaces:**
- Produces: `RunSnapshot.question: str`、`evidence_policy: str | None`、`accepted_source_count: int`、`required_source_count: int` 和 `official_complete_source_id: str | None`。
- Consumes: 运行开始时的 `StartTurn.message` 与 `evidence_assessed` 应用事件。

- [ ] **Step 1: 写失败测试，断言问题和策略可重建**

```python
def test_run_snapshot_preserves_question_and_evidence_policy_after_reload(tmp_path):
    snapshot = service.start_turn(StartTurn(session_id="one", message="今年获奖名单", ...))
    reloaded = ResearchService(database, workflow=workflow).get_run(snapshot.run_id)
    assert reloaded.question == "今年获奖名单"
    assert reloaded.evidence_policy == "current_complete_list"
    assert reloaded.required_source_count == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/application/test_models.py tests/application/test_service.py tests/application/test_run_store.py tests/api/test_routes.py -v`

Expected: FAIL，因为快照没有 `question` 和策略字段。

- [ ] **Step 3: 实现 DTO 和事件投射**

创建 run 时保存 `question=request.message`；在 `ResearchService._project_event()` 读取
`evidence_assessed` 的安全字段。保留既有 SQLite JSON payload 兼容性：缺少新字段的旧记录使用
空字符串/`None`/零值。

- [ ] **Step 4: 运行应用/API 测试确认通过**

Run: `uv run pytest tests/application tests/api -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/researchflow/application tests/application tests/api
git commit -m "feat: persist run questions and evidence policy"
```

### Task 4: 控制台会话历史、来源卡片与基础布局

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/App.css`
- Modify: `frontend/src/main.tsx`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Produces: `getSession(sessionId: string): Promise<SessionSnapshot>`，其中 `SessionSnapshot` 含有按时间排序的 `runs`。
- Consumes: Task 3 的 `RunSnapshot.question` 和证据策略字段，以及现有 `getRun`、SSE hook。

- [ ] **Step 1: 写失败前端测试，覆盖会话切换和来源链接**

```tsx
it("loads the newest persisted run when a saved session is selected", async () => {
  // mock /api/sessions/demo with two runs
  fireEvent.click(screen.getByRole("button", { name: "demo" }));
  expect(await screen.findByText("第二轮问题")).toBeInTheDocument();
});

it("switches a history item without posting another turn", async () => {
  fireEvent.click(screen.getByRole("button", { name: "第一轮问题" }));
  expect(screen.getByText("第一轮回答")).toBeInTheDocument();
  expect(fetchMock).not.toHaveBeenCalledWith(expect.stringContaining("/turns"), expect.anything());
});
```

- [ ] **Step 2: 运行前端测试确认失败**

Run: `npm --prefix frontend test -- --run src/App.test.tsx`

Expected: FAIL，因为当前点击会话只清空报告。

- [ ] **Step 3: 实现 `SessionSnapshot` API 和会话选择状态**

`App` 维护 `sessionRuns`、`selectedRunId` 和 `loadedRun`。点击会话调用 `getSession()`，默认
选择 `runs.at(-1)`；点击历史项目只赋值 `loadedRun`。请求失败时清空旧快照并显示错误，不提交研究。

- [ ] **Step 4: 实现来源卡片和样式**

创建 `App.css` 并在 `main.tsx` 导入。来源 URL 使用 `<a href target="_blank" rel="noreferrer">`；
候选标注“未读取正文”，拒绝项显示 reason。为 completed、waiting、insufficient 设独立状态样式，
`insufficient_evidence` 不能使用成功颜色。

- [ ] **Step 5: 运行前端测试和构建确认通过**

Run: `npm --prefix frontend test -- --run && npm --prefix frontend run build`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add frontend/src
git commit -m "feat: browse session history in console"
```

### Task 5: 跨层验收、文档与受控真实验证

**Files:**
- Modify: `tests/integration/test_web_console_acceptance.py`
- Modify: `docs/architecture.md`
- Modify: `docs/local-web-console.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1–4 的网页错误码、策略事件、快照字段及控制台 API。
- Produces: 离线端到端验收与可操作的 `researchflow.toml` 官方域名配置说明。

- [ ] **Step 1: 写失败跨层验收测试**

```python
def test_insufficient_list_evidence_is_not_grounded_or_completed(...):
    snapshot = client.post(...).json()
    assert snapshot["status"] == "insufficient_evidence"
    assert snapshot["generation_mode"] != "llm_grounded"

def test_session_snapshot_exposes_multiple_persisted_turns(...):
    assert [run["question"] for run in response.json()["runs"]] == ["first", "second"]
```

- [ ] **Step 2: 运行验收测试确认失败**

Run: `uv run pytest tests/integration/test_web_console_acceptance.py -v`

Expected: FAIL，直到策略字段和会话 UI/API 投射全部完成。

- [ ] **Step 3: 更新文档**

记录 `researchflow.toml`：

```toml
[web]
official_domains = ["example-official.org"]
```

说明这是单来源完整名单例外，默认空；记录历史浏览、候选/正文/拒绝分区和 `insufficient_evidence` 行为。

- [ ] **Step 4: 运行完整离线验证**

Run: `uv sync --frozen --extra llm --extra ui && uv lock --check && uv run pytest && uv run ruff check . && uv run ruff format --check . && npm --prefix frontend test -- --run && npm --prefix frontend run build`

Expected: 全部 PASS。

- [ ] **Step 5: 受控真实入口验证（若环境已配置）**

Run: `uv run researchflow chat "今年的诺贝尔奖获奖者都有哪些" --session-id evidence-quality-demo --enable-web --agent-mode llm --output-dir output/evidence-quality-demo`

Expected: 乱码/低质量来源出现在拒绝原因中；若未满足两来源或官方完整名单例外，输出 `insufficient_evidence` 而非 `llm_grounded`。记录成功读取数、采用来源和限制，不打印密钥。

- [ ] **Step 6: 提交**

```powershell
git add tests/integration docs README.md
git commit -m "test: cover session evidence acceptance"
```

## Plan Self-Review

- Spec coverage：Task 1 实现解码后质量屏障；Task 2 实现数量与官方例外；Task 3 使历史问题与策略可持久化；Task 4 消费历史并改善呈现；Task 5 端到端验收和文档。
- Type consistency：Task 2 的策略事件由 Task 3 的 `RunSnapshot` 投射，Task 4 的 `SessionSnapshot` 消费同一字段；Task 5 只使用此前定义的端点和错误码。
- Review Focus：五项输入/恢复风险均在所属任务明确附带测试。
- Proportion：计划定义接口和可验证行为，不转录实现代码；每项任务可独立 TDD、验证与提交。
