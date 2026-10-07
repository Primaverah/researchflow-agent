# 网页主内容、证据质量与 LLM 回退可见性设计

## 目标

让 Web research 模式只把可回答当前问题的网页主内容交给回答层；当 LLM 可用时优先
由它基于证据直接回答；当 LLM 失败时，安全地回退并让报告明确展示回退原因。

本设计解决一次真实运行：问题“村上春树是谁”抓取到了界面新闻的站点导航。该运行的
JSONL 显示 LLM planner 与 summarizer 均遇到 `AuthenticationError` / HTTP 401，因此
系统使用了抽取式回退，而抽取器又选取了含导航栏的页面开头文本。

## 非目标

- 不放宽 URL、SSRF、混合 DNS 或来源白名单安全策略。
- 不把搜索摘要伪装为已读网页正文。
- 不要求 LLM 在无有效证据时凭内部知识作答。
- 本次不实现安全重定向；现有的安全拒绝行为保持不变。

## 设计

### 1. 主内容提取

`SafeHttpClient` 的 HTML 解析改为收集结构化文本块而不是将所有可见文本拼接成一个
字符串。

优先级如下：

1. `<article>`、`<main>`、`role="main"`；
2. 名称含 `article`、`content`、`post`、`entry`、`story`、`detail` 的容器；
3. 无语义容器时，按段落长度、非链接文本比例、句子标点和正文关键词对块评分。

`nav`、`header`、`footer`、`aside`、`form`、脚本、样式、广告、菜单、版权、推荐容器
不进入正文候选。若候选只有短菜单/模板文本，`fetch_url` 返回可分类的
`web_low_quality_content` 失败，而不是成功的 `WebSource`。

### 2. 证据质量门槛

图编排器对成功抓取的正文执行可解释的意图检查：

- 编译器：正文含编译器名称或“编译器”语义证据；
- 年龄：正文含出生日期或出生年份证据；
- 代表作：正文含作品/电影/著作等作品证据；
- 人物简介（“X 是谁/是谁”）：正文同时含主体及至少一项身份、职业、国籍、出生、代表作
  或“作家/演员/科学家”等传记证据。

每个拒绝来源保留 URL 与机器可读原因，并在图 Trace 内记录。只有通过该检查的正文可
成为 `WebSource` evidence。

### 3. LLM 优先与回退结果

`LLMSummarizer` 继续只接收已验证正文。在 LLM 模式下它是唯一的首次总结尝试；模型
调用、结构化输出、引用集合或语言约束失败后才使用确定性摘要器。

总结边界改为返回文本及一个安全的生成状态：

- `llm_grounded`：LLM 产生且引用集合已验证；
- `extractive_fallback`：确定性摘要器产生；附脱敏原因，例如 `provider_error`、
  `invalid_citation`、`invalid_schema`、`language_constraint`。

图将该状态写入 JSONL `llm_decision`/graph trace，并在最终 Markdown 的“回答生成状态”
节展示。诊断不得包含 API key、Authorization 头、原始提示、完整网页内容或原始模型
响应。

### 4. 无有效证据的行为

若所有抓取页面都因正文质量或问题证据门槛被拒绝，则报告使用 `INSUFFICIENT`，列出
搜索候选及拒绝原因，不调用总结模型来编造简介。

## 数据流

```text
web_search candidates
  -> fetch_url / HTML block extraction
  -> reject boilerplate-only pages
  -> intent-specific evidence quality gate
  -> accepted WebSource evidence
  -> LLM grounded summary
       -> success: llm_grounded answer
       -> safe failure: extractive_fallback + visible reason
```

## 测试与验收

新增离线回归测试：

1. 无语义标签但有长导航、短正文的 HTML，只保留正文；纯导航 HTML 被拒绝。
2. “人物是谁”不能接受只提及名字、没有传记证据的页面；可接受含身份/职业证据的正文。
3. LLM 总结成功时，报告标明 `llm_grounded`；失败时先尝试模型、再回退，并显示脱敏原因。
4. 无合格来源时，报告标明证据不足且包含拒绝理由。
5. 原有网页安全、GBK/GB18030、离线、会话和全量测试不回归。

真实验收使用：

```powershell
uv run researchflow chat "村上春树是谁" --session-id cscs-demo2 --enable-web --agent-mode llm
```

检查最终 Markdown/JSONL：导航文本不进入摘要；来源是成功读取且通过人物简介门槛的
正文；若 LLM 认证或服务失败，报告明确显示安全的回退原因。
