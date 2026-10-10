# 会话历史与网页证据质量设计

## 目标

将本地 Web 控制台从“仅显示本轮快照”的功能骨架升级为可浏览持久会话历史的单用户
研究界面，并防止乱码、导航页、低质量正文或不足数量的来源被当作 grounded evidence。

成功标准：

1. 用户选择任一已有会话时，能看到其中每一轮运行，并可切换查看该轮报告和来源。
2. 乱码、空标题/模板、过短正文和低自然语言密度网页被拒绝，拒绝原因可见且不交给
   LLM 总结。
3. 时效性名单/获奖者列表默认至少需要两个独立有效来源；一条“官方且正文完整”的来源
   是唯一例外。
4. 未达到证据门槛的运行是 `insufficient_evidence`，不得显示 `llm_grounded`。
5. 控制台将候选、已读正文、拒绝来源和历史轮次清晰分开，并以最小可用布局呈现。

## 范围与非目标

范围包括本地 FastAPI、持久化运行快照、React 控制台、网页文本质量判断和研究图证据
评估。现有 SQLite 数据库和运行 ID 保持不变。

不实现账户、云端同步、跨用户协作、通用网页可信度评分或强制所有问题至少两条来源。
普通事实问题仍可由一条合格来源支持；该规则只收紧“时效性完整列表”意图。

## 数据与 API

`GET /api/sessions/{session_id}` 已返回 `SessionSnapshot` 和有序 `runs`，前端将正式消费
该接口。`RunSnapshot` 增加（或通过现有字段稳定投射）以下展示数据：

- `question`：本轮用户问题；由应用服务在创建/完成运行时保存，不能从报告正文反解析。
- `created_at`、`updated_at`：用于历史轮次标签。
- `evidence_status`、`evidence_gaps`、`generation_mode`：用于状态徽标和诊断。
- `evidence.candidates`、`read_sources`、`rejected_sources`：保持来源类别与失败原因。

保留单独的 `GET /api/runs/{run_id}` 用于事件重放后的刷新。会话切换只加载持久化快照，
不重新执行研究。

## 前端交互与视觉结构

页面维持三栏，但引入语义化卡片和基础 CSS：

```text
会话栏                         报告栏                         证据栏
├─ 新建会话                    ├─ 历史轮次（问题/状态/时间）  ├─ 搜索候选
├─ 会话列表                    ├─ 当前轮问题与状态            ├─ 已读取正文
└─ 选中会话                    ├─ 回答或澄清                  └─ 拒绝/读取失败
                               └─ 输入框与发送/继续研究
```

- 点击会话：获取该会话 runs，默认选择最后一轮；点击历史轮次仅切换已保存快照。
- 候选仅显示“搜索摘要，未读取正文”；已读与拒绝来源显示域名、标题、链接和原因。
- `insufficient_evidence` 使用明确的警告状态，不能以完成/grounded 样式表示。
- 所有回答按纯文本渲染，外链使用普通安全锚点，不注入网页 HTML。

## 网页读取质量

`SafeHttpClient.fetch` 在 HTML 提取后增加独立的 `assess_text_quality` 判断。它不依赖
特定网站，输入为标题、正文和请求语言线索，输出接受或一个稳定错误码。

拒绝条件：

- 标题为默认/无意义标题且正文不足；
- 正文长度不足以支持事实回答；
- Unicode replacement character、控制字符或典型 mojibake 序列比例超过阈值；
- 中文查询的正文正常中文字符比例过低，同时出现高比例 Latin-1 乱码模式；
- 提取结果主要是链接、导航或模板文本。

错误码使用 `web_garbled_content` 或 `web_low_quality_content`。解码依旧按 HTTP charset、
meta charset、UTF-8、GB18030、GBK 尝试；质量判断是解码成功后的第二道防线，而不是
关闭 URL/SSRF 检查或盲目更换编码。

## 来源数量与证据策略

图状态新增结构化的 `evidence_policy`，从 query/intent 推导：

- `standard`：一条相关、成功读取的正文可支持普通问题。
- `current_complete_list`：对“今年/最新 + 获奖者/名单/完整列表”等时效性列表问题，至少
  两个不同 URL 的有效正文。
- `official_complete_list`：若单一来源经通用官方域名/页面信号判定且正文覆盖请求的完整
  列表，则可满足 `current_complete_list` 的例外。

官方例外必须同时满足：HTTPS、域名精确匹配项目配置的 `web.official_domains`、标题/正文
明确表明年度和完整名单、内容质量通过。默认配置为空，避免把任意 `.org` 或 URL 中含有
`official` 的页面擅自认定为官方；用户可按研究领域配置官方域名，不能在代码中硬编码
Nobel、人物或游戏网站。

图在每批读取后重新评估。未达到门槛且仍有候选时继续读取；候选耗尽时，有限重新规划
更互补的查询。仍不足则产生诊断报告，`evidence_status=INSUFFICIENT`，不调用 LLM
事实总结。只有达到门槛的 sources 可以流入 summarizer；`generation_mode=llm_grounded`
也必须以该前置条件为真。

## 可观测性与恢复

新增或扩展安全事件字段：`evidence_policy`、`accepted_source_count`、`required_source_count`、
`official_complete_source_id`（如适用）及每个网页质量拒绝原因。正文、原始 HTTP body 和
密钥不写入事件。

所有字段保持 JSON-safe，随 `run_id` checkpoint 保存；会话浏览只能读取已保存状态，不能
触发工具、网页请求或文件写入。

## 测试策略

Python：

1. GBK/GB18030 正文仍可正常解码；Latin-1/UTF-8 误解产生的 mojibake 被拒绝。
2. 低质量/无标题模板页被拒绝，不能成为 `WebSource`。
3. 时效性名单的一条普通来源为 PARTIAL/INSUFFICIENT；两个独立来源为 SUFFICIENT。
4. 一条合格官方完整名单为 SUFFICIENT；相似但非官方/不完整来源不享有例外。
5. 不足证据不会调用 LLM summarizer，运行快照不是 completed/grounded。
6. API 会话端点返回历史 runs，重建服务后排序与内容不变。

前端：

1. 点击会话发起会话快照请求、选择最新 run、点击历史条目切换报告。
2. 候选/正文/拒绝来源分区和链接可见，`insufficient_evidence` 有明确状态。
3. 新建会话不污染旧会话或其历史。

端到端离线验收使用假的搜索/抓取/LLM，不依赖外网。真实网页测试仅作为受控手动验证，
其结果不作为单元测试的替代。
