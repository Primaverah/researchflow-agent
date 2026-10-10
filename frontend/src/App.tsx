import { useEffect, useState } from "react";

import {
  getRun,
  getSession,
  listSessions,
  resumeTurn,
  startTurn,
  type RunSnapshot,
  type SourceSnapshot,
} from "./api";
import { useRunEvents } from "./useRunEvents";

type AppProps = {
  run?: RunSnapshot;
  runId?: string;
};

function SourceList({ sources }: { sources: SourceSnapshot[] }) {
  return (
    <ul className="source-list">
      {sources.map((source) => (
        <li className="source-item" key={source.source_id}>
          <span className="source-kind">{source.kind === "web" ? "网页" : "文档"}</span>
          <span className="source-content">
            {source.url ? (
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.title}
              </a>
            ) : (
              source.title
            )}
            {source.reason ? <small>原因：{source.reason}</small> : null}
          </span>
        </li>
      ))}
    </ul>
  );
}

function EvidencePanel({ snapshot }: { snapshot?: RunSnapshot }) {
  const evidence = snapshot?.evidence;
  const groups = [
    { label: "搜索候选（未读取正文）", sources: evidence?.candidates ?? [] },
    { label: "已读取正文", sources: evidence?.read_sources ?? [] },
    { label: "拒绝或读取失败", sources: evidence?.rejected_sources ?? [] },
  ];
  const populatedGroups = groups.filter((group) => group.sources.length > 0);

  return (
    <aside className="panel evidence-panel" aria-label="研究证据">
      <div className="panel-heading">
        <p className="eyebrow">证据轨迹</p>
        <h2>研究证据</h2>
        {snapshot?.evidence_status ? (
          <span className="evidence-status">{snapshot.evidence_status}</span>
        ) : null}
      </div>
      {populatedGroups.length === 0 ? (
        <p className="empty-evidence">暂无可展示的研究证据</p>
      ) : (
        populatedGroups.map((group) => (
          <section className="evidence-group" key={group.label}>
            <h3>{group.label}</h3>
            <SourceList sources={group.sources} />
          </section>
        ))
      )}
    </aside>
  );
}

function compactId(value: string) {
  return value.length > 18 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

export function App({ run, runId }: AppProps) {
  const [loadedRun, setLoadedRun] = useState<RunSnapshot | undefined>(run);
  const [loadError, setLoadError] = useState("");
  const [sessions, setSessions] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [sessionRuns, setSessionRuns] = useState<RunSnapshot[]>([]);
  const [message, setMessage] = useState("");
  const snapshot = loadedRun ?? run;

  useEffect(() => {
    if (!runId) {
      return;
    }
    getRun(runId).then(setLoadedRun).catch(() => setLoadError("无法读取运行状态"));
  }, [runId]);

  useEffect(() => {
    listSessions()
      .then((items) => {
        setSessions(items);
        setSessionId((current) => current || items[0] || crypto.randomUUID());
      })
      .catch(() => setLoadError("无法读取会话列表"));
  }, []);

  useRunEvents(
    snapshot?.run_id,
    () => {
      if (snapshot) {
        getRun(snapshot.run_id)
          .then(setLoadedRun)
          .catch(() => setLoadError("实时更新失败，正在使用轮询状态"));
      }
    },
    () => {
      if (snapshot) {
        getRun(snapshot.run_id)
          .then(setLoadedRun)
          .catch(() => setLoadError("无法读取运行状态"));
      }
    },
  );

  async function submitTurn() {
    const content = message.trim();
    if (!content || !sessionId) {
      return;
    }
    try {
      const nextRun =
        snapshot?.status === "waiting_for_input"
          ? await resumeTurn(sessionId, content)
          : await startTurn(sessionId, content);
      setLoadedRun(nextRun);
      setMessage("");
      setSessions((items) =>
        items.includes(sessionId) ? items : [...items, sessionId],
      );
    } catch {
      setLoadError("无法启动研究");
    }
  }

  function startNewSession() {
    setSessionId(crypto.randomUUID());
    setLoadedRun(undefined);
    setSessionRuns([]);
    setLoadError("");
  }

  async function selectSession(nextSessionId: string) {
    setSessionId(nextSessionId);
    setLoadedRun(undefined);
    setSessionRuns([]);
    setLoadError("");
    try {
      const session = await getSession(nextSessionId);
      setSessionRuns(session.runs);
      setLoadedRun(session.runs.at(-1));
    } catch {
      setLoadError("无法读取会话历史");
    }
  }

  return (
    <main className="app-shell">
      <aside className="panel session-panel">
        <div className="panel-heading">
          <p className="eyebrow">ResearchFlow</p>
          <h1>会话</h1>
        </div>
        <button className="primary-button" type="button" onClick={startNewSession}>
          新建会话
        </button>
        {sessionId ? (
          <p className="current-session" title={sessionId}>
            当前会话：{compactId(sessionId)}
          </p>
        ) : null}
        <ul className="session-list">
          {sessions.map((item) => (
            <li key={item}>
              <button
                type="button"
                onClick={() => void selectSession(item)}
                title={item}
                className={item === sessionId ? "selected" : ""}
              >
                {compactId(item)}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section className="report-panel" aria-live="polite">
        <div className="report-heading">
          <div>
            <p className="eyebrow">研究结果</p>
            <h2>研究报告</h2>
          </div>
          {snapshot ? <span className="run-status">{snapshot.status}</span> : null}
        </div>
        {sessionRuns.length > 0 ? (
          <div className="history" aria-label="会话历史">
            {sessionRuns.map((item) => (
              <button
                type="button"
                key={item.run_id}
                onClick={() => setLoadedRun(item)}
                className={item.run_id === snapshot?.run_id ? "selected" : ""}
              >
                {item.question || "未命名问题"}
              </button>
            ))}
          </div>
        ) : null}
        {snapshot?.question ? <h3 className="question-title">{snapshot.question}</h3> : null}
        {snapshot?.interrupt_prompt ? (
          <p className="interrupt-prompt">{snapshot.interrupt_prompt}</p>
        ) : null}
        {snapshot?.answer ? <pre className="answer">{snapshot.answer}</pre> : null}
        {loadError ? <p role="alert" className="error-message">{loadError}</p> : null}
        <div className="composer">
          <textarea
            aria-label="研究问题"
            placeholder="输入需要研究的问题…"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
          />
          <button className="primary-button" type="button" onClick={submitTurn}>
            {snapshot?.status === "waiting_for_input" ? "继续研究" : "发送"}
          </button>
        </div>
      </section>
      <EvidencePanel snapshot={snapshot} />
    </main>
  );
}
