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
  if (sources.length === 0) {
    return <p>无</p>;
  }
  return (
    <ul>
      {sources.map((source) => (
        <li key={source.source_id}>
          {source.url ? (
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>
          ) : (
            source.title
          )}
          {source.reason ? `：${source.reason}` : ""}
        </li>
      ))}
    </ul>
  );
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
      <aside className="panel">
        <h1>会话</h1>
        <button type="button" onClick={startNewSession}>
          新建会话
        </button>
        {sessionId ? <p>当前会话：{sessionId}</p> : null}
        <ul>
          {sessions.map((item) => (
            <li key={item}>
              <button
                type="button"
                onClick={() => void selectSession(item)}
              >
                {item}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section className="report-panel">
        <h2>研究报告</h2>
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
        {snapshot?.question ? <h3>{snapshot.question}</h3> : null}
        {snapshot ? <p>运行状态：{snapshot.status}</p> : null}
        {snapshot?.interrupt_prompt ? <p>{snapshot.interrupt_prompt}</p> : null}
        {snapshot?.answer ? <pre>{snapshot.answer}</pre> : null}
        {loadError ? <p role="alert">{loadError}</p> : null}
        <textarea
          aria-label="研究问题"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />
        <button type="button" onClick={submitTurn}>
          {snapshot?.status === "waiting_for_input" ? "继续研究" : "发送"}
        </button>
      </section>
      <aside className="panel evidence-panel">
        <h2>搜索候选</h2>
        <SourceList sources={snapshot?.evidence.candidates ?? []} />
        <h2>已读取正文</h2>
        <SourceList sources={snapshot?.evidence.read_sources ?? []} />
        <h2>拒绝或读取失败</h2>
        <SourceList sources={snapshot?.evidence.rejected_sources ?? []} />
      </aside>
    </main>
  );
}
