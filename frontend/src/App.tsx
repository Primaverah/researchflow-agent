import { useEffect, useState } from "react";

import {
  getRun,
  listSessions,
  startTurn,
  type RunSnapshot,
  type SourceSnapshot,
} from "./api";

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
          {source.title}
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
  const [message, setMessage] = useState("");

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

  async function submitTurn() {
    const content = message.trim();
    if (!content || !sessionId) {
      return;
    }
    try {
      const nextRun = await startTurn(sessionId, content);
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
    setLoadError("");
  }

  const snapshot = loadedRun ?? run;
  return (
    <main>
      <aside>
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
                onClick={() => {
                  setSessionId(item);
                  setLoadedRun(undefined);
                  setLoadError("");
                }}
              >
                {item}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section>
        <h2>研究报告</h2>
        {snapshot ? <p>运行状态：{snapshot.status}</p> : null}
        {loadError ? <p role="alert">{loadError}</p> : null}
        <textarea
          aria-label="研究问题"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />
        <button type="button" onClick={submitTurn}>
          发送
        </button>
      </section>
      <aside>
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
