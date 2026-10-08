import { useEffect, useState } from "react";

import { getRun, type RunSnapshot, type SourceSnapshot } from "./api";

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

  useEffect(() => {
    if (!runId) {
      return;
    }
    getRun(runId).then(setLoadedRun).catch(() => setLoadError("无法读取运行状态"));
  }, [runId]);

  const snapshot = loadedRun ?? run;
  return (
    <main>
      <aside>
        <h1>会话</h1>
        <button type="button">新建会话</button>
      </aside>
      <section>
        <h2>研究报告</h2>
        {snapshot ? <p>运行状态：{snapshot.status}</p> : null}
        {loadError ? <p role="alert">{loadError}</p> : null}
        <textarea aria-label="研究问题" />
        <button type="button">发送</button>
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
