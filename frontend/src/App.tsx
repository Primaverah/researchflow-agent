import type { RunSnapshot, SourceSnapshot } from "./api";

type AppProps = {
  run?: RunSnapshot;
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

export function App({ run }: AppProps) {
  return (
    <main>
      <aside>
        <h1>会话</h1>
        <button type="button">新建会话</button>
      </aside>
      <section>
        <h2>研究报告</h2>
        <textarea aria-label="研究问题" />
        <button type="button">发送</button>
      </section>
      <aside>
        <h2>搜索候选</h2>
        <SourceList sources={run?.evidence.candidates ?? []} />
        <h2>已读取正文</h2>
        <SourceList sources={run?.evidence.read_sources ?? []} />
        <h2>拒绝或读取失败</h2>
        <SourceList sources={run?.evidence.rejected_sources ?? []} />
      </aside>
    </main>
  );
}
