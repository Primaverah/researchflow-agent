export function App() {
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
        <p>尚无候选</p>
        <h2>已读取正文</h2>
        <p>尚无已读取来源</p>
        <h2>拒绝或读取失败</h2>
        <p>尚无拒绝来源</p>
      </aside>
    </main>
  );
}
