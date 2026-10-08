import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("keeps candidates, read sources, and rejected sources distinct", () => {
    render(
      <App
        run={{
          run_id: "run-1",
          session_id: "session-1",
          status: "completed",
          evidence: {
            candidates: [
              {
                source_id: "candidate",
                title: "仅搜索摘要",
                url: "https://example.test/candidate",
                kind: "web",
                read: false,
                reason: null,
              },
            ],
            read_sources: [
              {
                source_id: "read",
                title: "已读正文",
                url: "https://example.test/read",
                kind: "web",
                read: true,
                reason: null,
              },
            ],
            rejected_sources: [
              {
                source_id: "rejected",
                title: "拒绝来源",
                url: "https://example.test/rejected",
                kind: "web",
                read: false,
                reason: "web_low_quality_content",
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "会话" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "搜索候选" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "已读取正文" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "拒绝或读取失败" })).toBeInTheDocument();
    expect(screen.getByText("仅搜索摘要")).toBeInTheDocument();
    expect(screen.getByText("已读正文")).toBeInTheDocument();
    expect(
      screen.getByText("拒绝来源：web_low_quality_content"),
    ).toBeInTheDocument();
  });
});
