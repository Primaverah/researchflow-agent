import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

describe("App", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

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

  it("renders a final answer as plain text instead of source HTML", () => {
    render(
      <App
        run={{
          run_id: "run-1",
          session_id: "session-1",
          status: "completed",
          answer: "<strong>受控回答</strong>",
          evidence: { candidates: [], read_sources: [], rejected_sources: [] },
        }}
      />,
    );

    expect(screen.getByText("<strong>受控回答</strong>")).toBeInTheDocument();
    expect(screen.queryByText("受控回答", { selector: "strong" })).toBeNull();
  });

  it("loads a supplied run without submitting another turn", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: "run-1",
          session_id: "session-1",
          status: "completed",
          evidence: {
            candidates: [],
            read_sources: [],
            rejected_sources: [],
          },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App runId="run-1" />);

    expect(await screen.findByText("运行状态：completed")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/runs/run-1");
  });

  it("lists sessions and starts a turn from the composer", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(["saved-session"])))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            run_id: "run-2",
            session_id: "saved-session",
            status: "completed",
            evidence: {
              candidates: [],
              read_sources: [],
              rejected_sources: [],
            },
          }),
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    expect(await screen.findByText("saved-session")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("研究问题"), {
      target: { value: "tool calling" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("运行状态：completed")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/sessions/saved-session/turns",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("starts a new local session before sending its first turn", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(["saved-session"])))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            run_id: "run-3",
            session_id: "new-session",
            status: "completed",
            evidence: {
              candidates: [],
              read_sources: [],
              rejected_sources: [],
            },
          }),
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { randomUUID: () => "new-session" });

    render(<App />);
    await screen.findByText("saved-session");
    fireEvent.click(screen.getByRole("button", { name: "新建会话" }));

    expect(screen.getByText("当前会话：new-session")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("研究问题"), {
      target: { value: "new question" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/sessions/new-session/turns",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
