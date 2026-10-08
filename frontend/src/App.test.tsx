import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("keeps candidates, read sources, and rejected sources distinct", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "会话" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "搜索候选" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "已读取正文" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "拒绝或读取失败" })).toBeInTheDocument();
  });
});
