import { describe, expect, it } from "vitest";
import {
  countH1Elements,
  hasTopLevelMarkdownHeading,
} from "@/lib/audit/heading-uniqueness";

describe("countH1Elements", () => {
  it("returns 0 when no top-level heading exists", () => {
    expect(countH1Elements("<p>hello</p><h2>section</h2>")).toBe(0);
  });

  it("counts a single h1 opening tag", () => {
    expect(countH1Elements('<h1 class="post-title">标题</h1>')).toBe(1);
  });

  it("detects duplicate h1 headings", () => {
    expect(countH1Elements("<h1>标题</h1><h1>标题</h1>")).toBe(2);
  });

  it("ignores closing tags and lookalike elements", () => {
    expect(countH1Elements("</h1><h1x><h1>ok</h1>")).toBe(1);
  });
});

describe("hasTopLevelMarkdownHeading", () => {
  it("rejects a body without headings", () => {
    expect(hasTopLevelMarkdownHeading("这是正文第一段。")).toBe(false);
  });

  it("accepts headings that start at level 2", () => {
    expect(hasTopLevelMarkdownHeading("## 章节\n### 小节")).toBe(false);
  });

  it("flags a top-level H1 heading", () => {
    expect(hasTopLevelMarkdownHeading("# WSL2 安装失败排查")).toBe(true);
  });

  it("flags an H1 heading without a space after the hash", () => {
    expect(hasTopLevelMarkdownHeading("#标题")).toBe(true);
  });
});
