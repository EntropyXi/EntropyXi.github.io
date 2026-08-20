import { describe, expect, it } from "vitest";
import { buildPostPath } from "@/lib/routing/post-path";

describe("buildPostPath", () => {
  it("encodes each permalink segment and keeps trailing slash", () => {
    expect(buildPostPath("2026/03/15/深度学习/流匹配与扩散模型/DDPM")).toBe(
      "/2026/03/15/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%B5%81%E5%8C%B9%E9%85%8D%E4%B8%8E%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B/DDPM/",
    );
  });

  it("preserves spaces as %20 in segments", () => {
    expect(buildPostPath("2026/05/17/深度学习/1. 从SDE开始")).toBe(
      "/2026/05/17/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/1.%20%E4%BB%8ESDE%E5%BC%80%E5%A7%8B/",
    );
  });
});
