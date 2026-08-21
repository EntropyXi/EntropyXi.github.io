import { describe, expect, it } from "vitest";

import {
  applyMathAccessibleNames,
  collectMathSources,
  normalizeMathAccessibleName,
} from "@/lib/markdown/math-accessibility";

describe("math accessibility", () => {
  it("normalizes whitespace without changing TeX commands", () => {
    expect(normalizeMathAccessibleName("  x^2  +\n  \\text{中文} ")).toBe(
      "数学公式：x^2 + \\text{中文}",
    );
  });

  it("preserves inline and display source order", () => {
    const tree = {
      type: "root",
      children: [
        {
          type: "element",
          tagName: "span",
          properties: { className: ["math-inline"] },
          children: [{ type: "text", value: "x + y" }],
        },
        {
          type: "element",
          tagName: "div",
          properties: { className: ["math-display"] },
          children: [
            { type: "text", value: "\\begin{aligned}a&=b\\end{aligned}" },
          ],
        },
      ],
    };

    expect(collectMathSources(tree)).toEqual([
      { display: false, source: "x + y" },
      { display: true, source: "\\begin{aligned}a&=b\\end{aligned}" },
    ]);
  });

  it("labels each rendered SVG and rejects count mismatches", () => {
    const svgProperties: Record<string, unknown> = { role: "img" };
    const formula = {
      type: "element",
      tagName: "mjx-container",
      children: [
        {
          type: "element",
          tagName: "svg",
          properties: svgProperties,
          children: [],
        },
      ],
    };
    const tree = { type: "root", children: [formula] };

    expect(
      applyMathAccessibleNames(tree, [{ display: false, source: "x^2" }]),
    ).toBe(1);
    expect(svgProperties.ariaLabel).toBe("数学公式：x^2");
    expect(() => applyMathAccessibleNames(tree, [])).toThrow(
      /without a captured TeX source/u,
    );
  });
});
