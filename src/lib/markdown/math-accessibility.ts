const MATH_SOURCES_KEY = "entropyxiMathSources";
const ACCESSIBLE_NAME_PREFIX = "数学公式：";

interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

interface MathSource {
  display: boolean;
  source: string;
}

interface MathVFile {
  data: Record<string, unknown>;
}

function classNames(node: HastNode): string[] {
  const value = node.properties?.className;
  if (Array.isArray(value))
    return value.filter((item): item is string => typeof item === "string");
  return typeof value === "string" ? value.split(/\s+/u) : [];
}

function textContent(node: HastNode): string {
  if (node.type === "text") return node.value ?? "";
  return node.children?.map(textContent).join("") ?? "";
}

function walk(node: HastNode, visitor: (node: HastNode) => void): void {
  visitor(node);
  for (const child of node.children ?? []) walk(child, visitor);
}

function findFirstSvg(node: HastNode): HastNode | undefined {
  if (node.type === "element" && node.tagName === "svg") return node;
  for (const child of node.children ?? []) {
    const svg = findFirstSvg(child);
    if (svg) return svg;
  }
  return undefined;
}

export function normalizeMathAccessibleName(source: string): string {
  return `${ACCESSIBLE_NAME_PREFIX}${source.replace(/\s+/gu, " ").trim()}`;
}

export function collectMathSources(tree: HastNode): MathSource[] {
  const sources: MathSource[] = [];

  walk(tree, (node) => {
    if (node.type !== "element") return;
    const classes = classNames(node);
    const languageMath = classes.includes("language-math");
    const mathDisplay = classes.includes("math-display");
    const mathInline = classes.includes("math-inline");
    if (!languageMath && !mathDisplay && !mathInline) return;

    sources.push({
      display: languageMath || mathDisplay,
      source: textContent(node),
    });
  });

  return sources;
}

export function applyMathAccessibleNames(
  tree: HastNode,
  sources: MathSource[],
): number {
  let renderedCount = 0;

  walk(tree, (node) => {
    if (node.type !== "element" || node.tagName !== "mjx-container") return;
    const source = sources[renderedCount];
    if (!source) {
      throw new Error(
        `MathJax rendered formula ${renderedCount + 1} without a captured TeX source`,
      );
    }

    const svg = findFirstSvg(node);
    if (!svg) {
      throw new Error(
        `MathJax formula ${renderedCount + 1} does not contain an SVG output node`,
      );
    }

    svg.properties ??= {};
    svg.properties.ariaLabel = normalizeMathAccessibleName(source.source);
    renderedCount += 1;
  });

  if (renderedCount !== sources.length) {
    throw new Error(
      `Captured ${sources.length} TeX sources but MathJax rendered ${renderedCount}`,
    );
  }

  return renderedCount;
}

export function captureMathAccessibilitySources() {
  return (tree: HastNode, file: MathVFile): void => {
    file.data[MATH_SOURCES_KEY] = collectMathSources(tree);
  };
}

export function labelMathJaxSvg() {
  return (tree: HastNode, file: MathVFile): void => {
    const sources = file.data[MATH_SOURCES_KEY];
    if (!Array.isArray(sources)) {
      throw new TypeError(
        "Math accessibility source capture must run before MathJax rendering",
      );
    }
    applyMathAccessibleNames(tree, sources as MathSource[]);
  };
}
