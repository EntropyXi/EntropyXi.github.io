# Agent 运行记录

## 主 Agent

- 唯一调度、核心逻辑实现、测试、浏览器验收和最终放行者；
- 所有项目命令从 Git Bash 发起；
- 当前分支：`codex/ayeez-ui-motion-refactor`。

## DeepSeek Harness

- 工作区：`Blog_file`；
- 模型：DeepSeek‑V4‑Flash；
- 思考程度：Max；
- 双独立只读高精度审查已完成；
- 第一轮一次非 Git Bash 只读调用已被主 Agent制止且不计入证据；
- 后续审查会话禁止使用无法指定为 Git Bash 的 Shell 工具。

## Antigravity CLI

- 版本：1.1.17；
- 工作区：`D:/Blog_file`；
- 账户：Google AI Pro 已登录；
- 模型：Gemini 3.7 Flash；
- 思考程度：High；
- 会话仍可复用；
- 所有主要前端可视界面必须由其实现，主 Agent只提供契约、文件租约和验收条件。

## 写入租约

- 阶段 0/0.5：主 Agent拥有文档、测试、构建配置与数学核心文件租约；Antigravity 不写文件；
- 阶段 1：只授予 Antigravity `docs/refactor/design-specs.md` 视觉规格租约；
- 阶段 2：主 Agent先完成脚本解耦、测试与动效契约；
- 阶段 3 起：按阶段把明确的 Astro/CSS/SVG 可视文件串行租给 Antigravity；
- 任何 Agent 完成后必须停止写入，主 Agent审查通过才释放到下一阶段。

## 阶段 0.5 放行记录

- 数学可访问性插件、构建配置、ADR、单测、E2E 与产物审计均由主 Agent实现；
- Antigravity 未写入阶段 0.5 文件；
- 全量门禁、1289/1289 公式产物审计、Chrome DOM 抽查与 Lighthouse 数学 SVG
  审计通过后，数学专项允许进入独立 commit；
- permalink、分页、聚合路由、Astro SSG、Pagefind 与 GitHub Pages 链路保持现状。
