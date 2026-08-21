# 阶段 0.5：数学与生产产物专项实施记录

## 决策

- 保留 Astro SSG、`remark-math` 与 `rehype-mathjax` 的构建期数学链路；
- 不改写 30 篇文章正文，不改变 permalink、canonical、feed 或搜索契约；
- 为每个 MathJax SVG 增加由同序原 TeX 生成的可访问名称；
- 移除仅用于 Hexo → Astro 迁移验收的 `/dev/math-spike/`、客户端 MathJax
  组件、vendor 复制脚本与直接依赖。

## 实现与失败策略

`src/lib/markdown/math-accessibility.ts` 在 `rehype-mathjax` 前捕获公式源，
在其后按 DOM 顺序标注 SVG。标签以“数学公式：”开头，仅折叠空白，不改写
TeX 命令。捕获数量、渲染数量或 SVG 输出不一致会直接使构建失败。

产物审计同时禁止：

- 可访问名称以外出现原始 `$$` 或 `\begin`；
- `mathjax-error`；
- 缺少可访问名称的公式 SVG；
- `/dev/math-spike/` 或 `vendor/mathjax` 重新进入 `dist/`。

## 验证结果

- `npm run check`：通过；
- Astro Check：55 个文件，0 error / 0 warning / 0 hint；
- 内容：30/30；
- Vitest：3 个文件、8 个测试；
- Playwright：9/9；
- 真实复杂公式样本：5 篇 × 桌面/390px 移动视口；
- 产物：1289/1289 个 `mjx-container` SVG 具有可访问名称；
- Chrome 人工 DOM 复核：DDIM 页 172/172 个公式具名，无页面级横向溢出；
- Lighthouse `svg-img-alt`：由失败变为通过；剩余 98 分来自历史正文 heading order，
  与数学 SVG 无关。

专项后的 `dist/` 约 9.6 MiB、136 个文件；JS 单文件 gzip 合计 108,388 bytes。
基线约 33 MiB、243 个文件、JS gzip 5,508,153 bytes。主要差值来自移除客户端
MathJax vendor；普通文章仍无需数学运行时脚本。

Lighthouse 原始结果：
`docs/refactor/lighthouse/article-ddim-math-a11y.json`。Windows 在报告写入后清理
Lighthouse 临时目录时仍可能报 `EPERM`，报告本身无 runtime error。

## 回滚

本专项独立于任何视觉文件。回滚时恢复专项 commit 即可重新得到迁移实验页、
客户端 vendor 复制流程与旧数学配置；30 篇文章内容及公开生产 URL 均无需回滚。
被移出的生成 vendor 还保留在本机临时备份
`/tmp/ayeez-mathjax-backup.gYucRf/mathjax`，同时可通过锁文件恢复依赖后重新生成。
