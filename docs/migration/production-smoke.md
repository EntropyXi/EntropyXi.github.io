# 生产 Smoke 记录

- 执行命令：`npm run smoke:production`
- 执行时间：2026-08-20（UTC 阶段 6 完成后）
- 目标：`https://entropyxi.github.io`

## 结果

- 81/81 旧站 HTML URL 返回 200，最终 pathname 与 canonical 路径一致。
- 4/4 保留静态资源返回 200：`/atom.xml`、`/search.xml`、`/sitemap.xml`、
  `/images/eigenvalue-error.png`。
- 44 个 NexT/Hexo 主题资源已按
  `docs/migration/legacy-asset-disposition.md` 书面处置，不保留旧 URL。

## 观察项

阶段 6 要求的 7 天观察期从本记录开始。若 7 天内出现历史文章 404、
canonical/pathname 漂移或公式回归，按 `docs/migration/rollback.md` 回滚。
