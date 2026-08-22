# Astro 切换与回滚手册

## 切换后状态

- 生产由 `.github/workflows/deploy.yml` 构建 Astro `dist/` 并部署 GitHub Pages。
- 触发分支：`source`。
- Node 版本来自 `.nvmrc`（22）。
- 回滚 tag：`pre-astro-migration` 指向 Hexo 最后基线
  `0780179a4e3054eb532dc5868c6929306fb0cfe5`。

## 回滚步骤

1. 切回 Hexo 最后 commit：

   ```bash
   git checkout pre-astro-migration
   git push origin pre-astro-migration:source --force
   ```

2. 恢复 `.github/workflows/deploy.yml` 的 Hexo 版本（使用
   `legacy:check` 并上传 `public/`）。

3. 确认 GitHub Pages Source 仍为 GitHub Actions（仓库 Settings → Pages）。

4. 用旧站 manifest 抽查首页与一篇数学文章。

## 验证命令

切换前本地全量验证：

```bash
npm ci
npm run check
```

`npm run check` 包含格式、Lint、类型、内容审计、单测、E2E（含构建
Pagefind 索引）与输出审计。

## Phase 8 重构回滚

如果需要回滚 Phase 8，切换到以下 tag/commit：

- Commit Hash: b35ec8d5d4cec033efe97bb87c37b8a96875623c
