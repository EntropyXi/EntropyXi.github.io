# 旧站行为冻结记录

本文件记录 Hexo/NexT 旧站迁移前必须冻结的站点行为。数据来源为
`D:\Blog_file` 工作树当前配置；Astro 迁移验收必须与此记录对齐。

## Giscus

- 注入位置：`source/_data/post-body-end.njk`。
- 挂载位置：`_config.next.yml` 的 `custom_file_path.postBodyEnd`（第 245 行）。
- 渲染条件：仅 `page.comments !== false` 的文章页输出评论容器。
- 脚本：`https://giscus.app/client.js`，带 `data-pjax`、`crossorigin="anonymous"`、`async`。

| 参数                     | 值                              |
| ------------------------ | ------------------------------- |
| `data-repo`              | `EntropyXi/EntropyXi.github.io` |
| `data-repo-id`           | `R_kgDORLLL8g`                  |
| `data-category`          | `General`                       |
| `data-category-id`       | `DIC_kwDORLLL8s4C9O73`          |
| `data-mapping`           | `pathname`                      |
| `data-strict`            | `0`                             |
| `data-reactions-enabled` | `1`                             |
| `data-emit-metadata`     | `0`                             |
| `data-input-position`    | `top`                           |
| `data-theme`             | `preferred_color_scheme`        |
| `data-lang`              | `zh-CN`                         |

## 搜索

- 生成器：`hexo-generator-searchdb`（`package.json`，`^1.5.0`）。
- 输出 URL：`/search.xml`。

`_config.yml`：

```yaml
search:
  path: search.xml
  field: post
  content: true
  format: html
```

`_config.next.yml`：

```yaml
local_search:
  enable: true
  top_n_per_article: 1
  unescape: false
  preload: false
```

## RSS / Atom

- 生成器：`hexo-generator-feed`（`package.json`，`^4.0.0`）。
- 输出 URL：`/atom.xml`。

`_config.yml`：

```yaml
feed:
  type: atom
  path: atom.xml
  limit: 20
  content: false
```

## Sitemap

- 生成器：`hexo-generator-sitemap`（`package.json`，`^3.0.1`）。
- 输出 URL：`/sitemap.xml`。

`_config.yml`：

```yaml
sitemap:
  path: sitemap.xml
```

## MathJax（额外冻结）

- 加载策略：NexT 在 `mathjax: true` 页面按需加载；`_config.next.yml` 第 152-158 行。

```yaml
math:
  every_page: false
  mathjax:
    enable: true
    tags: none
```

- Pandoc 使用 `--mathjax` 参数保留 TeX，浏览器侧由 MathJax 3 渲染。

## 部署

- 当前生产构建：`.github/workflows/deploy.yml`，名称 `Deploy to GitHub Pages`。
- 触发：`source` 分支 push 与 `workflow_dispatch`。
- 权限：`contents: read`、`pages: write`、`id-token: write`。
- 并发：`concurrency.group = "pages"`，`cancel-in-progress: true`。
- build job：
  - `actions/checkout@v4`，`fetch-depth: 0`
  - `actions/setup-node@v4`，Node `20`，`cache: npm`
  - 安装 Pandoc
  - `npm ci`
  - `npm run check`
  - `touch public/.nojekyll`
  - `actions/upload-pages-artifact@v3`，`path: ./public`
- deploy job：
  - 依赖 build
  - `actions/deploy-pages@v4`

`_config.yml` 中的 Hexo `deploy` 配置仅作为旧 `hexo deploy` 路径记录，实际生产由
GitHub Actions 部署。
