# EntropyXi 的个人博客

基于 Hexo + NexT 主题的深度学习与数值分析技术笔记。

**线上地址**：https://entropyxi.github.io

## 技术栈

| 组件     | 说明                                 |
| -------- | ------------------------------------ |
| 静态生成 | Hexo 8 + Pandoc 渲染 Markdown        |
| 主题     | NexT 8 Pisces（双栏布局）            |
| 公式渲染 | MathJax 3 客户端                     |
| 评论     | Giscus（GitHub Discussions）         |
| 搜索     | hexo-generator-searchdb 本地搜索     |
| 访问统计 | 暂停展示（生产统计数据核实后再启用） |
| CI/CD    | GitHub Actions → GitHub Pages        |

## 写一篇新笔记

```bash
# 在对应分类目录下创建 .md 文件
vim source/_posts/深度学习/流匹配与扩散模型/新文章.md
```

**Frontmatter 模板**：

```yaml
---
title: 文章标题
date: 2026-05-17 00:00:00
tags:
  - 深度学习
  - 流匹配与扩散模型
mathjax: true
categories:
  - 深度学习
  - 流匹配与扩散模型
---
<style>
.mjx-container, .MathJax_Display, .MathJax {
    overflow-x: auto !important;
    overflow-y: hidden;
    max-width: 100%;
    -webkit-overflow-scrolling: touch;
}
</style>
<!-- more -->

### 第一节
内容...
```

**格式规范**：

- 行内公式用 `$...$`，块级公式用 `$$...$$`
- `$$` 块前后各空一行
- 别用 Unicode `·`（中间点），用 `\cdot`
- 文章第一个段落前加 `<!-- more -->` 控制首页摘要

**Note 提示框**：

```markdown
{% note info %}
**定理**：这里写定理内容。
{% endnote %}

{% note warning %}
**注意**：这里写注意事项。
{% endnote %}
```

## 本地预览

```bash
npx hexo server
# 打开 http://localhost:4000
```

## 部署

```bash
git add source/_posts/...
git commit -m "new post: xxx"
git push origin source
```

推送 `source` 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。

## 本地验证

```bash
npm ci
npm run check
npm run server
```

`npm run check` 验证 frontmatter、生成站点并检查生成的 HTML 是否存在已知回归问题。

## 项目结构

```
Blog_file/
├── source/
│   ├── _posts/              # 笔记源文件 (.md)
│   │   ├── 深度学习/        # 分类：深度学习
│   │   │   ├── 流匹配与扩散模型/  # 子分类
│   │   │   │   ├── 体系/    # 理论体系笔记
│   │   │   │   └── 超分辨率/ # 超分辨率论文笔记
│   │   │   └── 线性回归/
│   │   └── 数值分析/
│   ├── _data/               # 自定义样式与脚本
│   │   ├── styles.styl      # 全局 CSS
│   │   └── post-body-end.njk # Giscus 评论注入
│   └── about/               # 关于页面
├── _config.yml              # Hexo 主配置
├── _config.next.yml         # NexT 主题配置
├── .github/workflows/       # GitHub Actions 部署
└── themes/                  # 通过 npm 管理的主题（NexT）
```

NexT 主题通过 npm 安装，仅通过 `_config.next.yml` 和 `source/_data/` 进行自定义。

## 优化清单

- [x] NexT Pisces 双栏 + 暗色模式
- [x] MathJax 公式渲染（暗色模式适配）
- [x] Giscus GitHub 评论
- [x] 本地搜索
- [x] Note 提示框（定理/注意/引理）
- [x] 不蒜子页面访问统计
- [x] RSS 订阅 `/atom.xml`
- [x] Sitemap `/sitemap.xml`
- [x] FancyBox 图片灯箱
- [x] 阅读进度条 + 回顶滚动百分比
- [x] 手机公式缩小适配
- [x] 系统 CJK 字体栈（免 Google Fonts）
