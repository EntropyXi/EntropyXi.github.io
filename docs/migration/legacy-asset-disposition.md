# 旧站静态资源处置记录

旧站 manifest 共有 48 个静态资源。Astro 站点保留了以下 4 个：

- `/atom.xml`
- `/search.xml`
- `/sitemap.xml`
- `/images/eigenvalue-error.png`

其余 44 个资源属于 NexT 主题与 Hexo 运行时脚本：

- `/css/main.css`、`/css/noscript.css`
- `/images/*`（NexT 主题图标与头像）
- `/js/**`（NexT 运行时脚本与第三方兼容脚本）

处置：随阶段 7 删除 NexT/Hexo 实现一并移除，不保留旧 URL。它们不是文章
内容或读者收藏链接，属于主题实现细节。若未来发现外部链接依赖其中某个
URL，可单独补一个静态文件或 301 页面。
