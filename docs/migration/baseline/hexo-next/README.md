# Hexo / NexT 视觉基线

这些截图冻结迁移前的页面信息结构、文章可读性和响应式表现。Astro 的第一阶段只需保证静态内容完整、公式正确、布局不溢出；不复刻 NexT 的入场动画、PJAX 或像素细节。

## 截图清单

| 文件 | 视口 / 页面 | 核验重点 |
| --- | --- | --- |
| `home-desktop.png` | 桌面首页 | 导航、文章列表、侧栏信息 |
| `archives-desktop.png` | 桌面归档 | 时间线与文章入口 |
| `categories-desktop.png` | 桌面分类 | 分类入口与数量信息 |
| `tags-desktop.png` | 桌面标签 | 标签入口与数量信息 |
| `about-desktop.png` | 桌面关于页 | 独立页面内容 |
| `home-mobile.png` | 移动首页 | 单列阅读、菜单与横向溢出 |
| `math-sde-mobile.png` | 移动数学文章 | 公式与正文在窄屏下的可读性 |
| `math-ddpm-desktop-full.png` | 桌面长文章 | DDPM 全文与公式 |
| `math-sde-desktop-full.png` | 桌面长文章 | SDE 全文与公式 |
| `math-sr3-desktop-full.png` | 桌面长文章 | SR3 全文与公式 |
| `math-ddim-desktop-full.png` | 桌面长文章 | DDIM 全文与公式 |
| `math-resshift-desktop-full.png` | 桌面长文章 | ResShift 全文与公式 |

## 使用规则

- 迁移后优先比较页面是否存在、正文是否完整、标题层级是否正确、公式是否渲染以及桌面/移动端是否无布局溢出。
- 截图中的 NexT 动画中间态、主题装饰、精确字号和间距不构成兼容要求。
- 后续 Ayeez 风格重构必须保留同一套内容与路由验收，不得以视觉改版为由删除文章能力。
- 新截图应写入新的目录或使用明确的新文件名，不覆盖本基线。
