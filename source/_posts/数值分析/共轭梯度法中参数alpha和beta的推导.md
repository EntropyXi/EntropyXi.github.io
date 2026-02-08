---
tags:
  - 数值分析
title: 共轭梯度法中参数alpha和beta的推导
mathjax: "true"
date: 2026-02-08 13:00:00
---
<style>
/* 强制让 MathJax 公式容器支持横向滚动 */
.mjx-container, .MathJax_Display, .MathJax {
    overflow-x: auto !important; /* 超出宽度时显示滚动条 */
    overflow-y: hidden;          /* 隐藏垂直滚动条 */
    max-width: 100%;             /* 限制最大宽度为屏幕宽度 */
    -webkit-overflow-scrolling: touch; /* 优化移动端滑动体验 */
}
</style>
我们在这里推导$\mathbf{p}_k = \mathbf{r}_k + \beta_{k-1} \mathbf{p}_{k-1}$中的参数$\alpha$和$\beta$

### $\alpha_k$ 的推导

我们现在的目标是：在已知当前位置 $\mathbf{x}_k$ 和搜索方向 $\mathbf{p}_k$ 的情况下，确定我们要走多远
这意味着我们要选择一个 $\alpha_k$，使得函数 $\varphi$ 在这条直线上达到**最小值**。
我们将 $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha \mathbf{p}_k$ 代入 $\varphi(\mathbf{u})$ 的表达式中，把 $\varphi$ 看作只关于 $\alpha$ 的函数 $f(\alpha)$：

$$
f(\alpha) = \varphi(\mathbf{x}_k + \alpha \mathbf{p}_k)
$$

展开 $\varphi(\mathbf{u}) = \mathbf{u}^T \mathbf{A} \mathbf{u} - 2\mathbf{u}^T \mathbf{b}$

$$
f(\alpha) = (\mathbf{x}_k + \alpha \mathbf{p}_k)^T \mathbf{A} (\mathbf{x}_k + \alpha \mathbf{p}_k) - 2(\mathbf{x}_k + \alpha \mathbf{p}_k)^T \mathbf{b}
$$

展开括号（利用 $\mathbf{A}$ 的对称性，$\mathbf{x}_k^T \mathbf{A} \mathbf{p}_k = \mathbf{p}_k^T \mathbf{A} \mathbf{x}_k$）

$$
f(\alpha) = \underbrace{\mathbf{x}_k^T \mathbf{A} \mathbf{x}_k - 2\mathbf{x}_k^T \mathbf{b}}_{\text{常数项}} + 2\alpha \mathbf{p}_k^T \mathbf{A} \mathbf{x}_k + \alpha^2 \mathbf{p}_k^T \mathbf{A} \mathbf{p}_k - 2\alpha \mathbf{p}_k^T \mathbf{b}
$$

为了求最小值，我们对 $\alpha$ 求导，并令导数为 $0$

$$
\frac{d f(\alpha)}{d \alpha} = 2 \mathbf{p}_k^T \mathbf{A} \mathbf{x}_k + 2\alpha \mathbf{p}_k^T \mathbf{A} \mathbf{p}_k - 2 \mathbf{p}_k^T \mathbf{b} = 0
$$

整理

$$
\alpha (\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k) + \mathbf{p}_k^T (\mathbf{A} \mathbf{x}_k - \mathbf{b}) = 0
$$

因为 $\mathbf{A} \mathbf{x}_k - \mathbf{b} = -\mathbf{r}_k$，所以

$$
\alpha (\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k) - \mathbf{p}_k^T \mathbf{r}_k = 0
$$

最后解出 $\alpha$

$$
\alpha_k = \frac{\mathbf{p}_k^T \mathbf{r}_k}{\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k}
$$

你会发现上面的公式分子是 $\mathbf{p}_k^T \mathbf{r}_k$，但在最终算法里我们写的是 $\mathbf{r}_k^T \mathbf{r}_k$。为什么它们会相等？
回顾 $\mathbf{p}_k$ 的定义：$\mathbf{p}_k = \mathbf{r}_k + \beta_{k-1} \mathbf{p}_{k-1}$。
两边同时左乘 $\mathbf{r}_k^T$

$$
\mathbf{r}_k^T \mathbf{p}_k = \mathbf{r}_k^T \mathbf{r}_k + \beta_{k-1} \underbrace{\mathbf{r}_k^T \mathbf{p}_{k-1}}_{0}
$$

在共轭梯度法中，有一个重要的性质：**当前的残差 $\mathbf{r}_k$ 与旧的搜索方向 $\mathbf{p}_{k-1}$ 正交**（即 $\mathbf{r}_k^T \mathbf{p}_{k-1} = 0$）
因此，分子可以简化为

$$
\mathbf{p}_k^T \mathbf{r}_k = \mathbf{r}_k^T \mathbf{r}_k
$$

最终得到

$$
\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k}
$$




### $\beta_k$ 的推导
我们在算法中定义了搜索方向的更新公式：

$$
\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k
$$

我们的目标是找到一个合适的 $\beta_k$，使得新的方向 $\mathbf{p}_{k+1}$ 与旧的方向 $\mathbf{p}_k$ 关于矩阵 $\mathbf{A}$ **共轭**
#### 列出共轭条件
根据共轭梯度的定义，我们需要

$$
\mathbf{p}_{k+1}^T \mathbf{A} \mathbf{p}_k = 0
$$

将 $\mathbf{p}_{k+1}$ 的定义代入上述条件中

$$
(\mathbf{r}_{k+1} + \beta_k \mathbf{p}_k)^T \mathbf{A} \mathbf{p}_k = 0
$$

展开括号

$$
\mathbf{r}_{k+1}^T \mathbf{A} \mathbf{p}_k + \beta_k \mathbf{p}_k^T \mathbf{A} \mathbf{p}_k = 0
$$

移项并求解 $\beta_k$

$$
\beta_k = - \frac{\mathbf{r}_{k+1}^T \mathbf{A} \mathbf{p}_k}{\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k}
$$

这个公式虽然正确，但计算起来很麻烦（需要做矩阵乘法 $\mathbf{A}\mathbf{p}_k$）。我们可以利用 $\alpha_k$ 的更新公式来化简它。
回顾残差的更新公式：$\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{A} \mathbf{p}_k$。
我们可以反解出 $\mathbf{A} \mathbf{p}_k$：

$$
\mathbf{A} \mathbf{p}_k = -\frac{1}{\alpha_k} (\mathbf{r}_{k+1} - \mathbf{r}_k)
$$

将这个式子代入 $\beta_k$ 分子的 $\mathbf{A} \mathbf{p}_k$ 中：

$$
\text{分子} = \mathbf{r}_{k+1}^T \mathbf{A} \mathbf{p}_k = \mathbf{r}_{k+1}^T \left[ -\frac{1}{\alpha_k} (\mathbf{r}_{k+1} - \mathbf{r}_k) \right]
$$

$$
= -\frac{1}{\alpha_k} (\mathbf{r}_{k+1}^T \mathbf{r}_{k+1} - \underbrace{\mathbf{r}_{k+1}^T \mathbf{r}_k}_{0})
$$

又因为新残差与旧残差正交 ($\mathbf{r}_{k+1}^T \mathbf{r}_k = 0$)，所以分子简化为

$$
-\frac{1}{\alpha_k} \mathbf{r}_{k+1}^T \mathbf{r}_{k+1}
$$现在看分母，回顾 $\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k}$，所以分母其实等于：

$$
\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\alpha_k}
$$
化简后分子和分母相除

$$
\beta_k = - \frac{-\frac{1}{\alpha_k} \mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\frac{1}{\alpha_k} \mathbf{r}_k^T \mathbf{r}_k}
$$

消去 $-\frac{1}{\alpha_k}$ 和 $\frac{1}{\alpha_k}$，最终就得到

$$
\beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k}
$$