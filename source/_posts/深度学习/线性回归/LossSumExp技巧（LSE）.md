---
title: LossSumExp技巧（LSE）
date: 2026-02-08 13:00:00
mathjax: true
tags:
  - 深度学习，线性回归
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

在手动实现的softmax和交叉熵损失中，我们在这两步是分步计算的，即我们在计算评估函数Xw+b后，我们先调用softmax计算$\frac{exp(x_i)}{\sum exp(x_j)}$，然后再把计算出来的softmax概率直接传给 log 计算交叉熵。但是这种分步计算数值是及其不稳定的，因为我们的softmax计算出来的数值再0到1的区间内，而当数值趋近于0时，很容易遇到数值下溢出导致log(0)报错。这在数值上是十分脆弱的。

所以我们使用LSE技巧，在损失函数内部，将这两部合并计算
- 核心逻辑：框架（如 PyTorch 的 nn.CrossEntropyLoss ）要求你传入未规范化的预测
- 内部黑盒：在损失函数内部，它并不会先算概率再算对数，而是利用数学恒等式将这两步合并计算
- 数学原理：它计算的是$\log(\sum exp(x_i))$，并使用了平移技巧（减去最大值c）

$$
LogSumExp(x)=c+\log\sum exp(x_i-c),\quad \text{其中c=max(x)}
$$

- 优势：这种合并计算的方式在计算机底层是极其稳定的，永远不会出现NaN（不是数字）或无穷大的错误

接下来我们证明等价性：
其中softmax公式：$\tilde{y}_i=\frac{exp(x_i)}{\sum_j exp(x_j)}$
交叉熵损失（针对单样本）：$L=-\log(\tilde{y}_y)$

将softmax的定义带入到损失函数L中

$$
L=-\log(\tilde{y}_i=\frac{exp(x_i)}{\sum_j exp(x_j)})
$$

利用对数的运算法则展开

$$
L=-[\log(exp(x_i))-log(\sum_{j}exp(x_j))]
$$

因为$\log(exp(x))=x$，所以

$$
L=-[x_i-log(\sum_{j}exp(x_j))]
$$

展开

$$
L=-x_i+\log(\sum_{j}exp(x_j))
$$

得证
