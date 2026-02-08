---
title: Adam算法
tags:
  - 深度学习
mathjax: "true"
date: 2026-02-08 13:00:00
categories:
  - 深度学习
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
### 偏差
我们完成了Momentum和RMSProp
如果我们将两者拼凑在一起
1. 更新一阶矩（动量）

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \mathbf{g}_t
$$

2. 更新二阶矩（自适应项）

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \mathbf{g}_t^2
$$

3. 参数更新

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

其中 $\beta_1$ 通常为 0.9，$\beta_2$ 通常为 0.999
接下来让我们看看当$t=1$时发生了什么
假设我们将$m_0,v_0$初始化为$\mathbf{0}$向量
带入公式

$$
v_1 = \beta_2 \cdot 0 + (1-\beta_2) \mathbf{g}_1^2 = (1 - 0.999) \mathbf{g}_1^2 = 0.001 \cdot \mathbf{g}_1^2
$$

我们观察到，计算出的二阶矩估计值仅为真实梯度平方的0.001倍！
这意味着估计值严重偏向于0
如果直接用这个$v_t$去做分母$\sqrt{v_1}$，分母会极小，导致更新步长会爆炸性地变大（或者在 $m_t$ 上导致步长极小，取决于分子分母谁偏得更多）
这种**初始化偏差**会导致训练初期极其不稳定

### 修正
为了消除这个偏差，我们需要从统计学角度推导一个修正系数

**我们先作一个期望分析**
假设真实梯度的二阶矩是平稳的（Stationary），记为 $E[\mathbf{g}^2]$。我们希望我们的估计量 $v_t$ 是**无偏**的，即希望 $E[v_t] = E[\mathbf{g}^2]$

让我们展开 $v_t$ 的递归式

$$
v_t = (1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \mathbf{g}_i^2
$$

对两边求期望 $E[\cdot]$

$$
\begin{aligned} 
E[v_t] &= E\left[ (1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \mathbf{g}_i^2 \right] \\ &= (1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} E[\mathbf{g}_i^2] \\ &\approx (1-\beta_2) E[\mathbf{g}^2] \sum_{i=1}^t \beta_2^{t-i} 
\end{aligned}
$$

这里 $\sum_{i=1}^t \beta_2^{t-i}$ 是一个等比数列求和，其值为

$$
\sum_{k=0}^{t-1} \beta_2^k = \frac{1 - \beta_2^t}{1 - \beta_2}
$$

代回期望公式

$$
\begin{aligned}
E[v_t] &\approx (1-\beta_2) E[\mathbf{g}^2] \cdot \frac{1 - \beta_2^t}{1 - \beta_2} \\ &= E[\mathbf{g}^2] \cdot (1 - \beta_2^t) 
\end{aligned}$$

**我们发现**，

$$
E[v_t] = \text{真实值} \times (1 - \beta_2^t)
$$

为了得到真实值，我们必须人为的除以系数$1 - \beta_2^t$
- 当 $t$ 很小时，$\beta_2^t = 0.999$，修正因子 $1 - 0.999 = 0.001$。我们把 $v_t$ 除以 $0.001$，正好把它放大了 1000 倍，还原了真实量级
- 当 $t$ 很大时，$\beta_2^t \to 0$，修正因子 $\to 1$。此时不再需要修正，因为 EMA 已经积累了足够的数据，偏差自然消失了
同理，对一阶矩 $m_t$ 也需要除以 $(1 - \beta_1^t)$

结合上述所有推导，我们得到完整的Adam算法流程
**迭代过程 (在时刻 $t$)**：
**计算梯度**：

$$
\mathbf{g}_t = \nabla f(\mathbf{x}_{t-1})
$$

 **更新一阶矩**：
 
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \mathbf{g}_t$$

**更新二阶矩**：

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) \mathbf{g}_t^2$$

**计算偏差修正后的一阶矩**：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

**计算偏差修正后的二阶矩**：

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**计算参数更新**：

$$
\mathbf{x}_t = \mathbf{x}_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$
