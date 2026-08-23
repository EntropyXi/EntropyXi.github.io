---
title: "Adam算法"
description: "梳理Adam优化器如何融合动量法与RMSProp的自适应学习率，阐释偏差校正机制及其收敛性质，是当前最广泛使用的深度学习优化器之一。"
date: "2026-02-08T13:00:00+08:00"
updated: "2026-02-08T13:00:00+08:00"
tags:
  - "深度学习"
categories:
  - "深度学习"
permalink: "2026/02/08/深度学习/Adam算法"
math: true
draft: false
---
Adam 优化器是深度学习中最广泛使用的自适应学习率算法，融合了动量法的加速收敛能力与 RMSProp 的逐参数自适应调节，并引入偏差校正机制解决训练早期的冷启动问题。

### 偏差

我们完成了 Momentum 和 RMSProp。

如果我们将两者拼凑在一起：

1. **更新一阶矩（动量）**：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \mathbf{g}_t
$$

2. **更新二阶矩（自适应项）**：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \mathbf{g}_t^2
$$

3. **参数更新**：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

其中 $\beta_1$ 通常取 $0.9$，$\beta_2$ 通常取 $0.999$。

接下来让我们看看当 $t = 1$ 时发生了什么。

假设我们将 $m_0, v_0$ 初始化为 $\mathbf{0}$ 向量，代入公式：

$$
v_1 = \beta_2 \cdot 0 + (1-\beta_2) \mathbf{g}_1^2 = (1 - 0.999) \mathbf{g}_1^2 = 0.001 \cdot \mathbf{g}_1^2
$$

我们观察到，计算出的二阶矩估计值仅为真实梯度平方的 $0.001$ 倍！

这意味着估计值严重偏向于 $0$。如果直接用这个 $v_1$ 去做分母 $\sqrt{v_1}$，分母会极小，导致更新步长爆炸性地变大（或者在 $m_t$ 上导致步长极小，取决于分子分母谁偏得更多）。

这种**初始化偏差**会导致训练初期极其不稳定。

### 修正

为了消除这个偏差，我们需要从统计学角度推导一个修正系数。

**我们先作一个期望分析**：

假设真实梯度的二阶矩是平稳的（Stationary），记为 $\mathbb{E}[\mathbf{g}^2]$。我们希望我们的估计量 $v_t$ 是**无偏**的，即希望 $\mathbb{E}[v_t] = \mathbb{E}[\mathbf{g}^2]$。

让我们展开 $v_t$ 的递归式：

$$
v_t = (1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \mathbf{g}_i^2
$$

对两边求期望 $\mathbb{E}[\cdot]$：

$$
\begin{aligned} 
\mathbb{E}[v_t] &= \mathbb{E}\left[ (1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \mathbf{g}_i^2 \right] \\
&= (1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \mathbb{E}[\mathbf{g}_i^2] \\
&\approx (1-\beta_2) \mathbb{E}[\mathbf{g}^2] \sum_{i=1}^t \beta_2^{t-i} 
\end{aligned}
$$

这里 $\sum_{i=1}^t \beta_2^{t-i}$ 是一个等比数列求和，其值为：

$$
\sum_{k=0}^{t-1} \beta_2^k = \frac{1 - \beta_2^t}{1 - \beta_2}
$$

代回期望公式：

$$
\begin{aligned}
\mathbb{E}[v_t] &\approx (1-\beta_2) \mathbb{E}[\mathbf{g}^2] \cdot \frac{1 - \beta_2^t}{1 - \beta_2} \\
&= \mathbb{E}[\mathbf{g}^2] \cdot (1 - \beta_2^t) 
\end{aligned}
$$

**我们发现**：

$$
\mathbb{E}[v_t] = \mathbb{E}[\mathbf{g}^2] \cdot (1 - \beta_2^t)
$$

为了得到真实值，我们必须人为地除以系数 $1 - \beta_2^t$：
- 当 $t = 1$ 时，$\beta_2^1 = 0.999$，修正因子 $1 - 0.999 = 0.001$。我们把 $v_t$ 除以 $0.001$，正好把它放大了 1000 倍，还原了真实量级；
- 当 $t$ 很大时，$\beta_2^t \to 0$，修正因子 $\to 1$。此时不再需要额外放大，因为 EMA 已经积累了足够的数据，初始偏差自然消失了。

同理，对一阶矩 $m_t$ 也需要除以 $(1 - \beta_1^t)$。

结合上述所有推导，我们得到完整的 Adam 算法流程。

**迭代过程（在时刻 $t$）**：

**计算梯度**：

$$
\mathbf{g}_t = \nabla f(\mathbf{x}_{t-1})
$$

**更新一阶矩**：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \mathbf{g}_t
$$

**更新二阶矩**：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \mathbf{g}_t^2
$$

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
