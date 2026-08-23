---
title: "RMSProp"
description: "介绍RMSProp算法如何通过指数加权移动平均对历史梯度平方做归一化，解决AdaGrad学习率单调递减的问题，使训练更加稳定。"
date: "2026-02-08T13:00:00+08:00"
updated: "2026-02-08T13:00:00+08:00"
tags:
  - "深度学习"
categories:
  - "深度学习"
permalink: "2026/02/08/深度学习/RMSProp"
math: true
draft: false
---

RMSProp 是一种**自适应学习率**方法。它通过计算梯度**二阶矩**的指数加权移动平均，来对每个参数的学习率进行独立缩放：

$$
\begin{aligned}
v_t &= \beta v_{t-1} + (1-\beta) (\mathbf{g}_t \odot \mathbf{g}_t) \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot \mathbf{g}_t
\end{aligned}
$$

其中：
- $\odot$ 表示哈达玛积（Hadamard product），即逐元素相乘；
- $v_t = \mathbb{E}[\mathbf{g}^2]_t$ 是梯度平方的滑动累积估计量；
- $\beta$ 是**衰减率**（典型值取 $0.9$ 或 $0.99$）；
- $\epsilon$ 是防止分母为零的**平滑项**。

### 构造与推导

我们从 RMSProp 的根源——**AdaGrad** 开始讲起。

动量法缓解了梯度的震荡问题，但所有参数依然共用同一个全局学习率 $\eta$。在处理稀疏特征或尺度差异巨大的参数（例如深度神经网络）时，我们希望：
- 对于频繁更新或梯度很大的参数，降低其学习率，防止其剧烈震荡发散；
- 对于稀疏或梯度较小的参数，增大其学习率，加快其收敛速度。

为了实现上述目标，我们需要构造一个调节因子，该调节因子应当与历史梯度的幅度成反比。

那么，我们该如何衡量一个参数在过去一段时间内的梯度**大小**？

简单的累加梯度 $\sum g_t$ 是不行的，因为正负方向的梯度会相互抵消。因此，我们必须累加梯度的**平方**。

定义 $r_t$ 为直到 $t$ 时刻所有历史梯度的平方和：

$$
r_t = \sum_{\tau=1}^{t} g_\tau^2
$$

现在，我们使用 $\sqrt{r_t}$ 作为分母对学习率进行归一化，这就是 AdaGrad 的核心公式：

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{r_t} + \epsilon} \odot g_t
$$

其中 $\epsilon$ 是一个防止分母为 0 的极小常数。

#### AdaGrad 的致命缺陷

虽然 AdaGrad 实现了各向异性的自适应步长，但在深度神经网络的长时间训练中，它暴露出一个致命缺陷。

观察 $r_t$ 的递归定义：

$$
r_t = \sum_{\tau=1}^{t} g_\tau^2 = r_{t-1} + g_t^2
$$

由于 $g_t^2 \geq 0$，累积项 $r_t$ 是单调递增的。随着训练步数 $t \to \infty$：
- $r_t \to \infty$；
- 有效学习率 $\frac{\eta}{\sqrt{r_t} + \epsilon} \to 0$。

如果在找到最优点之前，学习率就已经衰减到接近 0，参数更新就会过早停滞，无法收敛到最优解。

#### 引入 RMSProp 修正

为了解决学习率过早衰减的问题，我们需要一种既能衡量近期梯度大小、又不会无限累加历史梯度的机制。

我们可以通过**指数加权移动平均**（Exponential Moving Average, EMA）来实现：引入衰减系数 $\beta$，对历史梯度平方赋予衰减权重。

### RMSProp 的构造步骤

我们将 AdaGrad 的累加公式：

$$
r_t = r_{t-1} + g_t^2
$$

修改为 RMSProp 的滑动加权更新公式：

$$
v_t = \beta \cdot \underbrace{v_{t-1}}_{\text{历史信息}} + (1-\beta) \cdot \underbrace{g_t^2}_{\text{当前信息}}
$$

**步骤解析**：
1. 每一步中，旧的累积量 $v_{t-1}$ 都会乘以 $\beta$（例如 $0.9$）。这意味着很久以前的梯度信息会以指数速度衰减，不再主导分母；
2. 当前的梯度平方 $g_t^2$ 被赋予权重 $(1-\beta)$ 注入累积量；
3. 这种方式使得 $v_t$ 不再单调递增，而是成为局部时间窗口内梯度二阶矩的**无偏滑动估计量**。

结合上述推导，我们得到了 RMSProp 的完整参数更新流程。

对于参数 $\mathbf{w}$ 和当前梯度 $\mathbf{g}_t$：
- **更新二阶矩估计**：

$$
v_t = \beta v_{t-1} + (1-\beta) \mathbf{g}_t^2
$$

- **执行参数更新**：

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot \mathbf{g}_t
$$
