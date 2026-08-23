---
title: "LossSumExp技巧（LSE）"
description: "解释Log-Sum-Exp数值稳定技巧的数学原理，分析Softmax计算中指数溢出问题如何通过减去最大值得到缓解，是深度学习中的经典数值技巧。"
date: "2026-02-08T13:00:00+08:00"
updated: "2026-02-08T13:00:00+08:00"
tags:
  - "深度学习"
  - "线性回归"
categories:
  - "深度学习"
  - "线性回归"
permalink: "2026/02/08/深度学习/线性回归/LossSumExp技巧（LSE）"
math: true
draft: false
---

在手动实现 Softmax 与交叉熵损失（Cross Entropy Loss）时，若分步计算：先在模型输出得分 $\mathbf{z} = \mathbf{X}\mathbf{w} + b$ 上计算 Softmax 概率 $\hat{y}_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$，再将预测概率传入 $\log$ 函数计算交叉熵损失 $L = -\log \hat{y}_y$。这种分步计算在数值上极其不稳定：当得分 $z_i$ 很大时 $\exp(z_i)$ 容易发生**数值上溢**（Overflow，导致 `inf`）；而当概率 $\hat{y}_y$ 极小时又容易发生**数值下溢**（Underflow，导致 $\log(0) \to -\infty$ 或 `NaN`）。

为了解决这一问题，深度学习框架（如 PyTorch 的 `nn.CrossEntropyLoss`）在损失函数内部将 Softmax 与对数运算合并为 **Log-Sum-Exp（LSE）** 技巧进行统一计算。

- **核心逻辑**：模型直接输出未归一化的 logits（对数几率 $\mathbf{z}$），损失函数内部将 Softmax 与 Cross-Entropy 融合成单一算子；
- **数值稳定原理**：在指数求和前减去最大值 $c = \max_i z_i$，保证所有指数项的幂次不超过 0（$\exp(z_i - c) \leq 1$），彻底杜绝溢出：

$$
\operatorname{LogSumExp}(\mathbf{z}) = c + \log\left(\sum_i \exp(z_i - c)\right), \quad \text{其中 } c = \max_i z_i
$$

### 数学等价性证明

对于真实类别标签为 $y$ 的单样本，交叉熵损失定义为：

$$
L = -\log \hat{y}_y
$$

将 Softmax 概率公式 $\hat{y}_y = \frac{\exp(z_y)}{\sum_j \exp(z_j)}$ 代入损失函数 $L$：

$$
L = -\log\left(\frac{\exp(z_y)}{\sum_j \exp(z_j)}\right)
$$

利用对数的除法性质展开：

$$
\begin{aligned}
L &= -\left[ \log(\exp(z_y)) - \log\left(\sum_j \exp(z_j)\right) \right] \\
&= -z_y + \log\left(\sum_j \exp(z_j)\right)
\end{aligned}
$$

对第二项应用减最大值平移恒等式：

$$
\begin{aligned}
\log\left(\sum_j \exp(z_j)\right) &= \log\left(\sum_j \exp(z_j - c) \cdot \exp(c)\right) \\
&= \log\left(\exp(c) \cdot \sum_j \exp(z_j - c)\right) \\
&= c + \log\left(\sum_j \exp(z_j - c)\right)
\end{aligned}
$$

代入最终损失函数，得到数值极度稳定的解析表达式：

$$
L = -z_y + c + \log\left(\sum_j \exp(z_j - c)\right)
$$

这样，所有中间项的最大指数均为 0（$\exp(0) = 1$），既避免了 $\exp$ 溢出，又避免了 $\log(0)$ 产生 `NaN`，在计算机底层保证了极高的数值鲁棒性。
