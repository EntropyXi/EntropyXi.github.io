---
title: "UPSR"
description: "分析UPSR如何利用预训练扩散模型的先验知识实现盲超分辨率，讨论其潜在空间策略与退化估计器的协同设计思路。"
date: "2026-05-17T14:00:00+08:00"
updated: "2026-05-17T14:00:00+08:00"
tags:
  - "深度学习"
  - "流匹配与扩散模型"
categories:
  - "深度学习"
  - "流匹配与扩散模型"
  - "超分辨率"
permalink: "2026/05/17/深度学习/流匹配与扩散模型/超分辨率/UPSR"
math: true
draft: false
---

### Motivation

在传统基于扩散的超分辨率中，低分辨率（LR）图像中的空间先验信息往往被忽略，即：**平坦区域已经接近目标高分辨率（HR）图像，而边缘和纹理高频区域则距离真实流形较远**。以往的方法对整幅图像所有像素施加各向同性的均匀噪声，导致在已经清晰的平坦区域浪费大量去噪采样步骤。

因此，UPSR 提出了一种**各向异性内容自适应扩散过程**：平坦区域分配较低的噪声水平，而边缘和纹理区域则注入较大的噪声。为了实现区域自适应的噪声方差调制，UPSR 引入了基于残差的不确定性估计指标。

### Methodology

随着每个像素对残差 $|y-x|$ 的增加，高频纹理区域的感知质量对扩散噪声的强度极其敏感。

符号定义：HR 图像 $x^i$，对应的 LR 图像 $y^i$，辅助网络 SR 估计 $g(y^i)$，不确定性图 $\psi^i$。

论文将最终的超分图像建模为两部分：确定性 SR 估计项与不确定性扰动项：

$$
x^i = g(y^i) + \epsilon \psi(g(y^i))
$$

其中 $\epsilon$ 服从高斯分布或拉普拉斯分布。

类似地，可以将 $y^i$ 的不确定性与残差 $|x^i - y^i|$ 联系起来：

$$
x^i = y^i + \hat{\epsilon} \psi(y^i)
$$

如果辅助超分网络 $g(\cdot)$ 经过良好预训练，$g(y^i)$ 将非常接近真实 $x^i$，这意味着残差 $|g(y^i) - y^i|$ 可以很好地近似 $|x^i - y^i|$。因此，论文将不确定性估计定义为：

$$
\psi_{\mathrm{est}}(y) = \frac{1}{2} |g(y) - y|
$$

在获得不确定性估计后，计算局部权重调制系数 $w_u$，应用其调制扩散过程中不同像素区域的噪声水平：

$$
w_u(y) := u(\psi_{\mathrm{est}}(y))
$$

该方法将 ResShift 的前向扩散分布：

$$
q(x_t \mid x_{t-1}, x_0, y_0) = \mathcal{N}\left(x_t; x_{t-1} + \alpha_t(y_0 - x_0), \kappa^2 \alpha_t \mathbf{I}\right)
$$

和后向过程分布：

$$
q(x_{t-1} \mid x_t, x_0, y_0) = \mathcal{N}\left(x_{t-1}; \frac{\eta_{t-1}}{\eta_t} x_t + \frac{\alpha_t}{\eta_t} x_0, \kappa^2 \frac{\eta_{t-1}}{\eta_t} \alpha_t \mathbf{I}\right)
$$

分别改写为各向异性调制形式：

$$
q(x_t \mid x_{t-1}, x_0, y_0) = \mathcal{N}\left(x_t; x_{t-1} + \alpha_t(y_0 - x_0), \kappa^2 w_u(y_0)^2 \alpha_t \mathbf{I}\right)
$$

以及：

$$
q(x_{t-1} \mid x_t, x_0, y_0) = \mathcal{N}\left(x_{t-1}; \frac{\eta_{t-1}}{\eta_t} x_t + \frac{\alpha_t}{\eta_t} x_0, \kappa^2 w_u(y_0)^2 \frac{\eta_{t-1}}{\eta_t} \alpha_t \mathbf{I}\right)
$$

给定一张 LR 图像 $y_0$，首先通过辅助超分网络 $g(\cdot)$ 获得其粗糙 SR 估计 $g(y_0)$；然后计算不确定性估计 $\psi_{\mathrm{est}}(y_0) = \frac{1}{2}|g(y_0) - y_0|$，得到权重调制矩阵 $w_u(y_0) = u(\psi_{\mathrm{est}}(y_0))$。

整体训练损失函数为：

$$
\mathcal{L}(\theta) = \sum_t \left[ \| f_\theta(x_t, y_0, g(y_0), t) - x_0 \|_2^2 + \lambda \mathcal{L}_{\mathrm{per}}(f_\theta(x_t, y_0, g(y_0), t), x_0) \right]
$$

其中 $\lambda$ 是平衡均方误差与感知损失 $\mathcal{L}_{\mathrm{per}}$ 的权衡超参数。

### 附录：权重映射函数 $u(\cdot)$ 的设计

论文将区域特定扰动的权重系数与不确定性估计之间的关系建模为一个单调递增的分段线性函数 $u'(\cdot)$，随后进行对角化处理：

$$
w_u(\mathbf{y}_0) = u(\boldsymbol{\psi}_{\mathrm{est}}(\mathbf{y}_0)) = \operatorname{diag}\bigl(u'(\boldsymbol{\psi}_{\mathrm{est}}(\mathbf{y}_0))\bigr)
$$

分段映射函数 $u'(\cdot)$ 构造为：

$$
u'(\psi) = \begin{cases} 
\frac{1 - b_u}{\psi_{\max}} \psi + b_u, & \text{if } 0 \le \psi \le \psi_{\max} \\
1, & \text{otherwise}
\end{cases}
$$

对于不确定性估计 $\psi_{\mathrm{est}} \in [0, \psi_{\max}]$ 的区域，将 $u'(\cdot)$ 定义为具有基线偏移量 $b_u$ 和斜率 $(1 - b_u)/\psi_{\max}$ 的线性增长函数，确保输出保持在 $[b_u, 1]$ 范围内。正偏移量 $b_u$ 确保了平坦区域具有非零的最低基线噪声水平，防止完全丧失去噪能力。
