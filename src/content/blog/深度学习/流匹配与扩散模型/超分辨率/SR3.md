---
title: "SR3"
description: "剖析SR3论文将条件扩散模型应用于图像超分辨率的方法架构，分析低分辨率图像作为条件输入时的特征融合策略与训练细节。"
date: "2026-05-17T14:00:00+08:00"
updated: "2026-05-17T14:00:00+08:00"
tags:
  - "深度学习"
  - "流匹配与扩散模型"
categories:
  - "深度学习"
  - "流匹配与扩散模型"
  - "超分辨率"
permalink: "2026/05/17/深度学习/流匹配与扩散模型/超分辨率/SR3"
math: true
draft: false
---

### Motivation
传统的超分辨率方法，如SRResnet等回归模型，直接采用均方误差。这种方法在数学上等价于在给定条件 $\mathbf{x}$ 下求解目标分布的**条件期望** $\mathbb{E}[\mathbf{y} \mid \mathbf{x}]$。然而，求期望，求平均的操作会导致多种可能的高频纹理被平均化，从而产生过度平滑的视觉效果。而 GAN 等生成对抗网络又存在训练不稳定和病态性问题

### Construction
##### A. 前向过程
前向过程与传统 DDPM 完全相同，是一个固定的马尔可夫链。在 T 个时间步内逐步向 $\mathbf{y}_0$ 添加方差预定义的**高斯噪声**

$$q(\mathbf{y}_t \mid \mathbf{y}_{t-1}) = \mathcal{N}(\mathbf{y}_t; \sqrt{1-\beta_t}\mathbf{y}_{t-1}, \beta_t \mathbf{I})$$
所以一步加噪则为
$$q(\mathbf{y}_t \mid \mathbf{y}_0) = \mathcal{N}(\mathbf{y}_t; \sqrt{\bar{\alpha}_t}\mathbf{y}_0, (1-\bar{\alpha}_t)\mathbf{I})$$
利用重参数化技巧，我们将上述过程表述为
$$\mathbf{y}_t = \sqrt{\bar{\alpha}_t}\mathbf{y}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
#### B. 反向去噪与训练
我们希望学习方向过程分布 $p_\theta(\mathbf{y}_{t-1} \mid \mathbf{y}_t, \mathbf{x})$ 以消除噪声。由于前向步骤足够小，逆向转移也可以建模为高斯分布
由贝叶斯定理，当给定 $\mathbf{y}_0$ 时，真实的后验分布 $q(\mathbf{y}_{t-1} \mid \mathbf{y}_t, \mathbf{y}_0)$ 严格可解：
$$q(\mathbf{y}_{t-1} \mid \mathbf{y}_t, \mathbf{y}_0) = \frac{q(\mathbf{y}_t \mid \mathbf{y}_{t-1})q(\mathbf{y}_{t-1} \mid \mathbf{y}_0)}{q(\mathbf{y}_t \mid \mathbf{y}_0)}$$
整理配方得：
$$q(\mathbf{y}_{t-1} \mid \mathbf{y}_t, \mathbf{y}_0) = \mathcal{N}(\mathbf{y}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{y}_t, \mathbf{y}_0), \tilde{\beta}_t \mathbf{I})$$
其中方差与均值为：
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{y}_t, \mathbf{y}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{y}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{y}_t$$
由前向推导可知 $\mathbf{y}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{y}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})$，代入 $\tilde{\boldsymbol{\mu}}_t$ 消去 $\mathbf{y}_0$：
$$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$
构造神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, t)$ 逼近 $\boldsymbol{\epsilon}$，推断分布的均值参数化为：
$$\boldsymbol{\mu}_\theta(\mathbf{x}, \mathbf{y}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, t) \right)$$
所以训练的完整流程就是
1. 采样低分辨率图像$x$和对应的高分辨率带噪图像$y_0$~$q(x, y_0)$，然后从均匀分布中采样时间步，然后获取对应的连续噪声方差$\gamma$，接着采样高斯噪声$\epsilon$
2. 使用双三次插值将$x$上采样至目标分辨率$x_{up}$，然后用重参数化后的马尔可夫链对$y_0$更新至$y_t$，然后对两个张量进行拼接，输入进神经网络，计算预测噪声$\epsilon_\theta$
3. 计算 L1 loss
4. 计算梯度，使用优化器对模型参数 $\theta$ 进行迭代
5. 重复直至 loss 收敛
#### C. 推理过程
1. 输入：一张低分辨率图片 $x$。然后用双三次插值将 $x$ 上采样至目标分辨率 $x_{up}$，创建一个与 $x_{up}$维度完全一致的高斯张量 $y_T$，这个张量就是逆向马尔可夫链的起点
2. 将这两个张量进行拼接，连同时间步信息 $\gamma_t$ 一同送入已训练好的 Unet 模型中。网络进行去噪运算，输出其对注入噪声的预测值 $\epsilon_\theta$
3. 然后使用该噪声计算一个去噪后图像的估计值$\boldsymbol{\mu}_\theta(\mathbf{x}, \mathbf{y}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, t) \right)$
4. 为这个估计值注入随机性。我们在确定性的均值上增加一个小的随机扰动，即遵循Langevin动力学$\boldsymbol{\mu}_\theta(\mathbf{x}, \mathbf{y}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, t) \right) + \sigma_tz$。但是对于最后一步 t=1 我们不再添加任何噪声，此时直接令 $y_0=\mu_\theta$
5. 最终输出 超分辨率图像$y_0$
$w = 1/(\|\Delta\|_2^2 + c)^p$ 
$\mathcal{L} = \|\Delta\|_2^{2\gamma}$
$\|\Delta\|_2^{2\gamma}$
$p = 1 - \gamma$
$\texttt{sg}(w) \cdot \mathcal{L}$
$w = 1/(\|\Delta\|_2^2 + c)^p$ 
$\mathcal{L} = \|\Delta\|_2^2$
