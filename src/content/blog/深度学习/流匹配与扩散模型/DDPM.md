---
title: "DDPM"
description: "系统推导去噪扩散概率模型的核心原理，包括前向加噪过程的马尔可夫性质、反向去噪的变分下界推导以及ε参数化下的简化训练目标。"
date: "2026-03-15T21:40:00+08:00"
updated: "2026-03-15T21:40:00+08:00"
tags:
  - "深度学习"
  - "流匹配与扩散模型"
categories:
  - "深度学习"
  - "流匹配与扩散模型"
permalink: "2026/03/15/深度学习/流匹配与扩散模型/DDPM"
math: true
draft: false
---

### Motivation
我们希望把一个纯粹的噪声转化成一张可读图片（去噪）。

### 加噪
想要关注去噪过程，我们先来看加噪过程是怎么样的。我们尝试以下过程：

$$
x_t = x_0 + t\beta\epsilon
$$

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; x_0, t\beta\mathbf{I})
$$

但是我们会发现，以这样的方式来进行加噪会导致方差随时间步线性发散（$t\beta$ 项），而我们希望加噪终点的边缘分布保持为标准高斯分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。因此我们引入方差保持的加噪形式：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})
$$

由高斯分布的递推性质，可直接一步得到任意时刻 $t$ 的边缘分布：

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$。当 $t\to\infty$ 时，$\bar{\alpha}_t\to 0$，$1-\bar{\alpha}_t\to 1$，从而收敛到标准高斯分布。

利用重参数化技巧，可以将前向采样表达为：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

### 去噪
去噪的过程，形式上我们就是要去最大化 $p(x_0)$；对于神经网络而言，我们就是要最大化 $p_\theta(x_{t-1} \mid x_t)$。为了概率密度函数的计算方便，我们等价于最小化负对数似然 $-\log p_\theta(x_0)$。

我们再次回顾整个正向过程：

$$
\begin{aligned}
q(x_1,\dots,x_T \mid x_0) &= q(x_1 \mid x_0)q(x_2 \mid x_1,x_0)\dots q(x_T \mid x_{T-1},\dots,x_0) \\
&= q(x_1 \mid x_0)q(x_2 \mid x_1) \dots q(x_T \mid x_{T-1})
\end{aligned}
$$

我们会发现，每个加噪步骤只依赖于前一步（马尔可夫性质）：

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T}q(x_t \mid x_{t-1})
$$

所以逆向联合分布建模为：

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1} \mid x_t)
$$

> **为什么 $p(x_T)$ 不是个条件概率？** 因为它不需要条件，我们是从无条件的标准高斯噪声 $p(x_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始逆向采样的。

### 损失函数
因为我们训练的目标是最小化负对数似然，那我们不妨展开一下：

$$
\begin{aligned}
-\log p_\theta(x_0) &= -\log \int p_\theta(x_{0:T}) \, dx_{1:T} \quad \text{（对所有隐变量路径积分）} \\
&= -\log \int q(x_{1:T} \mid x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \, dx_{1:T} \\
&= -\log \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]
\end{aligned}
$$

由 Jensen 不等式：

$$
-\log p_\theta(x_0) \leq -\mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]
$$

右侧即是我们的变分下界（ELBO）。把 ELBO 展开（详见 DDPM 附录 A）：

$$
-\mathbb{E}_q\left[D_{\mathrm{KL}}(q(x_T \mid x_0)\parallel p(x_T))\right] + \sum_{t>1} D_{\mathrm{KL}}(q(x_{t-1} \mid x_t,x_0)\parallel p_\theta(x_{t-1} \mid x_t)) - \log p_\theta(x_0 \mid x_1)
$$

因为第一项是最先与最后的比较，与模型参数 $\theta$ 无关；最后一项是最后一步的重建项，对于上千步的扩散过程可以独立建模，所以主体损失简化为：

$$
\mathcal{L}_{\mathrm{vlb}} = \mathbb{E}_q\left[\sum_{t>1}D_{\mathrm{KL}}(q(x_{t-1} \mid x_t,x_0)\parallel p_\theta(x_{t-1} \mid x_t))\right]
$$

我们衡量的是真实后验分布 $q(x_{t-1} \mid x_t, x_0)$ 与参数化模型分布 $p_\theta(x_{t-1} \mid x_t)$ 之间的 KL 散度。

**关于真实后验的小注释**：

因为真实的无条件逆向转移 $q(x_{t-1} \mid x_t)$ 包含不可积的整体数据分布，无法直接解析计算，所以需要神经网络来逼近它。要训练神经网络 $\theta$，我们需要一个真实的监督目标。

如果我们以真实数据 $x_0$ 作为条件，即已知当前马尔可夫链是从哪一张具体的真实图片 $x_0$ 出发的，逆向后验分布就完全解析可解了。

利用贝叶斯公式和马尔可夫性质展开 $q(x_{t-1} \mid x_t, x_0)$：

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

根据马尔可夫性质，$x_t$ 只依赖于 $x_{t-1}$，因此 $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$，代入后得到：

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

各项均为高斯分布，配方后得到也是一个高斯分布：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\boldsymbol{\mu}}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})
$$

其中方差与均值分别为：

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

$$
\tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

由前向公式 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$ 反解出 $x_0$：

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left(x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}\right)
$$

代入 $\tilde{\boldsymbol{\mu}}_t$ 消去 $x_0$：

$$
\tilde{\boldsymbol{\mu}}_t(x_t, \boldsymbol{\epsilon}) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)
$$

在 DDPM 中，作者固定反向过程的方差为 $\sigma_t^2 \mathbf{I}$（如 $\sigma_t^2 = \tilde{\beta}_t$ 或 $\beta_t$），将参数化均值建模为：

$$
\boldsymbol{\mu}_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right)
$$

两个同方差高斯分布之间的 KL 散度简化为均值差的二范数平方：

$$
D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t)) = \frac{1}{2\sigma_t^2} \| \tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta \|^2
$$

代入 $\tilde{\boldsymbol{\mu}}_t$ 与 $\boldsymbol{\mu}_\theta$：

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \boldsymbol{\epsilon}} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t) \|^2 \right]
$$

#### 简化损失函数（Simple Loss）

作者发现丢弃复杂的权重系数、直接使用均方误差，能够大幅提升样本生成质量：

$$
\mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{t, x_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t) \|^2 \right]
$$

在实际训练时，每次从 $\{1, \dots, T\}$ 中均匀随机采样时间步 $t$，采样高斯噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，仅需一次前向网络调用即可计算梯度并更新网络。
