---
title: DDPM
date: 2026-03-15 21:40:00
tags:
  - 深度学习
  - 流匹配与扩散模型
mathjax: true
categories:
  - 深度学习
  - 流匹配与扩散模型
---
<style>
/* 强制让 MathJax 公式容器支持横向滚动 */
.mjx-container, .MathJax_Display, .MathJax {
    overflow-x: auto !important;
    overflow-y: hidden;
    max-width: 100%;
    -webkit-overflow-scrolling: touch;
}
</style>

### Motivation
我们希望把一个存粹的噪声转化成一张可读图片（去噪）  

### 加噪
想要关注去噪过程，我们先来看加噪过程是怎么样的
我们尝试以下过程

$x_t=x_0+t\beta\epsilon$
$q(x_t|x_0)=N(x_0,t\beta)$

但是我们会发现，以这样的方式来进行加噪会导致方差爆炸($t\beta$项)，而我们想要的这个纯粹的噪声一般指的是标准高斯分布，所以我们做出以下变化

$q(x_t|x_{t-1})=\sqrt{1-\beta}x_{t-1}+\sqrt{\beta}\cdot\epsilon$
$q(x_t|x_0)=\sqrt{\bar{\alpha}_t}x_0+\sqrt{(1-\bar{\alpha}_t)}\cdot\epsilon$
$\text{where} \quad \bar{\alpha}_t=(1-\beta)^t$
因为$t\to\infty,\quad\bar{\alpha}_t\to0,\quad((1-\bar{\alpha}_t)\to1)$
这就达成了我们的结果。
而我们现在用的是固定的$\beta$，而论文中每次变换用的是不同的$\beta$，所以我们把$\bar{\alpha}_t$改写成这样
$\bar{\alpha}_t=\prod \limits_{t=1}^{T}(1-\beta_t)$

### 去噪
去噪的过程，形式上我们就是要去最大化$p(x_0)$，对于神经网络而言，我们就是要最大化$p_\theta(x_{t-1}|x_t)$.为了概率密度函数的计算方便，我们其实就是在最小化负对数似然$-\log p_\theta(x_0)$

我们再次回顾整个正向过程
$q(x_1,\dots,x_T|x_0)=q(x_1|x_0)q(x_2|x_1,x_0)\dots q(x_T|x_{T-1},\dots,x_0)$
$q(x_1,\dots,x_T|x_0)=q(x_1|x_0)q(x_2|x_1) \dots q(x_T|x_{T-1})$
我们会发现，每个加噪步骤只依赖于前一步（马尔可夫性质）
$q(x_{1:T}|x_0)=q(x_1|x_0)q(x_2|x_1) \dots q(x_T|x_{T-1}$
$q(x_{1:T}|x_0)=\prod \limits_{t=1}^{T}q(x_t|x_{t-1})$

所以，$p_\theta(x_{0:T})=p(x_T)\prod \limits_{t=1}^{T}p_\theta(x_{t-1}|x_t)$
`为什么P_theta不是个条件概率？因为他不需要条件，我们是从Gaussian噪声开始的，不带任何先验知识`

### 损失函数
因为我们训练的目标是最小化负对数似然，那我们不妨展开一下
$-\log p_\theta(x_0)=-\log \int p_\theta(x_{0:T})dx_{1:T}$，`对所有条件路径求和`
$-\log p_\theta(x_0)=-\log \int q(x_{1:T}|x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}dx_{1:T}$
$-\log p_\theta(x_0)=-\log E_{q(x_{1:T}|x_0)}[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]$
由Jensen不等式，
$-\log p_\theta(x_0) \leq -E_{q(x_{1:T}|x_0)}[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]$
右侧即是我们的ELBO
我们把ELBO展开（详见DDPM附录A）
$-E_q[D_{KL}(q(x_T|x_0)||p(x_T)))]+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))-\log p_\theta(x_0|x_1))]$
因为第一项是最先与最后的比较，模型无关
最后一项是最后一步的过程，对于上千步的过程来说可以忽略不计
所以最后就变成
$-E_q[\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))]$
我们衡量的是真实后验分布与我们每一步的分布求和的KL散度

**关于真实后验的小注释**：
	因为真实的$q(x_{t-1}|x_t)$无法计算，所以引入神经网络来近似它。但是，要训练theta，我们需要一个真实标签。如果目标$q(x_{t-1}|x_t)$本身算不出来，损失函数就无法构建，我们需要在$q$中寻找替代目标
	所以如果我们条件化$x_0$，即假设我们知道当前这条马尔可夫链是从哪一张具体的真实图片$x_0$开始加噪的，那么逆向过程的后验概率就完全可解了
	我们利用贝叶斯公式和马尔可夫性质展开$q(x_{t-1} | x_t, x_0)$

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)}$$
根据马尔可夫性质，$x_t$只依赖于$x_{t-1}$，因此$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$，带入后得到
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}$$全部是已知量，所以$q(x_{t-1} | x_t, x_0)$是一个均值和方差完全已知的高斯分布$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$$
我们的假设基于我们知道$x_0$，但在真实情况中，我们不知道$x_0$，所以我们需要神经网络去拟合
论文中，作者固定了方差$\sigma_t$，这意味着我们只需要去拟合均值$\mu_t$。所以就是要去衡量均值的相对距离，KL散度就变成$D_{KL}=\frac{1}{2\sigma_t^2}||\hat{\mu}_t-\mu_0||^2$
所以，我们要去最小化的就是这个东西$E_q[\sum_{t>1}\frac{1}{2\sigma_t^2}||\hat{\mu}_t-\mu_0||^2]$
我们可以写成这样
$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$
$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)$
$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$
$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$
The Final Loss
$\mathcal{L} = \mathbb{E}_q \left[ \sum_{t>1} \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$
但是，求和的时间复杂度还是太大了。我们需要一种方法，能够在每次迭代时只计算极少数（甚至一个）时间步的损失，但同时保证这种简化后的计算方向（即梯度）在统计意义上仍然指向全局最优解。
我们认为KL散度是离散均匀分布的，所以我们可以写成
$$T\cdot\mathbb{E}_{q, t} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
在SGD中，目标函数乘以一个大于0的常数不会改变极值点位置。因此我们通常丢弃常数T，得到最终形式
$$\mathbb{E}_{q, t} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
$\mathcal{L} =  \sum_{t>1} \mathbb{E}_q \left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$
$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$
