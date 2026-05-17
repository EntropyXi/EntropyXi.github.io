---
title: DDIM
date: 2026-05-10 20:11:00
tags:
  - 深度学习
  - 流匹配与扩散模型
mathjax: true
categories:
  - 深度学习
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
DDPM 的核心问题不在训练目标，而在采样过程。标准 DDPM 通过一个固定的前向马尔可夫链逐步把数据 $x_0$ 破坏为高斯噪声，再训练反向马尔可夫链从 $x_T$ 逐步恢复 $x_0$。由于反向过程需要依次执行，若训练时使用 $T=1000$ 个时间步，推理时通常也需要接近同等数量的网络调用。这使 DDPM 的采样速度明显慢于 GAN 或一次前向生成模型。

DDIM 的核心想法是：**DDPM 的噪声预测训练目标只依赖边缘分布 $q(x_t\mid x_0)$，并不直接依赖完整的前向联合分布 $q(x_{1:T}\mid x_0)$**。因此，只要构造一个新的前向联合分布，使其所有边缘分布仍满足

$$
q(x_t\mid x_0)=\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)\mathbf{I}\right)
$$
就可以继续使用 DDPM 已训练好的噪声预测网络，而不必重新训练。

这里需要注意符号差异。DDPM 中常把累计乘积记为 $\bar{\alpha}_t$，而 DDIM 论文中直接记为 $\alpha_t$。因此本文沿用 DDIM 论文记法：

$$
\alpha_t=\prod_{s=1}^{t}(1-\beta_s)
$$
其中 $\alpha_t$ 随 $t$ 增大而递减，$\alpha_0=1$，$\alpha_T\approx 0$。

DDIM 的贡献并不是提出新的神经网络结构，而是重新解释和改写采样过程。它构造了一族非马尔可夫前向过程 $q_\sigma(x_{1:T}\mid x_0)$，这些过程与 DDPM 拥有相同边缘分布和相同训练目标，但对应的反向生成过程可以通过控制 $\sigma_t$ 改变随机性。当 $\sigma_t=0$ 时，采样过程退化为确定性映射，从而形成所谓的 implicit model。

### Construction
##### A. 前向过程
DDPM 的标准前向过程为

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}\left(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\mathbf{I}\right)
$$
递推可得边缘分布

$$
q(x_t\mid x_0)
=
\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)\mathbf{I}\right)
$$
利用重参数化技巧，可以写成

$$
x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon,
\quad
\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
DDPM 的训练目标一般写为

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right]
$$
其中 $x_t$ 只通过边缘分布 $q(x_t\mid x_0)$ 生成。因此，只要新的前向过程保持这个边缘分布不变，训练样本的构造方式就不变，噪声预测目标也不变。

DDIM 构造的非马尔可夫联合分布为

$$
q_\sigma(x_{1:T}\mid x_0)
=
q_\sigma(x_T\mid x_0)
\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0)
$$
其中

$$
q_\sigma(x_T\mid x_0)
=
\mathcal{N}\left(x_T;\sqrt{\alpha_T}x_0,(1-\alpha_T)\mathbf{I}\right)
$$
并定义

$$
q_\sigma(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}\left(
x_{t-1};
\sqrt{\alpha_{t-1}}x_0
+\sqrt{1-\alpha_{t-1}-\sigma_t^2}
\frac{x_t-\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}},
\sigma_t^2\mathbf{I}
\right)
$$
这个式子是 DDIM 的关键。它不是把 $x_t$ 只看作由 $x_{t-1}$ 转移而来，而是在给定 $x_0$ 的条件下，让 $x_{t-1}$ 直接依赖 $x_t$ 和 $x_0$。因此该前向过程一般不是马尔可夫过程。

下面说明为什么它仍然保持 DDPM 的边缘分布。假设

$$
x_t\mid x_0
\sim
\mathcal{N}\left(\sqrt{\alpha_t}x_0,(1-\alpha_t)\mathbf{I}\right)
$$
则可以写成

$$
x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon_t,
\quad
\epsilon_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
代入 $q_\sigma(x_{t-1}\mid x_t,x_0)$ 的均值项：

$$
\frac{x_t-\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}}
=
\epsilon_t
$$
因此

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}x_0
+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\epsilon_t
+\sigma_t z_t
$$
其中

$$
z_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
且 $z_t$ 与 $\epsilon_t$ 独立。由于两个独立高斯变量的线性组合仍为高斯变量，其均值为零，方差为

$$
\left(1-\alpha_{t-1}-\sigma_t^2\right)\mathbf{I}
+\sigma_t^2\mathbf{I}
=
(1-\alpha_{t-1})\mathbf{I}
$$
所以

$$
x_{t-1}\mid x_0
\sim
\mathcal{N}\left(
\sqrt{\alpha_{t-1}}x_0,
(1-\alpha_{t-1})\mathbf{I}
\right)
$$
这说明只要 $x_t$ 的边缘分布与 DDPM 一致，由上述条件分布生成的 $x_{t-1}$ 边缘分布也与 DDPM 一致。由反向归纳可知，所有 $q_\sigma(x_t\mid x_0)$ 都保持为

$$
q_\sigma(x_t\mid x_0)
=
\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)\mathbf{I}\right)
$$

这一步是 DDIM 推导中比较关键的地方。DDIM 改变的是 $x_1,\cdots,x_T$ 之间的联合依赖关系，而不是单个时间步的边缘加噪分布。因此训练时仍然可以按照 DDPM 的方式，从 $x_0$ 直接采样 $x_t$ 并训练 $\epsilon_\theta(x_t,t)$。

#### B. 反向去噪与训练
DDIM 的训练目标仍然是噪声预测。给定

$$
x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon
$$
神经网络学习

$$
\epsilon_\theta(x_t,t)\approx \epsilon
$$
于是可以由 $x_t$ 和预测噪声反推出对 $x_0$ 的估计：

$$
\hat{x}_0(x_t,t)
=
\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}
$$
这个式子只是对前向重参数化公式做代数变形。若 $\epsilon_\theta(x_t,t)$ 精确等于真实噪声 $\epsilon$，则 $\hat{x}_0=x_0$。

DDIM 的反向生成过程利用 $q_\sigma(x_{t-1}\mid x_t,x_0)$ 的形式，但推理时真实 $x_0$ 不可见，所以用 $\hat{x}_0$ 替代 $x_0$。因此采样公式为

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}
\left(
\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\alpha_t}}
\right)
+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\epsilon_\theta(x_t,t)
+\sigma_t\epsilon_t
$$
其中

$$
\epsilon_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

将这个公式拆开，可以得到三个部分：

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}
\underbrace{
\left(
\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\alpha_t}}
\right)
}_{\text{预测的 }x_0}
+\underbrace{
\sqrt{1-\alpha_{t-1}-\sigma_t^2}\epsilon_\theta(x_t,t)
}_{\text{指向 }x_t\text{ 的方向}}
+\underbrace{
\sigma_t\epsilon_t
}_{\text{随机噪声}}
$$
第一项是把当前状态 $x_t$ 还原到数据空间估计 $\hat{x}_0$，再按照时间步 $t-1$ 的信噪比缩放。第二项保留与当前噪声方向一致的分量，使 $x_{t-1}$ 仍然落在正确的边缘噪声水平上。第三项是额外随机性，由 $\sigma_t$ 控制。

因此，$\sigma_t$ 并不是训练得到的网络输出，而是采样过程中的方差控制参数。论文给出一个常用参数化：

$$
\sigma_t(\eta)
=
\eta
\sqrt{
\frac{1-\alpha_{t-1}}{1-\alpha_t}
}
\sqrt{
1-\frac{\alpha_t}{\alpha_{t-1}}
}
$$
其中 $\eta$ 是控制随机性的超参数。当 $\eta=1$ 时，对应与 DDPM 后验方差一致的随机采样形式；当 $\eta=0$ 时，$\sigma_t=0$，采样过程变成确定性过程。

从训练角度看，DDIM 没有引入新的损失函数。其变分目标可以化为带权重的噪声预测目标：

$$
\mathcal{L}_\gamma(\theta)
=
\sum_{t=1}^{T}
\gamma_t
\mathbb{E}_{x_0,\epsilon}
\left[
\left\|
\epsilon_\theta
\left(
\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon,t
\right)
-\epsilon
\right\|_2^2
\right]
$$
其中 $\gamma_t$ 是与时间步和方差选择有关的正权重。与 DDPM 类似，实际实现中常使用简化的未加权目标。其原因是最优的噪声预测函数本身不依赖这些正权重，只要不同时间步的模型输出没有强制共享到无法表达的程度，改变权重主要影响优化过程，而不是目标函数的理论最优解。

完整训练流程可以概括为：
1. 从数据分布中采样 $x_0$
2. 从时间步集合中采样 $t$
3. 采样高斯噪声 $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$
4. 构造 $x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$
5. 输入 $x_t$ 和 $t$，由网络预测 $\epsilon_\theta(x_t,t)$
6. 最小化 $\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2$

这与 DDPM 的训练流程一致。因此 DDIM 可以直接复用一个已经训练好的 DDPM 噪声预测网络。

#### C. 推理过程
推理时从标准高斯噪声开始：

$$
x_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
然后选择一个从大到小的采样时间序列

$$
\tau_S>\tau_{S-1}>\cdots>\tau_1
$$
其中 $S$ 可以远小于训练时的 $T$。如果训练时 $T=1000$，推理时可以只选择 $S=50$、$S=20$，甚至更少的时间步。DDIM 的加速来自这里：网络调用次数从 $T$ 次减少到 $S$ 次。

在每一步，先计算

$$
\hat{x}_0
=
\frac{x_{\tau_i}-\sqrt{1-\alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i},\tau_i)}
{\sqrt{\alpha_{\tau_i}}}
$$
然后按 DDIM 采样公式得到上一状态：

$$
x_{\tau_{i-1}}
=
\sqrt{\alpha_{\tau_{i-1}}}\hat{x}_0
+\sqrt{1-\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}
\epsilon_\theta(x_{\tau_i},\tau_i)
+\sigma_{\tau_i}\epsilon_i
$$
其中

$$
\epsilon_i\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
最后得到 $x_0$ 的生成样本。

当 $\sigma_t=0$ 时，上述公式变为

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}\hat{x}_0
+\sqrt{1-\alpha_{t-1}}\epsilon_\theta(x_t,t)
$$
也就是

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}
\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\alpha_t}}
+\sqrt{1-\alpha_{t-1}}\epsilon_\theta(x_t,t)
$$
此时给定初始噪声 $x_T$、模型参数 $\theta$ 和采样时间序列，整个生成过程没有额外随机项，输出是确定的。DDIM 中的 implicit 主要指这一点：模型没有显式给出一个可逐步计算似然的随机反向链，而是通过固定的确定性过程把潜变量 $x_T$ 映射到样本 $x_0$。

完整推理流程可以概括为：
1. 采样初始潜变量 $x_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$
2. 选择采样子序列 $\{\tau_i\}_{i=1}^{S}$，其中 $S\ll T$
3. 在每个时间步用网络预测 $\epsilon_\theta(x_{\tau_i},\tau_i)$
4. 将预测噪声换算为 $\hat{x}_0$
5. 根据 $\sigma_{\tau_i}$ 选择确定性或随机性采样
6. 重复直到得到 $x_0$

### Noise Schedule
DDIM 中需要区分两个概念：训练噪声日程 $\alpha_t$ 和推理采样轨迹 $\tau$。

训练噪声日程决定边缘分布

$$
q(x_t\mid x_0)
=
\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)\mathbf{I}\right)
$$
它一般在训练前固定，例如线性 $\beta_t$ 或余弦日程。DDIM 并不要求重新设计训练噪声日程。

推理采样轨迹决定实际访问哪些时间步。若训练使用 $T=1000$，DDIM 可以只取一个子序列：

$$
\tau=\{1,20,40,\cdots,1000\}
$$
反向采样时从 $\tau_S=1000$ 走到 $\tau_1=1$。这种做法没有改变训练分布，而是跳过大量中间时间步，用非马尔可夫反向过程直接连接较远的噪声水平。

方差参数 $\sigma_t$ 控制采样随机性。常用形式为

$$
\sigma_t(\eta)
=
\eta
\sqrt{
\frac{1-\alpha_{t-1}}{1-\alpha_t}
}
\sqrt{
1-\frac{\alpha_t}{\alpha_{t-1}}
}
$$
其中 $\eta$ 可以理解为随机性插值系数：

$$
\eta=0
\quad\Rightarrow\quad
\text{deterministic DDIM}
$$

$$
\eta=1
\quad\Rightarrow\quad
\text{DDPM-like stochastic sampling}
$$

当 $\eta$ 较小时，采样轨迹更稳定，不同采样步数下的高层语义更容易保持一致；当 $\eta$ 较大时，过程引入更多随机扰动，生成样本的多样性可能增加，但在极短采样链中更容易积累误差。

### Experiments
论文的实验重点不是证明 DDIM 在完整 1000 步采样时全面超过 DDPM，而是比较在较少采样步数下的质量和效率。实验中使用相同训练好的 DDPM 模型，只改变采样过程，因此可以直接观察采样器本身的影响。

在 CIFAR10 和 CelebA 等数据集上，论文使用 FID 评价生成质量。实验结果显示，当采样步数较少时，确定性或低随机性的 DDIM 通常比 DDPM 式采样更稳定。特别是在 $10$、$20$、$50$ 步这类短轨迹设置下，DDPM 的随机反向链容易因为跳步过大而产生明显质量下降，而 DDIM 由于直接构造跨时间步的确定性方向，退化相对较小。

论文报告 DDIM 可以在墙钟时间上达到约 $10\times$ 到 $50\times$ 的采样加速。这个结论的来源很直接：采样时间大致与网络调用次数线性相关。如果从 $1000$ 步减少到 $100$、$50$ 或 $20$ 步，理论计算量也近似按相同比例下降。

另一个实验现象是 latent consistency。对于确定性 DDIM，若固定同一个初始噪声 $x_T$，即使使用不同长度的采样轨迹，生成图像的高层语义仍倾向于保持一致。这一点与 DDPM 不同。DDPM 在每一步都加入随机噪声，即使初始 $x_T$ 相同，后续随机扰动也会改变生成路径。因此 DDIM 更适合直接在潜变量 $x_T$ 上做插值。

需要客观看待这些结果。DDIM 的少步采样提升并不意味着任意少的步数都能保持质量。当采样步数压缩到极低时，网络预测误差和离散化误差会被放大，生成质量仍会下降。DDIM 的实际价值在于给出一个更合理的少步采样路径，而不是消除少步采样本身的信息损失。

### Appendix
##### A. 与 DDPM 的差异
DDPM 的前向过程是马尔可夫链：

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
$$
其中每一步只依赖上一个状态。其反向生成过程也建模为马尔可夫链：

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
$$

DDIM 的非马尔可夫前向过程则为

$$
q_\sigma(x_{1:T}\mid x_0)
=
q_\sigma(x_T\mid x_0)
\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0)
$$
其中 $x_{t-1}$ 的条件分布显式依赖 $x_0$。因此它不再是标准意义上的前向马尔可夫扩散过程。

两者的关键相同点是边缘分布一致：

$$
q_\sigma(x_t\mid x_0)
=
q(x_t\mid x_0)
=
\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)\mathbf{I}\right)
$$
这也是 DDIM 可以复用 DDPM 训练目标的原因。

两者的关键差异在反向采样。DDPM 的每一步都按照高斯后验加入随机扰动；DDIM 则允许通过 $\sigma_t$ 控制随机性。当 $\sigma_t=0$ 时，DDIM 的反向过程是确定性的：

$$
x_T \mapsto x_{T-1}\mapsto \cdots \mapsto x_0
$$
这个映射由神经网络和采样日程共同决定。

##### B. 确定性极限与 ODE 视角
当 $\sigma_t\to 0$ 时，DDIM 采样公式中的随机项消失：

$$
\sigma_t\epsilon_t\to 0
$$
因此

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}\hat{x}_0
+\sqrt{1-\alpha_{t-1}}\epsilon_\theta(x_t,t)
$$
此时每一步都是确定性函数：

$$
x_{t-1}=F_\theta(x_t,t)
$$
给定 $x_T$ 后，整个轨迹唯一确定。

为了看出 ODE 形式，引入变量

$$
\bar{x}_t=\frac{x_t}{\sqrt{\alpha_t}}
$$
以及噪声尺度

$$
\sigma^\mathrm{noise}_t
=
\sqrt{\frac{1-\alpha_t}{\alpha_t}}
$$
注意这里的 $\sigma^\mathrm{noise}_t$ 表示信噪比重参数化后的噪声尺度，不是前文控制随机性的 $\sigma_t$。由

$$
x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon
$$
两边除以 $\sqrt{\alpha_t}$ 得到

$$
\bar{x}_t=x_0+\sigma^\mathrm{noise}_t\epsilon
$$
DDIM 的确定性更新可以写成近似形式

$$
\bar{x}_{t-1}
=
\bar{x}_t
+
\left(
\sigma^\mathrm{noise}_{t-1}
-
\sigma^\mathrm{noise}_{t}
\right)
\epsilon_\theta(x_t,t)
$$
这与欧拉法求解常微分方程的形式一致：

$$
\frac{d\bar{x}}{d\sigma^\mathrm{noise}}
=
\epsilon_\theta(x,t)
$$
或者写成微分形式：

$$
d\bar{x}
=
\epsilon_\theta(x,t)\ d\sigma^\mathrm{noise}
$$
因此，当时间步足够密集时，确定性 DDIM 可以看作对某条概率流 ODE 的离散化近似。反向生成是沿 ODE 从噪声端积分到数据端；若模型足够准确，也可以反向积分，把数据编码回潜变量。

这个解释并不意味着 DDIM 本身就是连续模型。实际实现仍然是离散采样，只是当 $\sigma_t=0$ 且步长变小时，其更新形式与神经常微分方程的欧拉离散化具有一致结构。

DDIM 的主要优势有三个。

第一，它不需要重新训练模型。因为训练目标只依赖 $q(x_t\mid x_0)$，而 DDIM 保持该边缘分布不变，所以已有 DDPM 模型可以直接换用 DDIM 采样器。

第二，它显著减少采样步数。通过选择较短的采样子序列，DDIM 可以把原本上百或上千次网络调用压缩到几十次。这种加速主要来自减少网络前向次数，而不是单次网络计算变快。

第三，确定性采样带来更稳定的潜空间结构。当 $\sigma_t=0$ 时，$x_T$ 与生成样本之间形成确定性映射，因此可以在 $x_T$ 空间进行插值，并得到相对连续的语义变化。

DDIM 的局限也比较明确。

第一，少步采样仍然会损失质量。DDIM 改善了短采样链的稳定性，但不能消除离散步长过大带来的误差。步数极少时，$\epsilon_\theta$ 的预测误差会在跨步更新中被放大。

第二，DDIM 的效果依赖原 DDPM 模型的噪声预测能力。它本质上是采样过程改写，而不是数据分布建模能力的独立提升。如果基础模型训练不足，DDIM 无法通过采样公式弥补模型误差。

第三，确定性过程会降低采样随机性。对于需要更高多样性的任务，完全令 $\sigma_t=0$ 不一定总是最优。实际使用时通常需要在采样速度、样本质量和多样性之间调节 $\eta$。

因此，更准确的概括是：DDIM 提供了一种在不改变 DDPM 训练的前提下重构采样路径的方法。它的本质贡献是把“慢速随机反向马尔可夫链”扩展为“一族可控随机性、可跳步、可确定化的反向生成过程”。这使 DDPM 的训练结果可以被更高效地利用，但并不改变扩散模型依赖噪声预测网络学习数据先验这一事实。
