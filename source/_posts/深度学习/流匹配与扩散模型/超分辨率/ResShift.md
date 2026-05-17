---
title: ResShift
date: 2026-05-17 14:00:00
tags:
  - 深度学习
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
传统基于扩散模型的超分辨率方法，如 SR3 或 LDM 类方法，通常继承 DDPM 的基本假设：前向过程需要把目标高分辨率图像逐步破坏到标准高斯噪声，然后反向过程再从高斯噪声恢复图像。这个设计在无条件图像生成中是合理的，因为生成任务没有额外观测量，只能从一个固定先验分布开始采样。

但是在超分辨率任务中，低分辨率图像 $y_0$ 已经给出了大量结构信息。若仍然从纯高斯噪声开始反向采样，就相当于忽略了 $y_0$ 中已有的低频轮廓，只把它作为条件输入给网络。这会导致两个问题：第一，马尔可夫链需要足够长，才能从噪声状态逐步回到图像流形；第二，后续用 DDIM 等方法强行压缩采样步数时，容易造成细节不足或过度平滑。

ResShift 的核心想法是：**超分辨率并不需要从纯噪声生成整张 HR 图像，而是需要把 LR 图像与 HR 图像之间的残差逐步补回来**。因此论文不再构造从 $x_0$ 到高斯白噪声的扩散链，而是构造一条连接 HR 图像 $x_0$ 与 LR 图像 $y_0$ 的马尔可夫链。前向过程逐步把 $x_0$ 平移到 $y_0$ 附近，反向过程则从 $y_0$ 附近开始，逐步恢复出 $x_0$。

### Construction
##### A. 前向过程
论文记 HR 图像为 $x_0$，LR 图像为 $y_0$。为了使二者可以直接相减，若 $y_0$ 的空间尺寸较小，则先通过插值将其上采样到与 $x_0$ 相同的分辨率。定义两者之间的残差为

$$
e_0 = y_0 - x_0
$$
这里的残差方向是从 HR 指向 LR。因此，如果从 $x_0$ 出发并逐步加入 $e_0$，最终状态就会移动到 $y_0$ 附近。

论文引入一个单调递增的平移序列 $\{\eta_t\}_{t=1}^{T}$，满足 $\eta_1 \to 0$，$\eta_T \to 1$。记

$$
\alpha_t =
\begin{cases}
\eta_1, & t=1 \\
\eta_t-\eta_{t-1}, & t>1
\end{cases}
$$
于是 ResShift 的前向转移分布被定义为

$$
q(x_t \mid x_{t-1}, x_0, y_0)
=
\mathcal{N}\left(x_t;\ x_{t-1}+\alpha_t e_0,\ \kappa^2 \alpha_t \mathbf{I}\right)
$$
其中 $\kappa$ 是控制整体噪声强度的超参数，$\mathbf{I}$ 是单位矩阵。

这个式子可以拆成两个部分理解：

$$
x_t = x_{t-1}+\alpha_t(y_0-x_0)+\kappa\sqrt{\alpha_t}\epsilon_t,
\quad \epsilon_t \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$
第一项 $x_{t-1}$ 是当前状态，第二项 $\alpha_t(y_0-x_0)$ 是向 LR 图像方向移动的残差平移，第三项是高斯扰动。与 DDPM 的区别在于，DDPM 的前向均值是对原图做缩放，而 ResShift 的前向均值是沿着 $x_0 \to y_0$ 的残差方向移动。

由递推关系可得边缘分布：

$$
q(x_t \mid x_0,y_0)
=
\mathcal{N}\left(x_t;\ x_0+\eta_t e_0,\ \kappa^2\eta_t\mathbf{I}\right)
$$
即

$$
x_t = x_0+\eta_t(y_0-x_0)+\kappa\sqrt{\eta_t}\epsilon
$$
也可以写成

$$
x_t = (1-\eta_t)x_0+\eta_t y_0+\kappa\sqrt{\eta_t}\epsilon
$$
这个式子说明 $x_t$ 的确定性部分是 $x_0$ 与 $y_0$ 的线性插值。当 $\eta_t \to 0$ 时，$x_t$ 接近 $x_0$；当 $\eta_t \to 1$ 时，$x_t$ 接近 $y_0$，并叠加方差为 $\kappa^2\mathbf{I}$ 的扰动。

上述边缘分布可以由归纳法得到。假设

$$
x_{t-1}=x_0+\eta_{t-1}e_0+\kappa\sqrt{\eta_{t-1}}\epsilon_{t-1}
$$
代入前向转移公式：

$$
x_t
=x_0+(\eta_{t-1}+\alpha_t)e_0
+\kappa\sqrt{\eta_{t-1}}\epsilon_{t-1}
+\kappa\sqrt{\alpha_t}\epsilon_t
$$
由于 $\eta_{t-1}+\alpha_t=\eta_t$，且两个独立高斯变量之和仍为高斯变量，其方差相加，所以噪声项整体服从

$$
\kappa\sqrt{\eta_t}\epsilon,\quad \epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
因此得到边缘分布 $q(x_t\mid x_0,y_0)$。

#### B. 反向去噪与训练
反向过程希望学习

$$
p_\theta(x_{t-1}\mid x_t,y_0)
$$
也就是给定当前状态 $x_t$ 和 LR 条件 $y_0$，预测上一个状态 $x_{t-1}$。与 DDPM 类似，论文将反向转移分布建模为高斯分布：

$$
p_\theta(x_{t-1}\mid x_t,y_0)
=
\mathcal{N}\left(x_{t-1};\ \mu_\theta(x_t,y_0,t),\ \Sigma_\theta(x_t,y_0,t)\right)
$$
训练时 $x_0$ 是已知的，因此真实后验

$$
q(x_{t-1}\mid x_t,x_0,y_0)
$$
可以解析计算。由贝叶斯公式：

$$
q(x_{t-1}\mid x_t,x_0,y_0)
\propto
q(x_t\mid x_{t-1},x_0,y_0)q(x_{t-1}\mid x_0,y_0)
$$
其中

$$
q(x_{t-1}\mid x_0,y_0)
=
\mathcal{N}\left(x_{t-1};\ x_0+\eta_{t-1}e_0,\ \kappa^2\eta_{t-1}\mathbf{I}\right)
$$
而

$$
q(x_t\mid x_{t-1},x_0,y_0)
=
\mathcal{N}\left(x_t;\ x_{t-1}+\alpha_t e_0,\ \kappa^2\alpha_t\mathbf{I}\right)
$$
令 $a=\eta_{t-1}$，$b=\alpha_t$，则 $\eta_t=a+b$。把第二个分布写成关于 $x_{t-1}$ 的观测形式：

$$
x_t-b e_0 = x_{t-1}+\kappa\sqrt{b}\epsilon
$$
这相当于先验

$$
x_{t-1}\sim \mathcal{N}(x_0+a e_0,\kappa^2a\mathbf{I})
$$
和观测

$$
x_t-b e_0 \mid x_{t-1}\sim \mathcal{N}(x_{t-1},\kappa^2b\mathbf{I})
$$
两个高斯分布相乘。利用高斯共轭性质，后验方差为

$$
\left(\frac{1}{\kappa^2a}+\frac{1}{\kappa^2b}\right)^{-1}\mathbf{I}
=
\kappa^2\frac{ab}{a+b}\mathbf{I}
=
\kappa^2\frac{\eta_{t-1}}{\eta_t}\alpha_t\mathbf{I}
$$
后验均值为

$$
\frac{b}{a+b}(x_0+a e_0)+\frac{a}{a+b}(x_t-b e_0)
$$
整理后，含 $e_0$ 的两项相互抵消：

$$
\frac{b}{a+b}ae_0-\frac{a}{a+b}be_0=0
$$
所以真实后验为

$$
q(x_{t-1}\mid x_t,x_0,y_0)
=
\mathcal{N}\left(
x_{t-1};\
\frac{\eta_{t-1}}{\eta_t}x_t+\frac{\alpha_t}{\eta_t}x_0,\
\kappa^2\frac{\eta_{t-1}}{\eta_t}\alpha_t\mathbf{I}
\right)
$$
这一步是 ResShift 推导中比较关键的地方。虽然前向过程显式依赖 $y_0-x_0$，但是在给定 $x_t$ 和 $x_0$ 后，真实后验均值中的残差项会抵消，最终只保留 $x_t$ 与 $x_0$ 的线性组合。

由于推理时 $x_0$ 未知，论文用神经网络 $f_\theta(x_t,y_0,t)$ 预测 $x_0$，并把反向均值参数化为

$$
\mu_\theta(x_t,y_0,t)
=
\frac{\eta_{t-1}}{\eta_t}x_t
+\frac{\alpha_t}{\eta_t}f_\theta(x_t,y_0,t)
$$
方差则直接采用真实后验方差：

$$
\Sigma_\theta(t)
=
\kappa^2\frac{\eta_{t-1}}{\eta_t}\alpha_t\mathbf{I}
$$

训练目标来自负证据下界中的 KL 项：

$$
\min_\theta \sum_t
D_{\mathrm{KL}}
\left[
q(x_{t-1}\mid x_t,x_0,y_0)
\Vert
p_\theta(x_{t-1}\mid x_t,y_0)
\right]
$$
因为两边高斯分布使用相同方差，KL 散度只剩均值之间的二次项。真实均值与模型均值的差为

$$
\mu_q-\mu_\theta
=
\frac{\alpha_t}{\eta_t}\left(x_0-f_\theta(x_t,y_0,t)\right)
$$
又因为

$$
\Sigma_t^{-1}
=
\frac{\eta_t}{\kappa^2\eta_{t-1}\alpha_t}\mathbf{I}
$$
所以

$$
D_{\mathrm{KL}}
\propto
\frac{1}{2}
\left(\frac{\alpha_t}{\eta_t}\right)^2
\frac{\eta_t}{\kappa^2\eta_{t-1}\alpha_t}
\left\|x_0-f_\theta(x_t,y_0,t)\right\|_2^2
$$
整理得到

$$
\mathcal{L}(\theta)
=
\sum_t w_t
\left\|f_\theta(x_t,y_0,t)-x_0\right\|_2^2
$$
其中

$$
w_t=\frac{\alpha_t}{2\kappa^2\eta_t\eta_{t-1}}
$$
论文指出，实际训练时省略权重 $w_t$ 会带来更好的经验效果，因此实现中主要使用未加权的 MSE 形式。

从官方代码角度看，训练阶段先用

$$
z_t=(1-\eta_t)z_0+\eta_t z_y+\kappa\sqrt{\eta_t}\epsilon
$$
构造中间状态。这里 $z_0,z_y$ 可以是图像空间变量，也可以是 VQGAN 的 latent code。网络输出的目标由 `model_mean_type` 决定：

$$
\text{START\_X}: z_0
$$

$$
\text{RESIDUAL}: z_y-z_0
$$

$$
\text{EPSILON}: \epsilon
$$

$$
\text{EPSILON\_SCALE}: \kappa\sqrt{\eta_t}\epsilon
$$
论文主公式采用的是预测 $x_0$ 的参数化形式。若代码采用预测残差或噪声的形式，本质上仍会先把网络输出换算为 $\hat{x}_0$，再代入后验均值。

#### C. 推理过程
推理时只给定 LR 图像 $y_0$，没有真实 HR 图像 $x_0$。ResShift 的起点不是标准高斯噪声，而是以 $y_0$ 为中心的先验分布：

$$
p(x_T\mid y_0)\approx \mathcal{N}(x_T;\ y_0,\ \kappa^2\mathbf{I})
$$
在官方实现中，这一步对应

$$
x_T = y_0+\kappa\sqrt{\eta_T}\epsilon,\quad \epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
由于 $\eta_T\approx 1$，所以该式与 $\mathcal{N}(y_0,\kappa^2\mathbf{I})$ 基本一致。

然后从 $t=T$ 到 $t=1$ 逐步反向采样。每一步先由网络预测

$$
\hat{x}_0=f_\theta(x_t,y_0,t)
$$
再计算后验均值

$$
\mu_\theta(x_t,y_0,t)
=
\frac{\eta_{t-1}}{\eta_t}x_t
+\frac{\alpha_t}{\eta_t}\hat{x}_0
$$
最后采样得到

$$
x_{t-1}
=
\mu_\theta(x_t,y_0,t)
+\mathbf{1}_{t\ne 1}
\sqrt{\kappa^2\frac{\eta_{t-1}}{\eta_t}\alpha_t}\ z,
\quad z\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
其中 $\mathbf{1}_{t\ne 1}$ 表示最后一步不再加入随机噪声。

官方代码中的 `p_sample` 与这个公式是一一对应的。代码先通过 `p_mean_variance` 计算 `out["mean"]` 和 `out["log_variance"]`，然后执行

$$
\text{sample}
=
\text{mean}
+\mathbf{1}_{t\ne 0}
\exp\left(\frac{1}{2}\text{log\_variance}\right)\epsilon
$$
因为

$$
\exp\left(\frac{1}{2}\log\sigma_t^2\right)=\sigma_t
$$
所以这正是高斯重参数化采样。代码中的 `nonzero_mask` 只是在最后一个时间步关闭噪声，保证输出不再被额外扰动。

完整推理流程可以概括为：
1. 输入 LR 图像 $y_0$，必要时先上采样到目标分辨率
2. 若使用 latent 版本，则通过 VQGAN encoder 得到 $z_y$
3. 从 $z_T=z_y+\kappa\sqrt{\eta_T}\epsilon$ 开始采样
4. 在每个时间步用网络预测 $\hat{z}_0$ 或等价的残差、噪声表示
5. 将预测结果换算为 $\hat{z}_0$，并代入后验均值公式
6. 按高斯后验采样 $z_{t-1}$
7. 当 $t=1$ 时不再加噪，最终得到 $\hat{z}_0$，再由 decoder 还原为 SR 图像

### Noise Schedule
ResShift 的效率不仅来自残差平移的马尔可夫链，也来自对 $\eta_t$ 的设计。由边缘分布

$$
q(x_t\mid x_0,y_0)
=
\mathcal{N}\left(x_t;\ x_0+\eta_t(y_0-x_0),\ \kappa^2\eta_t\mathbf{I}\right)
$$
可知，$\sqrt{\eta_t}$ 同时影响残差平移程度和噪声标准差。因此论文直接设计 $\sqrt{\eta_t}$，而不是直接设计 $\eta_t$。

端点被设为

$$
\eta_1=\min\left(\left(\frac{0.04}{\kappa}\right)^2,0.001\right)
$$
和

$$
\eta_T=0.999
$$
前者保证 $q(x_1\mid x_0,y_0)$ 接近 $x_0$，后者保证 $x_T$ 接近以 $y_0$ 为中心的先验。

对中间时间步 $t=2,\cdots,T-1$，论文采用非均匀几何序列：

$$
\sqrt{\eta_t}
=
\sqrt{\eta_1}\ b_0^{\beta_t}
$$
其中

$$
\beta_t
=
\left(\frac{t-1}{T-1}\right)^p(T-1)
$$

$$
b_0
=
\exp\left[
\frac{1}{2(T-1)}
\log\frac{\eta_T}{\eta_1}
\right]
$$
超参数 $p$ 控制 $\sqrt{\eta_t}$ 的增长速度。论文实验显示，$T$ 和 $p$ 会影响保真度与真实感之间的权衡。一般来说，较大的 $p$ 会提高参考指标，如 PSNR、SSIM，但会削弱模型生成细节的能力；较小的 $p$ 更有利于感知质量，但也可能带来与真实 HR 图像的偏离。

超参数 $\kappa$ 控制噪声强度。若 $\kappa$ 过小，采样过程随机性不足，模型更接近确定性恢复；若 $\kappa$ 过大，状态会偏离 LR 附近的有效先验，恢复结果也可能变得平滑。论文在主要实验中采用 $T=15$，$p=0.3$，$\kappa=2.0$。

### Appendix
##### A. 与 SR3 的差异
SR3 的前向过程是

$$
q(y_t\mid y_0)
=
\mathcal{N}\left(y_t;\sqrt{\bar{\alpha}_t}y_0,(1-\bar{\alpha}_t)\mathbf{I}\right)
$$
其终点是标准高斯噪声附近。反向过程的起点也是高斯噪声：

$$
y_T\sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$
因此 SR3 需要较长的采样链来完成从噪声到图像的恢复。

ResShift 的前向过程是

$$
q(x_t\mid x_0,y_0)
=
\mathcal{N}\left(x_t;\ (1-\eta_t)x_0+\eta_t y_0,\ \kappa^2\eta_t\mathbf{I}\right)
$$
其终点不是纯噪声，而是

$$
x_T\approx y_0+\kappa\epsilon
$$
这使反向过程从 LR 图像附近开始。两者的根本差别在于：SR3 学习的是从噪声分布回到 HR 图像分布，而 ResShift 学习的是从 LR 邻域沿残差方向回到 HR 图像。

##### B. 对残差平移机制的理解
如果忽略噪声项，ResShift 的前向过程可以写成

$$
x_t=(1-\eta_t)x_0+\eta_t y_0
$$
这是一条从 $x_0$ 到 $y_0$ 的线段。反向过程则沿相反方向移动：

$$
y_0 \to x_0
$$
因此 ResShift 的“扩散”并不是单纯增加噪声，而是把残差信息逐步移除。反向采样时，网络要补回的主要对象就是

$$
x_0-y_0
$$
也就是 LR 图像缺失的高频细节和退化修正项。

从线性代数角度看，若把图像展平成向量，则 $x_0,y_0,e_0\in\mathbb{R}^{n}$，前向均值

$$
(1-\eta_t)x_0+\eta_t y_0
$$
位于 $x_0$ 和 $y_0$ 张成的仿射直线上。协方差矩阵

$$
\kappa^2\eta_t\mathbf{I}
$$
表示每个维度上加入相同方差的各向同性扰动。因此 ResShift 并没有显式建模像素之间的协方差结构，而是把复杂的图像先验交给神经网络 $f_\theta$ 学习。

##### C. 局限性
论文附录指出，ResShift 在严重退化的漫画图像上仍可能失败。这个现象与训练退化模型有关：多数现实超分辨率方法使用人工构造的合成退化来训练，而真实退化类型更复杂，合成退化不能完全覆盖。因此 ResShift 的主要贡献在扩散链设计与采样效率上，而不是彻底解决真实退化建模问题。
