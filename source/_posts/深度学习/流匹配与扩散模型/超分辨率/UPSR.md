---
title: UPSR
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
LR图像中的固有信息被忽略，即：**平坦区域已经接近目标，而边缘和纹理区域则距离较远**。而以往的方法并没有关注到这个问题，导致整个图像区域的噪声采样各向同性化，导致计算资源被占用，进而导致采样过程缓慢。所以 UPSR 提出噪声各项异性的扩散过程。其中平坦区域被分配较低的噪声水平（当 $t \to 0$ 时），而边缘和纹理区域则接收较大的噪声（当 $t \to T$ 时）。为了实现区域性的噪声控制，引入不确定性来作为重要指标。

### Methodology
论文从不同的整体噪声水平应用于基于扩散的 SR 模型时，输出的保真度（fidelity）和感知质量（perceptual quality）如何变化开始观察到，随着每个像素对残差$|y-x|$增加，感知质量差距迅速增加而保真度差距几乎不变。这意味着在图像的高频细节区域，噪声的强度对图像质量的影响更大。那怎么样构建这种机制呢？论文使用了不确定性估计来构建

符号：HR 图像 $x^i$ ，其对应的 LR 图像 $y^i$ ，SR 估计 $g(y^i)$ ，不确定性 $\psi^i$ 
论文认为最终的 SR 图像要由两部分组成：SR 项和不确定性项

$$
x^i=g(y^i)+\epsilon\psi(g(y^i))
$$
其中 $\epsilon$ 代表高斯分布或者拉普拉斯分布
类似地，可以将 $y^i$ 的不确定性与残差 $|x^i - y^i|$ 联系起来

$$x^i = y^i + \hat{\epsilon} \psi(y^i)$$
其中 $\hat{\epsilon}$ 代表取决于退化模式的未知分布。如果 $g(\cdot)$ 经过了良好的训练，假设 $g(y^i)$ 非常接近 $x^i$，这意味着 $|g(y^i) - y^i|$ 类似于 $|x^i - y^i|$。因此，论文利用残差 $|g(y^i) - y^i|$ 作为 $y^i$ 不确定性的估计。具体来说，将不确定性估计定义为
$$\psi_{est}(y) = \frac{1}{2} |g(y) - y|$$
经过前面的推导积累，论文提出用适应图像内容的各向异性噪声来替换常用的各向同性高斯噪声。在获得不确定性估计后，我们计算权重系数 $w_u$，应用它来调制扩散过程中不同区域的噪声水平。论文称这种策略为 UNW，然后将噪声加权系数 $w_u$ 建模为关于不确定性的单调递增函数
$$w_u(y) := u(\psi_{est}(y))$$
这个方法将ResShift的前向扩散分布
$$q(x_t \mid x_{t-1}, x_0, y_0) = \mathcal{N}(x_t \mid x_{t-1} + \alpha_t(y_0 - x_0), \kappa^2 \alpha_t \mathbf{I})$$
和后向过程分布
$$q(x_{t-1} \mid x_t, x_0, y_0) = \mathcal{N}\left(x_{t-1} \mid \frac{\eta_{t-1}}{\eta_t} x_t + \frac{\alpha_t}{\eta_t} x_0, \kappa^2 \frac{\eta_{t-1}}{\eta_t} \alpha_t \mathbf{I}\right)$$
分别改写为
$$\mathcal{N}\left(x_t \mid x_{t-1} + \alpha_t(y_0 - x_0), \kappa^2 w_u(y_0)^2 \alpha_t \mathbf{I}\right)$$
和
$$\mathcal{N}\left(x_{t-1} \mid \frac{\eta_{t-1}}{\eta_t} x_t + \frac{\alpha_t}{\eta_t} x_0, \kappa^2 w_u(y_0)^2 \frac{\eta_{t-1}}{\eta_t} \alpha_t \mathbf{I}\right)$$
如图所示，给定一张 LR 图像 $y_0$，我们首先通过辅助 SR 网络 $g(\cdot)$ 获得其 SR 估计 $g(y_0)$。然后我们将 $y_0$ 的不确定性估计为 $\psi_{est}(y_0) = \frac{1}{2}|g(y_0) - y_0|$，并获得权重系数为 $w_u(y_0) = u(\psi_{est}(y_0))$
![[Pasted image 20260418195519.png]]
并把损失函数改写为
$$\mathcal{L}(\theta) = \sum_t \left[ \| f_\theta(x_t, y_0, g(y_0), t) - x_0 \|_2^2 + \lambda L_{per}(f_\theta(x_t, y_0, g(y_0), t), x_0) \right]$$
其中 $\lambda$ 是控制保真度和感知质量之间权衡的超参数
### UPSR 的训练过程
![[Pasted image 20260418200617.png]]
### Appendix
##### $u(·)$的设计
论文将区域特定扰动的权重系数与不确定性估计之间的关系建模为一个单调递增函数 $u'(\cdot)$，随后进行对角化处理
$$w_u(\boldsymbol{y}_0) = u(\boldsymbol{\psi}_{est}(\boldsymbol{y}_0)) = \text{diag}(u'(\boldsymbol{\psi}_{est}(\boldsymbol{y}_0)))$$
然后 $u'(·)$ 就被构造为
$$u'(\psi) = 
\begin{cases} 
\frac{(1 - b_u)}{\psi_{max}} \psi + b_u & \text{if } 0 \le \psi \le \psi_{max} \\
1 & \text{otherwise}
\end{cases}
$$
这采用了分段函数的形式
对于不确定性估计 $\psi_{est}(y^i_0) \in [0, \psi_{max}]$ 的区域，我们将 $u'(\cdot)$ 定义为一个具有偏移量 $b_u$ 和斜率 $(1 - b_u)/\psi_{max}$ 的线性函数，确保输出保持在 $[b_u, 1]$ 范围内。同时正偏移量 $b_u$ 确保了最低的噪声水平，防止边缘和纹理区域由于偶尔不准确的不确定性估计而被分配极低的噪声水平
![[Pasted image 20260428222639.png]]
