---
title: 总结：Diffusion与CNF的统一
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

$$s_\theta \approx \nabla_x\log p_t(x)$$
$$\frac{dx}{dt}=f_t(x)-\frac{1}{2}g_t^2s_\theta(x,t)$$
所以 Diffusion 可以被解释为一个以 score 间接定义速度场的 CNF 
Diffusion：先学 score，再得到 ODE
CNF：直接学 ODE 速度场
总的体系可以用以下这幅图描述
![[ff5525f4867004143f1ce88aca722524 1.jpg]]
