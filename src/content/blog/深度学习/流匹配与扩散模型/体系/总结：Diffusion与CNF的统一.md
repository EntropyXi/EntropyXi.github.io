---
title: "总结：Diffusion与CNF的统一"
description: "对扩散模型与连续正则化流进行统一总结，梳理SDE、ODE、CNF与流匹配之间的等价关系与转换路径，形成完整的生成模型理论体系。"
date: "2026-05-17T14:00:00+08:00"
updated: "2026-05-17T14:00:00+08:00"
tags:
  - "深度学习"
  - "流匹配与扩散模型"
categories:
  - "深度学习"
  - "流匹配与扩散模型"
permalink: "2026/05/17/深度学习/流匹配与扩散模型/体系/总结：Diffusion与CNF的统一"
math: true
draft: false
---

本系列笔记从 SDE 出发，经过 DDPM 反向推导、ODE 概率流、连续正则化流（CNF）到流匹配条件路径，最终揭示了扩散模型与连续归一化流之间的深层统一关系：

$$
s_\theta(x, t) \approx \nabla_x\log p_t(x)
$$

$$
\frac{dx}{dt} = f_t(x) - \frac{1}{2}g_t^2 s_\theta(x, t)
$$

所以 Diffusion 可以被解释为一个以 score 间接定义速度场的连续归一化流（CNF）：
- **Diffusion**：先学习分布的 score $\nabla_x \log p_t(x)$，再导出概率流 ODE；
- **CNF / Flow Matching**：直接在条件路径上回归 ODE 的速度场 $v_\theta(x, t)$。

从生成模型的全局视角来看，二者本质上都是在求解一个将简单先验分布连续推运（pushforward）至复杂真实数据分布的动力学系统。
