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

本系列笔记从SDE出发，经过DDPM反向推导、ODE概率流、连续正则化流到流匹配条件路径，最终揭示了扩散模型与连续归一化流之间的深层统一关系。

$$s_\theta \approx \nabla_x\log p_t(x)$$
$$\frac{dx}{dt}=f_t(x)-\frac{1}{2}g_t^2s_\theta(x,t)$$
所以 Diffusion 可以被解释为一个以 score 间接定义速度场的 CNF 
Diffusion：先学 score，再得到 ODE
CNF：直接学 ODE 速度场
总的体系可以用以下这幅图描述
