---
layout: post
title: Cuda Warp Divergence
date: 2024-06-09
categories: Cuda
author: daiybh
tags: cuda,Warp
comments: true
---

在 CUDA 内核函数中使用 if 语句会影响效率，尤其是在条件分支较多且线程分布不均匀的情况下。

CUDA 采用SIMT（Single Instruction, Multiple Threads）模型，在一个warp（32个线程）内的**所有线程必须同步执行同一条指令**。

如果warp中的不同线程需要执行不同的指令（即分支），则这些线程会按顺序执行每个分支的代码，这种现象称为warp divergence。

<!--more-->

# Warp Divergence

如果warp中的线程执行不同的路径，CUDA硬件会依次执行每条路径，并对不需要执行的线程进行屏蔽，直到所有路径都执行完毕。这样会导致性能下降。

# 最小化 Warp Divergence

重构代码：尽量将分支逻辑移出并行代码，或重构算法以减少分支。
统一分支条件：如果可能，确保一个warp中的所有线程尽量执行相同的分支。
使用简单条件：简化条件判断，减少分支数量。