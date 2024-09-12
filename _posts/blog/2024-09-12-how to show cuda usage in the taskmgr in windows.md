---
layout: post
title: how to show cuda usage in the taskmgr in windows
categories: [cuda,windows]
description: how to show cuda usage in the taskmgr in windows
keywords: cuda,windows
tags: [cuda,windows]
---

sometime  we can not see the cuda usage in the taskmgr

<!--more-->

## we can disable the  "accelerated GPU scheduling"

Have had this problem for a while, once had CUDA in the Task Manager -> GPU 0 -> Graph dropdown menus, then disappeared. 

The solution for me is to go to 
Settings -> System -> Display -> 
Graphics settings (down the bottom of page) -> 
Hardware-accelerated GPU scheduling -> **Switch to "Off"**. 

系统->屏幕设置->显示->显示设置->硬件加速GPU计划

系统->屏幕->显示卡->默认图形设置->硬件加速GPU计划

After reboot the CUDA option had returned to the graphs in the Task Manager GPU window.


document in msdn: https://docs.microsoft.com/en-us/windows-hardware/drivers/display/accelerated-gpu-scheduling