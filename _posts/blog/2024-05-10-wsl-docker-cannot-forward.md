---
layout: post
title: 外部机器访问wsl内的docker 服务
date: 2024-05-10
Author: daiybh
tags: [wsl,mirrored,docker]
comments: true
toc: true
---

** 外部机器访问wsl内的docker 服务**

可以使用 NAT 模式，然后设置端口转发实现

但是这种模式需要频繁的操作，并且wsl IP 不好获取，

所以强烈建议使用 **networkingMode=mirrored** 模式 [官方配置](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#wslconfig)


# 问题

但是这种模式设置后 会发现 ，本机windows 不能访问这个docker服务 

比如使用docker 创建nginx服务在8000端口上，

想在windows上访问 localhost:8000 或者 windowsIP(192.168.0.23):8000 ----> 都会失败

但是在另外机器上通过浏览器访问 192.168.0.23:8000  ---> 就可以通达


# 解决办法

这个问题原始讨论在 [github issues](https://github.com/microsoft/WSL/issues/10494) 


 [解决方案是](https://github.com/microsoft/WSL/issues/10494#issuecomment-1754170770)

添加下面配置到wslconfig


    [experimental]
    autoMemoryReclaim=gradual
    networkingMode=mirrored
    dnsTunneling=true
    firewall=false
    autoProxy=false
    hostAddressLoopback=true


并且添加 
    
    "iptables": false 

到 /etc/docker/daemon.json 就好了
如果是使用snap 安装的 配置文件在  /var/snap/docker/current/config/daemon.json


