---
layout: post
title: docker usage
date: 2024-05-06
Author: daiybh
tags: [docker, document]
comments: true
toc: true
---

**docker usage**

<!--more-->


# build docker

在当前目录编译 docker

    docker build -t mydocker:version .
    docker build -t mydocker:v1 .

# run docker [link](https://yeasy.gitbook.io/docker_practice/container/run)

    所需要的命令主要为 docker run。

例如，下面的命令输出一个 “Hello World”，之后终止容器。


    docker run ubuntu:18.04 /bin/echo 'Hello world'
    Hello world

这跟在本地直接执行 /bin/echo 'hello world' 几乎感觉不出任何区别。

下面的命令则启动一个 bash 终端，允许用户进行交互。


    docker run -t -i ubuntu:18.04 /bin/bash
    root@af8bae53bdd3:/#