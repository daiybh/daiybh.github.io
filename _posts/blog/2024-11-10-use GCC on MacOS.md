---
layout: post
title: Use Gcc on Macos
categories: [macos]
description: Use Gcc on Macos
---

# Use Gcc on Macos

strong suggest  install gcc with **macport** .

use  HomeBrew maybe  miss some headers.




<!--more-->

## first need install macport

## also need  xcode 

    xcode-select --install

## install gcc
```shell

sudo port install gcc@13

```

## select gcc as default compiler on macos

```shell
sudo port select --set gcc mp-gcc13
```


now you can check if the gcc is your.

```shell
gcc --version

which gcc
```

# for CMAKE

also need set CC and CXX  that Cmake need them to detect .

1. set CC and CXX in your shell env.
```shell
    export CC=/opt/local/bin/gcc-mp-13
    export CXX=/opt/local/bin/g++-mp-13
    export CMAKE_C_COMPILER=/opt/local/bin/gcc-mp-13
    export CMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-13
```

2. set CC and CXX in your CMakeLists.txt

```shell
set(CMAKE_C_COMPILER /opt/local/bin/gcc-mp-13)
set(CMAKE_CXX_COMPILER /opt/local/bin/g++-mp-13)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -stdlib=libc++")
```