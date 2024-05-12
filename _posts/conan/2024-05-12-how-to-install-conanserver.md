---
layout: post
title: how-to-install-conanserver
date: 2024-05-12
categories: Conan
author: daiybh
tags: [conan,conan-server, document]
keywords: conan, conan-server
comments: true
---

this document show how to install and configuration conan-server ,include use docker.

<!--more-->

# install

## direct use conan-server

the official  [link](https://docs.conan.io/2/reference/conan_server.html) 

if need private packages for team or company  can do this .

    pip install conan-server

then just run this command 

    conan_server  

or run in backend. 

    nohup conan_server &

or run with systemctl follow this [document](2024/05/09/linux-systemctl/)


## run in docker

also we can run conan_server in docker 

pull the images from docker hub

  docker pull conanio/conan_server

run docker

    docker run -v ${PWD}:/root/.conan_server -p 9300:9300 conanio/conan_server


* mount currectfolder to "/root/.conan_server"

  so in currectfolder  will have  server.conf  and data

* maps the port  hostport:containerport 9300 to 9300


so goto [Configuration](#configuration)


**remeber after configuration ,need restart the server  or the container.**

# Configuration

default server configuration is ~/.conan_server/server.conf

the folder struct is

```
.
├── data  "all recipe will be here"
├── server.conf
└── version.txt
```

## give write permissions

line:62  remove the "#"  

let 
```
[write_permissions]

*/*@*/*: *
```

## change the user name

the default user is demo, so need change it to others

```
[users]
demo:demo
```