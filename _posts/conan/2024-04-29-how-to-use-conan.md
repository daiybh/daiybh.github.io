---
layout: post
title: how to use conan in cpp
categories: Conan
keywords: conan, document
tags: [conan, document]
description: how to use conan in cpp
---

this page is show "how to use conan in cpp "

<!--more-->
## Introduce

simply introduce for conan in cpp.


testPorject [link](https://github.com/daiybh/conan-usage/)


## Install Conan-server  and start

the official  [link](https://docs.conan.io/2/reference/conan_server.html) 

if need private packages for team or company  can do this .

    pip install conan-server

### Configuration

default server configuration is ~/.conan_server/server.conf


#### give write permissions

line:62  remove the "#"  

let 
```
[write_permissions]

*/*@*/*: *
```

#### change the user name

the default user is demo, so need change it to others

[users]
demo:demo

## Install client

the official [link](https://docs.conan.io/2/installation.html)

you need 

* python>=3.6
* pip / pip3

    pip3 install conan

## **Set the Profile before everything**

this will affect  "~/.conan2/profiles/default"

    conan profile detect --force


set cppstd=20

open the file change 

in windows

    compiler.cppstd=20

in linux

    compiler.cppstd=gnu20



## Create "mypkg" package

    mkdir mypkg && cd mypkg

    conan new cmake_lib -d name=mypkg -d version=1.0

    conan create . -s build_type=Debug

    conan create . -s build_type=Release



## upload "mypkg" package to conan-server

check if already add local conan-server


    conan remote list


if no:

//conan remote add "<name_for later use>" "<remote conan Server URL>"

    conan remote add my_local_server http://localhost:9300



    conan search "mypkg" -r=my_local_server

will be failed "ERROR: Recipe 'mypkg' not found"

    conan upload mypkg/1.0 -r=my_local_server

search again

    conan search "mypkg" -r=my_local_server

sucess.


## clean local Cache

remove local cache

    conan remove "mypkg" --confirm
```
conan search "mypkg"


Found 1 pkg/version recipes matching mypkg in my_local_server
conancenter
  ERROR: Recipe 'mypkg' not found
my_local_server
  mypkg
    mypkg/1.0
```	
	

## use "mypkg" from local "Conan-server"	

create app

    conan new cmake_exe -d name=testApp -d version=1.0

goto the code folder

install the library 

* --build_type=Debug for debug and release 

* --output-folder=build the output folder is "build"

* --build=missing  if there don't have the library from conanserver then can build

```
    conan install . --output-folder=build --build=missing -s build_type=Debug

    conan install . --output-folder=build --build=missing -s build_type=Release
```

* if in windows

```    
    cmake --preset conan-default
```

* if in linux

```
    cmake --preset conan-release
    cmake --preset conan-debug
```

compile it for debug and release

    cmake --build --preset="conan-release"
    
    cmake --build --preset="conan-debug"

