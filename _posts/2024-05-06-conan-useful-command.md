---
layout: post
title: conan-useful-command
date: 2024-05-06
Author: daiybh
tags: [conan,command, document]
comments: true
toc: true
---

**Conan useful command**


# remote server

## list remote server

    conan remote list

## add remote server

this command will affect "C:\Users\xxx\.conan2\remotes.json"

    conan remote add <your-remote-server-name> http://yoururl:yourport


# **Set the Profile before everything**

this will affect  "C:\Users\xxx\.conan2\profiles\default"

    conan profile detect --force

if want change cppstd to 20  ,open the file change 
    
    compiler.version=20


# create package

## create library

    conan new cmake_lib -d name=<your-package-name> -d version=<your-package-version>

    conan new cmake_lib -d name=logib -d version=1.0

## create APP(Exe)

    conan new cmake_exe -d name=<your-App-name> -d version=<your-App-version>

    conan new cmake_exe -d name=TestApp -d version=1.0


## list local packages

    conan list "*"

### list more detail for package

* list the package from remote 

    conan list loglib/1.5:* -r=nas

* list the package from local

    conan list loglib/1.5:*


Result
```
nas
  loglib
    loglib/1.5
      revisions
        04f48a810f38e640feabe48b1938970b (2024-05-06 06:52:36 UTC)
          packages
            f923b2725de93dc70a9c66b3fa3ab195f893064c
              info
                settings
                  arch: x86_64
                  build_type: Debug
                  compiler: msvc
                  compiler.cppstd: 20
                  compiler.runtime: dynamic
                  compiler.runtime_type: Debug
                  compiler.version: 193
                  os: Windows
                options
                  shared: False
                  unicode: True
```

## search packages from remote

    conan search "loglib"

    conan search "loglib/1.0*"


# upload package to remote

    conan upload loglib/1.3 -r=<your-remote-server-name>

