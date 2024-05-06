---
layout: post
title: conan-useful-command
date: 2024-05-06
Author: daiybh
tags: [conan,command, document]
comments: true
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

## search packages from remote

    conan search "loglib"

    conan search "loglib/1.0*"


# upload package to remote

    conan upload loglib/1.3 -r=<your-remote-server-name>

