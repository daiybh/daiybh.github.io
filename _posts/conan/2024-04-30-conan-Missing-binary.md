---
layout: post
title: conan-missing-binary
date: 2024-04-30
categories: Conan
Author: daiybh
tags: [conan, document]
keywords: conan, document
comments: true
---

**Conan Missing binary**


<!--more-->


if call 
    
    conan create . -s build_type=Debug

show those message.

```
myloglib/1.0: Forced build from source
Requirements
    fmt/10.0.0#87ee19b1d95cc0f447dfa7a9fbb78444:242c2889ffbb34b5cb5b8e66f6891e9db8cb71fe - Missing
    myloglib/1.0#e1d698e5d9a2a89f4195055b03cd18b4:04f7ccdd16a26e5ae5e162d1fbf39d85c02bca6d - Build
    spdlog/1.11.0#af34b1d6b407cb6ba415ad82a3d6ba36:860ca7a85db0a8544bf79620db578ef8fc676d55 - Missing
ERROR: Missing binary: fmt/10.0.0:242c2889ffbb34b5cb5b8e66f6891e9db8cb71fe
ERROR: Missing binary: spdlog/1.11.0:860ca7a85db0a8544bf79620db578ef8fc676d55

fmt/10.0.0: WARN: Can't find a 'fmt/10.0.0' package binary '242c2889ffbb34b5cb5b8e66f6891e9db8cb71fe' for the configuration:
[settings]
arch=x86_64
build_type=Debug
compiler=msvc
compiler.cppstd=14
compiler.runtime=dynamic
compiler.runtime_type=Debug
compiler.version=193
os=Windows
[options]
header_only=False
shared=False
with_os_api=True
```

it mean can not found the correct recipe on remote server

it need  

spdlog/1.11.0#af34b1d6b407cb6ba415ad82a3d6ba36:860ca7a85db0a8544bf79620db578ef8fc676d55

https://docs.conan.io/2/knowledge/faq.html#error-missing-prebuilt-package

we can  build from the source ,then push the recipe to remote

    conan install . --output-folder=build --build=missing -s build_type=Debug
    conan install . --output-folder=build --build=missing -s build_type=Release


then check if the spdlog in local cache

     conan list "spdlog:*"

```
Found 2 pkg/version recipes matching spdlog in local cache
Local Cache
  spdlog
    spdlog/1.11.0
      revisions
        af34b1d6b407cb6ba415ad82a3d6ba36 (2024-03-01 19:57:03 UTC)
          packages
            860ca7a85db0a8544bf79620db578ef8fc676d55
              info
                settings
                  arch: x86_64
                  build_type: Debug
                  compiler: msvc
                  compiler.cppstd: 14
                  compiler.runtime: dynamic
                  compiler.runtime_type: Debug
                  compiler.version: 193
                  os: Windows
                options
                  header_only: False
                  no_exceptions: False
                  shared: False
                  use_std_fmt: False
                  wchar_filenames: False
                  wchar_support: False
                requires
                  fmt/10.0.Z
    spdlog/1.14.0
      revisions
        9bddb6b2c7819bfde7784714af1f27bd (2024-04-26 00:29:33 UTC)
          packages
            b2428e3396b902156afef679f479469cb9e15b99
              info
                settings
                  arch: x86_64
                  build_type: Release
                  compiler: msvc
                  compiler.cppstd: 14
                  compiler.runtime: dynamic
                  compiler.runtime_type: Release
                  compiler.version: 193
                  os: Windows
                options
                  header_only: False
                  no_exceptions: False
                  shared: False
                  use_std_fmt: False
                  wchar_filenames: False
                  wchar_support: False
                requires
                  fmt/10.2.Z
```

