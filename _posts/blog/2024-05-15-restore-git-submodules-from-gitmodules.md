---
layout: post
title: restore git submodules from gitmodules
date: 2024-05-10
Author: daiybh
tags: [git,gitmodules]
comments: true
toc: true
---

restore-git-submodules-from-gitmodules

<!--more-->
https://stackoverflow.com/questions/11258737/restore-git-submodules-from-gitmodules

```
git config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
    while read path_key path
    do
        url_key=$(echo $path_key | sed 's/\.path/.url/')
        url=$(git config -f .gitmodules --get "$url_key")
        git submodule add $url $path
    done

```