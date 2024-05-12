---
layout: post
title: 使用 docker-jekyll 创建pages
categories: jekyll
tags: [jekyll,docker]
keywords: jekyll,docker,pages
comments: true
toc: true
---

使用 docker-jekyll 创建pages

<!--more-->
只能使用 **jekyll:3.8**  

# 创建blog


    export site_name="my-blog" && export MSYS_NO_PATHCONV=1
    docker run --rm   --volume="$PWD:/srv/jekyll"   -it jekyll/jekyll:3.8 \
  sh -c "chown -R jekyll /usr/gem/ && jekyll new $site_name" \
  && cd $site_name

# build

    export JEKYLL_VERSION=3.8
    docker run --rm --volume="$PWD:/srv/jekyll:Z" -it jekyll/builder:3.8 jekyll build

创建一个自动编译的

    docker run --rm   --volume="$PWD:/srv/jekyll:Z"   -it jekyll/builder:3.8   jekyll build --incremental --watch

# 运行服务
    
    docker run --rm   --volume="$PWD:/srv/jekyll:Z"   --publish 10086:4000   jekyll/jekyll:3.8 jekyll serve




可以参考完整的教程 [link](/2019/12/14/How-to-build-a-blog-with-github-pages/)