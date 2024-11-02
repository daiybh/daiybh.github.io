---
layout: post
title: how to 'proxy_pass' in nginx
categories: [nginx,proxy_pass]
description: how to 'proxy_pass' in nginx
keywords: nginx,proxy_pass
tags: [nginx,proxy_pass]
---

we have a http://aaa.com domain.

1. want  http://aaa.com/server1 to http://127.0.0.1:8080

2. want  http://aaa.com/server2 to http://127.0.0.1:8081


but how to do it?

<!--more-->
# nginx 配置

```
server {
    listen 80;
    server_name aaa.com;

    location /server1/ {
        rewrite ^/server1/(.*)$ /$1 break;
        proxy_pass http://127.0.0.1:8080;

        # 使用 proxy_redirect 将后端返回的重定向路径前加上 /server1
        proxy_redirect / /server1/;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /server2/ {
        rewrite ^/server2/(.*)$ /$1 break;
        proxy_pass http://127.0.0.1:8081;

        # 使用 proxy_redirect 将后端返回的重定向路径前加上 /server2
        proxy_redirect / /server2/;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
# rewrite ^/server1/(.*)$ /$1 break;

我们是通过/server1/ /server2/来区分请求的，
所以需要去掉 /server1/ /server2/  把剩下的path 转发到后端服务器，

http://bbb.com/server1/aa?a=1&b=2  ->  http://127.0.0.1:8080/aa?a=1&b=2

#  proxy_redirect / /server1/;

由于后端返回的是相对路径 URL /xxx/1.html  

组合上原HOST 后 成为 http://bbb.com/xxx/1.html  
但是 nginx 中 没有 xxx这个location  

真实的 url 应该是  http://bbb.com/server1/xxx/1.html

所以需要用 proxy_redirect 来修改 HTTP 头中的 Location 重定向路径。
proxy_redirect 可以将后端返回的相对路径重写为带有 /server1 前缀的路径。

http://127.0.0.1:8080/xxx/1.html  -> http://bbb.com/server1/xxx/1.html