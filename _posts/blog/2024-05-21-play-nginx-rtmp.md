---
layout: post
title: 使用 Nginx-RTMP 搭建简单的流媒体服务器
categories: blog
keywords: nginx, rtmp
tags: [nginx, rtmp]
description: how to use conan in cpp
---

在linux 下快速搭建rtmp流媒体服务器

详细的参考 ：

[1](https://cloud.tencent.com/developer/article/2212849)
    
[2](https://cloud.tencent.com/developer/article/1451824?from=15425)
<!--more-->

# 流媒体服务相关配置

配置在 /etc/nginx/nginx.conf 

1、打开nginx配置文件(nginx.conf)，在末尾添加如下代码：

```
rtmp {
    server {
        listen 1935;
        application live {
            live on;
        }
    }
}
```

2、在nginx配置文件(nginx.conf)，在server中添加如下代码：

```
        # rtmp stat
        location /stat {
            rtmp_stat all;
            rtmp_stat_stylesheet stat.xsl;
        }
        location /stat.xsl {
            # you can move stat.xsl to a different location
            root /usr/build/nginx-rtmp-module;
        }

        # rtmp control
        location /control {
            rtmp_control all;
        }

```

3、配置完成后，检查配置是否正确：nginx -t 。

4、配置没有问题重启nginx：nginx -s reload

# 检查RTMP服务是否生效

1、在浏览器中输入:http://+服务器ip+端口+stat                         

此端口是 http server 端口 ，默认是80

例如：http://10.134.64.142:80/stat

2、浏览器中出现下图，则表示rtmp服务生效了。

![img](https://ask.qcloudimg.com/http-save/yehe-5521279/1mcrpjtpa0.jpeg)

# 向RTMP服务推流

1、在这里我需要借助ffmpeg进行推流，ffmpeg安装在这里不进行赘述。

2、推流服务器地址格式如下：

rtmp://+ip+":"端口+/live/+"其他"     //其他是我们任意起的名字

此端口是 nginx rtmp里面配置的端口，
例如：

rtmp://10.134.64.142:1935/live/selftest

3、输入下面命令想服务器推流：

ffmpeg -re -i +视频路径 -c copy -f flv +推流服务器地址


    ffmpeg -f gdigrab -framerate 30 -i desktop -c:v h264 -qp 0 -acodec aac -f flv rtmp://10.134.64.142:1935/live/abcd

4、浏览器的页面中的live streams出现如下则表示推流成功。

![img](https://ask.qcloudimg.com/http-save/yehe-5521279/f8fwgs0hqv.jpeg)

# VLC收看直播流

1、开发VLC media player软件，并打开媒体选项中的开发网络串流，输入刚才的推流的地址：

rtmp://10.134.64.142:1935/live/selftest

如果视频可以正常播放则说明整个流程没有问题了。