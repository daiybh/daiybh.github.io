---
layout: post
title: use systemctl in linux
date: 2024-05-09
Author: daiybh
tags: [linux,systemctl]
comments: true
toc: true
---

# use systemctl in linux

<!--more-->




## 新建  /etc/systemd/system/mystart.service

填入以下内容：                                                                                                            

```
[Unit]                                                                                                                    
Description=Led Python API                                                                                                
After=network.target                                                                                                      
                                                                                                                          
[Service]                                                                                                                 
Type=simple                                                                                                               
Restart=always                                                                                                            
ExecStart=python /root/mycode.py                                             
                                                                                                                          
[Install]                                                                                                                 
WantedBy=multi-user.target       

```

## 添加或者修改配置文件（.service) 需要重新加载
    
    systemctl daemon-reload

# 使用systemctl 管理服务

## 启动服务
    
    systemctl start mystart.service

## 查看服务
    
    systemctl status mystart.service

## 查看服务输出
    
    journalctrl -u mystart -e

## 设置开机启动

    systemctl enable mystart.service


