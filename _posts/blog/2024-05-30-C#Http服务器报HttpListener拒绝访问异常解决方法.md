---
layout: post
title: C#Http服务器报HttpListener拒绝访问异常解决方法
categories: [blog]
description: 在使用C#的System.Net.HttpListener进行客户端推送消息监听时，出现System.Net.HttpListenerException:“拒绝访问” 问题，现将解决方案记录一下，给遇到相同问题的朋友提供参考
keywords: C#, http,拒绝访问
---

# 问题描述： 

在使用C#的System.Net.HttpListener进行客户端推送消息监听时，出现System.Net.HttpListenerException:“拒绝访问” 问题，现将解决方案记录一下，给遇到相同问题的朋友提供参考。
<!--more-->
# 具体代码：

```
httpobj = new System.Net.HttpListener();
//定义url及端口号，通常设置为配置文件
httpobj.Prefixes.Add("http://+:8080/");
//启动监听器
httpobj.Start();
```

# 解决方案如下：

1、以管理员权限打开CMD命令行

2、先删除可能存在的错误urlacl，这里的*号代指localhost、127.0.0.1、192.168.199.X本地地址和+号等

    命令：netsh http delete urlacl url=http://+:8080/ 

3、将上面删除的*号地址重新加进url，user选择所有人

    命令：netsh http add urlacl url=http://+:8080/  user=Everyone

4、CMD配置防火墙

    netsh advfirewall firewall Add rule name=\"命令行Web访问8080\" dir=in protocol=tcp localport=8080 action=allow

5、通过输入命令：```netsh http show urlacl``` 查看新配置urlacl是否配置成功

经过如上设置服务端就可以以httpListener.Prefixes.Add(“http://+:8080/”);监听地址开启监听。客户端可以通过访问服务端8080端口。服务端本机也可以在浏览器中以localhost和127.0.0.1访问自身http服务器。