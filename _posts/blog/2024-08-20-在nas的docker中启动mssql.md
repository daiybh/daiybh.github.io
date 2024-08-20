---
layout: post
title: 在nas的docker中启动mssql
categories: [docker, mssql]
description: nas docker mssql
keywords: docker, mssql
tags: [nas, docker, mssql]
---

在nas的docker中启动mssql 

如果需要数据持久化

需要注意权限问题 

不然 加上 -v   就会启动失败

```
/opt/mssql/bin/permissions_check.sh: line 4: [: : integer expression expected
/opt/mssql/bin/permissions_check.sh: line 59: [: : integer expression expected
SQL Server 2019 will run as non-root by default.
This container is running as user mssql.
To learn more visit https://go.microsoft.com/fwlink/?linkid=2099216.
/opt/mssql/bin/sqlservr: Error: The system directory [/.system] could not be created. File: LinuxDirectory.cpp:420 [Status: 0xC0000022 Access Denied errno = 0xD(13) Permission denied]
```


# 解决方法

参考文章 https://learn.microsoft.com/en-us/sql/linux/sql-server-linux-docker-container-security?view=sql-server-ver16

加上 ```-u 0:0```

```
docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=MyStrongPassword" -u 0:0 -p 1433:1433 -v /mssql:/var/opt/mssql -d mcr.microsoft.com/mssql/server:2019-latest
```
