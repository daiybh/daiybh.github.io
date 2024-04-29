---
layout: post
title: 在powershell 中自动设置vs编译环境
date: 2019-07-24
Author: daiybh
tags: [sample, document]
comments: true
---


# check if already have  the PROFILE

```Powershell
PS> $PROFILE
C:\Users\user1\Documents\PowerShell\Microsoft.PowerShell_profile.ps1
PS> $PROFILE.AllUsersAllHosts
C:\Program Files\PowerShell\7\profile.ps1
```

# create the profile [link](https://learn.microsoft.com/zh-cn/powershell/scripting/learn/shell/creating-profiles?view=powershell-7.4#how-to-create-your-personal-profile)

```
    if (!(Test-Path -Path $PROFILE)) {
  New-Item -ItemType File -Path $PROFILE -Force
}
```


open the $PROFILE with notepad/code/...

put those code into the file . then open a new powershell , will take effect.

```

function setVSDevShell {
    
$vspath="C:\Program Files\Microsoft Visual Studio\2022\Community"
Import-Module ("$vspath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll")


Enter-VsDevShell -VsInstallPath "$vspath" -SkipAutomaticLocation

    
}

setVSDevShell;
```