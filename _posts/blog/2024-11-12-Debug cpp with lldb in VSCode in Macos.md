---
layout: post
title: Debug cpp with lldb in VSCode in MacOS
categories: [macos,lldb]
description: Debug cpp with lldb in VSCode in MacOS
---

we want debug the cpp with lldb/gdb in VSCode in MacOS


but gdb is not work in macos(M1).(maybe yes, but I don't found )

so we use lldb.

<!--more-->

but if we direclty use lldb in vscode, it will not work. 

It always report 

```
Starting: "/usr/bin/lldb" --interpreter=mi
error: unknown option: --interpreter=mi
```


Fininally, I found the document that we can use lldb in vscode.

https://code.visualstudio.com/docs/cpp/lldb-mi

it recommend to use lldb-mi.

lldb-mi is in "~/.vscode/extensions/ms-vscode.cpptools-<version>/debugAdapters/lldb-mi/bin"

better is copy it to /usr/local/bin or some other path in your $PATH

that can short path in the launch.json.


those is the full launch.json

```json

{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug with LLDB",
            "type": "cppdbg",
            "request": "launch",
            // Resolved by CMake Tools:
            "program": "${command:cmake.launchTargetPath}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [
                {
                    // add the directory where our target was built to the PATHs
                    // it gets resolved by CMake Tools:
                    "name": "PATH",
                    "value": "$PATH:${command:cmake.launchTargetDirectory}"
                },
                {
                    "name": "OTHER_VALUE",
                    "value": "Something something"
                }
            ],
            "externalConsole": false,
            "MIMode": "lldb",
            "miDebuggerPath": "/opt/local/bin/lldb-mi",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for lldb",
                    "text": "settings set target.macosx-version-min 10.15",  // Optional: Set macOS version
                    "ignoreFailures": true
                }
            ],
            "logging":{
                "engineLogging": true
            }
            
        }
    ]
}

```



this launch.json can launch current project in debug mode.