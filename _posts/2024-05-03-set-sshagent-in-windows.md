---
layout: post
title: set sshagent in windows
date: 2024-05-03
Author: daiybh
tags: [ssh-agent, document]
comments: true
---

**set sshagent in windows**

Always follow this page with success. [page](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/working-with-ssh-key-passphrases)

## Enable the ssh-agent service on your Windows 10 box.

1. Start-> Type 'Services' and click on the Services App that appears.
2. Find the OpenSSH Authentication Agent service in the list.
3. Right-click on the OpenSSH Authentication Agent service, and choose 'Properties'.
4. Change the Startup type: to Automatic.
5. Click the Start button to change the service status to Running.
6. Dismiss the dialog by clicking OK, and close the Services app.

## Add your key to the ssh-agent

1. Open your shell of preference (I'll use Windows Powershell in this example, applies to Powershell Core too).
2. Add your SSH key to the ssh-agent: ssh-add (you can add the path to your key as the first argument if it differs from the default).
3. Enter your passphrase if/when prompted to do so.


**but sometime unlucky ,it still ask passpharases**

so we  need  follow this [answer](https://stackoverflow.com/a/58784438/1249911)

## To set GIT_SSH permanently

1. Open File Explorer. Start-> type 'File Explorer' and click on it in the list.
2. Right-click 'This PC' and click on 'Properties'.
3. Click on 'Advanced system settings'.
4. Click the 'Environment Variables...' button.
5. Under 'User variables for your_user_name' click New...
6. Set Variable name: field to GIT_SSH
7. Set the Variable value: field to path-to-ssh.exe (typically C:\Windows\System32\OpenSSH\ssh.exe).
8. Click OK to dismiss the New User Variable dialog.
9. Click OK to dismiss the Environment Variables dialog.
10. Retry the steps in Try Git + SSH above.