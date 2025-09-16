---
layout: post
title: publisher could not be verified
categories: [link]
description: publisher could not be verified
---

The publisher could not be verified. Are you sure you want to run this software?


![IMG](https://arcserve-2.my.salesforce.com/sfc/dist/version/download/?oid=00DfJ000009Hi8g&ids=068fJ000002athr&d=%2Fa%2FfJ00000031D8%2F9yPLaqz3bY.PJeS7hMQEFM.XqFaCT9Lm0ARvEAJz.sM&asPdf=false)

[article](https://support.arcserve.com/s/article/201990449?language=en_US)

# Solution:

1. Click Start > Run and type gpedit.msc. Click OK

2. Go to User Configuration > Administrative Templates > Windows Components > Attachment Manager

3. Open the Inclusion list for moderate risk file types setting.

4. Set the policy to Enabled, then add  *.bat in Specify high risk extensions box.

![IMG](https://arcserve-2.my.salesforce.com/sfc/dist/version/download/?oid=00DfJ000009Hi8g&ids=068fJ000002aths&d=%2Fa%2FfJ00000031D9%2F36TRliQvd_Kv4z35UGhDJWz7JXyl9FWZS6YVrO1hL2c&asPdf=false)

5. Click OK and close the Group Policy Editor.