---
layout: post
title: Adjust GPU (NVIDIA) Clock.
categories: [nvidia]
description: Adjust GPU (NVIDIA) Clock.
---

## NVIDIA-SMI -lgc and -rgc Usage Guide

NVIDIA's `nvidia-smi` (NVIDIA System Management Interface) provides the `-lgc` and `-rgc` options, allowing users to manually control GPU clock frequencies.

<!--more-->

### **Observed Behavior Without Adjusting LGC**
If the LGC setting is not adjusted, the following code exhibits different behaviors:

```cpp
void single_GPU(int gpuid) {
    {
        cudaSetDevice(gpuid);
        int uyvy_size = 3840 * 2160 * 2;
        uint8_t* pUyvy = new uint8_t[uyvy_size];
        uint8_t* pGPU_uyvy;
        cudaMalloc((void**)&pGPU_uyvy, uyvy_size);
        cudaEvent_t start, stop;
        cudaStream_t stream;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaStreamCreate(&stream);

        uint64_t g_loop = 0;
        float elapsedTime = 0;
        while (1) {
            cudaEventRecord(start, stream);

            cudaMemcpy(pGPU_uyvy, pUyvy, uyvy_size, cudaMemcpyHostToDevice);
            
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);

            printf("%d>> %lld >> %2.2f\n", gpuid, g_loop++, elapsedTime);
            Sleep(1000);
            // If sleep is not added, elapsedTime remains as expected.
            // If sleep is added, elapsedTime deviates from expectations.
            // If LGC is adjusted, elapsedTime decreases.
        }
    }
}
```

<table>
<tr>
<td>
<pre>
If sleep is added without setting LGC:

0>> 0 >> 1.78
0>> 1 >> 1.87
0>> 2 >> 1.97
0>> 3 >> 1.94
0>> 4 >> 1.89
0>> 5 >> 1.97
0>> 6 >> 1.91
0>> 7 >> 10.62
0>> 8 >> 11.03
0>> 9 >> 10.97
0>> 10 >> 10.94
0>> 11 >> 10.92
0>> 12 >> 10.92
0>> 13 >> 11.06
</pre>
</td><td>
<pre>
If sleep is not added:

0>> 4661 >> 1.64
0>> 4662 >> 1.73
0>> 4663 >> 1.65
0>> 4664 >> 1.63
0>> 4665 >> 1.68
0>> 4666 >> 1.65
0>> 4667 >> 1.66
0>> 4668 >> 1.67
0>> 4669 >> 1.65
0>> 4670 >> 1.64
0>> 4671 >> 1.66
</pre>
</td><td>
<pre>
If LGC is adjusted and sleep is added:

0>> 0 >> 1.76
0>> 1 >> 1.81
0>> 2 >> 1.81
0>> 3 >> 1.81
0>> 4 >> 1.83
0>> 5 >> 1.83
0>> 6 >> 1.82
0>> 7 >> 5.52
0>> 8 >> 6.30
0>> 9 >> 5.75
0>> 10 >> 6.62
0>> 11 >> 5.75
0>> 12 >> 5.74
0>> 13 >> 5.75
0>> 14 >> 5.78
</pre>
</td></tr></table>

## **1. `nvidia-smi -lgc` (Lock GPU Clock Frequency)**

### **Command Format:**
```
nvidia-smi -lgc <min_clock>,<max_clock>
```
This command locks the GPU clock frequency within a specified range to prevent it from entering low-power mode or downclocking.

### **Example:**
```
nvidia-smi -lgc 1000,2000
```
#### **Meaning:**
- Ensures the GPU frequency remains within the **1000MHz ~ 2000MHz** range.
- Prevents excessive downclocking or frequency fluctuations.

### **Check Supported Clock Ranges:**
```
nvidia-smi -q -d CLOCK
```
#### **Example Output:**
```yaml
Supported Clocks for GPU 00000000:01:00.0
Memory Clocks MHz : 5001, 5500
Graphics Clocks MHz : 300, 1000, 1500, 2000
```
This means the GPU supports frequency ranges from **300MHz to 2000MHz**.

---
## **2. `nvidia-smi -rgc` (Restore Default Frequency)**

### **Command Format:**
```
nvidia-smi -rgc
```
This command restores the GPU to its **default dynamic clock adjustment mode**, removing the `-lgc` restriction.

### **Example:**
```
nvidia-smi -rgc
```
#### **Meaning:**
- Allows the GPU to dynamically adjust its frequency.
- Restores the default **power-saving or performance mode** settings.

---
## **3. `-lgc` and `-rgc` Use Cases**

| **Use Case** | **Use `-lgc`** | **Use `-rgc`** |
|:---|:---|:---|
| Prevent excessive GPU downclocking (e.g., AI training, CUDA computing) | ✅ | ❌ |
| Lock stable performance, avoid frequency fluctuations | ✅ | ❌ |
| Restore default dynamic frequency adjustment | ❌ | ✅ |
| Prevent power constraints from affecting performance | ✅ | ❌ |

By using `nvidia-smi -lgc`, you can maintain stable GPU performance, and if needed, `nvidia-smi -rgc` restores the default settings.

