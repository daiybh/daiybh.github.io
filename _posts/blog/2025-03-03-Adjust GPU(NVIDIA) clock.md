---
layout: post
title: Adjust GPU(NVIDIA) clock.
categories: [nvidia]
description: Adjust GPU(NVIDIA) clock.
---

NVIDIA-SMI -lgc 和 -rgc 用法介绍
NVIDIA 的 nvidia-smi（NVIDIA System Management Interface）提供了 -lgc 和 -rgc 选项，可以手动控制 GPU 的工作频率。

<!--more-->


在使用中发现 如果不调整 lgc  

那么 以下代码会有不同的表现

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

			printf("%d>> %lld >> %2.2f\n",gpuid, g_loop++, elapsedTime);
			Sleep(1000); 
            //如果不加sleep 那么 elapsedTime 会在预期内
            //如果加了sleep 那么 elapsedTime 会在预期外
            // 如果调整了 LGC 那么 elapsedTime会变小
		}
	}
}
```

## 如果加了 sleep
```
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
```
## 如果没有加 sleep
```
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
```

## 如果调整了lgc  并加sleep
```
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
```


# 1. nvidia-smi -lgc（锁定 GPU 频率）

命令格式：

 
    nvidia-smi -lgc <min_clock>,<max_clock>

该命令用于锁定 GPU 频率范围，避免 GPU 进入低功耗模式或降频。

示例
 
    nvidia-smi -lgc 1000,2000

含义：

让 GPU 频率在 1000MHz ~ 2000MHz 之间运行，防止过低或过高的波动。
查看 GPU 频率支持的范围

    nvidia-smi -q -d SUPPORTED_CLOCKS

输出示例：

```yaml
Supported Clocks for GPU 00000000:01:00.0
Memory Clocks MHz : 5001, 5500
Graphics Clocks MHz : 300, 1000, 1500, 2000
```
表示该 GPU 支持的频率范围为 300MHz 到 2000MHz。

# 2. nvidia-smi -rgc（恢复默认频率）

命令格式：

    nvidia-smi -rgc

该命令用于恢复 GPU 频率到默认的动态调节模式，取消 -lgc 限制。

示例

    nvidia-smi -rgc

含义：

让 GPU 重新允许动态调整频率，恢复默认的省电或性能模式。

# 3. -lgc 和 -rgc 的适用情况

|适用场景	|使用 -lgc	|使用 -rgc|
|:---|:---|:---|
|防止 GPU 过度降频（如 AI 训练、CUDA 计算）	|✅	|❌|
|锁定稳定性能，避免频率波动	|✅	|❌|
|恢复默认动态频率	|❌	|✅|
|防止功耗限制影响计算	|✅	|❌|