<!-- markdownlint-disable MD041 -->

| 架构 | A100 | H100 | H200 | GH200 | B100 | B200 | Full B200 | GB200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Ampere | Hopper | Hopper | Hopper | Blackwell | Blackwell | Blackwell | Blackwell |
| 显存大小 | 80GB | 80GB | 141GB | 96/144GB | 180/192GB | 180/192GB | 192GB | 384GB |
| 显存宽带 | 2TB/s | 3.35TB/s | 4.8TB/s | 4/4.9TB/s | 8TB/s | 8TB/s | 8TB/s | 16TB/s |
| FP16稠密算力(FLOPS) | 312T | 1P | 1P | 1P | 1.75P | 2.25P | 2.5P | 5P |
| INT8稠密算力(OPS) | 624T | 2P | 2P | 2P | 3.5P | 4.5P | 5P | 10P |
| FP8稠密算力(FLOPS) | X | 2P | 2P | 2P | 3.5P | 4.5P | 5P | 10P |
| FP6稠密算力(FLOPS) | X | X | X | X | 3.5P | 4.5P | 5P | 10P |
| FP4稠密算力(FLOPS) | X | X | X | X | 7P | 9P | 10P | 20P |
| NVLink宽带 | 600GB/s | 900GB/s | 900GB/s | 900GB/s | 1.8TB/s | 1.8TB/s | 1.8TB/s | 3.6TB/s |
| 功耗 | 400W | 700W | 700W | 1000W | 700W | 1000W | 1200W | 2700W |
| 备注 | 1个Die | 1个Die | 1个Die | 1个Grace CPU 1个H200 CPU | 2个Die  | 2个Die | 2个Die | 1个Grace CPU 2个Blackwell CPU |

[die (integrated circuit)](https://en.wikipedia.org/wiki/Die_(integrated_circuit))  裸晶 **裸片**

虚拟化技术

包括以下几种主要方式：

- 直通（Pass through）：将整个物理GPU设备直接分配给一个虚拟机或容器，适用于对GPU性能要求较高的应用，但缺乏灵活性和资源共享能力。
- 共享（Sharing）：将物理GPU设备划分为多个逻辑单元，多个虚拟机或容器可以共享同一个GPU资源，实现了资源的共享和隔离，提高了资源利用率，适用于并发较高但对性能要求不高的应用，但在性能和隔离性方面存在一定的限制。
- 全虚拟化（Full Virtualization）：通过软件模拟GPU硬件，使得虚拟机或容器可以独立运行，适用于对资源隔离和安全性要求较高的应用，但由于性能损耗较大，主要应用于对安全性要求较高的场景。
- GPU池化（GPU Pooling）：将多个物理GPU资源统一管理和调度，实现了资源的按需分配和动态调整，提供了更高的资源利用率和调度灵活性。

虚拟化技术实现体现三个层次，即用户层、内核层和硬件层。在不同软硬件层技术实现可分类为：

- 用户层：API 拦截和 API forwarding。
- 内核层：GPU 驱动拦截；GPU 驱动半虚拟化：Para Virtualization。
- 硬件层：
  - 硬件虚拟化：Virtualization；
  - SRIOV：Single Root I/O Virtualization；
  - Nvidia MIG：Multi-Instance GPU。

Kubernetes上 NVIDIA 提供了3种解决方案：

- 整卡（Full GPU）：整卡是指将整个 NVIDIA GPU 分配给单个用户或应用程序。在这种配置下，应用可以完全占用 GPU 的所有资源， 并获得最大的计算性能。整卡适用于需要大量计算资源和内存的工作负载，如深度学习训练、科学计算等。
- vGPU（Virtual GPU）：vGPU 是一种虚拟化技术，允许将一个物理 GPU 划分为多个虚拟 GPU，每个虚拟 GPU 分配给不同的虚拟机或用户。 vGPU 使多个用户可以共享同一台物理 GPU，并在各自的虚拟环境中独立使用 GPU 资源。 每个虚拟 GPU 可以获得一定的计算能力和显存容量。vGPU 适用于虚拟化环境和云计算场景，可以提供更高的资源利用率和灵活性。
- MIG（Multi-Instance GPU）：MIG 是 NVIDIA Ampere 架构引入的一项功能，它允许将一个物理 GPU 划分为多个物理 GPU 实例，每个实例可以独立分配给不同的用户或工作负载。 每个 MIG 实例具有自己的计算资源、显存和 PCIe 带宽，就像一个独立的虚拟 GPU。 MIG 提供了更细粒度的 GPU 资源分配和管理，可以根据需求动态调整实例的数量和大小。 MIG 适用于多租户环境、容器化应用程序和批处理作业等场景。

| 方案名称 | 描述 | 优势 | 不足 | 演进阶段 |
| --- | --- | --- | --- | --- |
| nvidia MPS | 是nvidia为了进行GPU共享而推出的一套方案,由多个CUDA程序共享同一个GPU context,从而达到多个CUDA程序共享GPU的目的。同时可以设定每个CUDA程序占用的GPU算力的比例。 | 官方出品，稳定可靠，实现了资源隔离。 | 多个CUDA程序共享了同一个GPU context，一个程序异常会导致全局。 | 阶段1 |
| nvidia vGPU | 可以将一个设备映射到多个不同的虚拟机中去 | 同上 | 每个vGPU的显存和算力都是固定的，无法灵活配置。需要单独的license授权。 | 阶段2 |
| nvidia MIG | 多实例，MIG可将A100GPU划分为多达七个实例，每个实例均与各自的高带宽显存、缓存和计算核心完全隔离。 | 同上 | 目前只针对A100等有限的实现了MIG，对老卡则无法配置 | 阶段2 |
| 阿里cGPU | 通过劫持内核驱动，为容器提供了虚拟的GPU设备从而实现了显存和算力的隔离； | 适配开源标准的Kubernetes和NVIDIA Docker方案；用户侧透明。AI应用无需重编译，执行无需CUDA库替换；同时支持GPU的显存和算力隔离。 | 需要定制内核版本，没有开源。 | 阶段2+ |
| 腾讯qGPU | 它是目前业界唯一真正实现了故障隔离、显存隔离、算力隔离、且不入侵生态的容器GPU共享的技术。（官方表述） | 适配开源标准的K8s和Docker，无需重编译AI应用，运行时无需替换CUDA库，升级CUDA支持GPU的显存和算力隔离 | 没有继续维护 | 阶段2+ |
| 趋动科技 OrionX(猎户座) | 实现了GPU共享的能力，在资源隔离方面，使用了CUDA劫持的方案，通过MPS以及其他方式限制算力。 | 基础的：显存隔离、算力隔离；高级的：实现了GPU资源池化，可以远程调用；通过统一管理GPU，降低GPU的管理复杂度和成本 | 需要定制化使用，Orion Client Runtime的容器镜像。没有开源 | 阶段4 |
| 百度双引擎 GPU | 双引擎GPU容器虚拟化架构，采用了用户态和内核态两套隔离引擎，以满足用户对隔离性、性能、效率等多方面不同侧重的需求。 | 基础的：显存隔离、算力隔离。显存MB级隔离；算力1%级分配；高级的：编码器隔离、高优抢占、显存超发、显存池化。2种模式：可选择内核态和用户态模式 | 技术复杂度高，没有开源 | 阶段4 |
