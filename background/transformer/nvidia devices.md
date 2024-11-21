# NVIDIA DEVICES

| 英文全称                       | 英文缩写 | 中文名称         | 技术说明                                                     |
| ------------------------------ | -------- | ---------------- | ------------------------------------------------------------ |
| Streaming Multiprocessor       | SM       | 流式多处理器     | NVIDIA GPU 的核心组件，负责执行图形和计算任务，每代产品中都会进行优化和增强以提升性能 |
| Deep Learning Super Sampling   | DLSS     | 深度学习超级采样 | 利用深度学习神经网络提取渲染场景的多维特征，智能结合多帧图像中的细节来构造高质量的最终图像 |
| Real-Time Ray Tracing          | RTX      | 实时光线追踪     | NVIDIA RTX 平台融合光线追踪、深度学习和栅格化技术，通过 NVIDIA Turing GPU 架构提供实时光线追踪渲染 |
| Tensor Core                    | -        | 张量核心         | 专为深度学习矩阵运算设计的硬件单元，用于加速 AI 计算和某些图形任务 |
| Ray Tracing Core               | RT Core  | 光线追踪核心     | 专用硬件单元，用于加速实时光线追踪计算，包括光线与场景中对象的相交测试等 |
| Multi-GPU Technology           | -        | 多GPU技术        | 如 SLI 和 NVLink，支持多个 GPU 协同工作，提供更高的图形渲染和计算性能 |
| NVIDIA Core SDK                | NVAPI    | NVIDIA 核心 SDK  | 提供对 NVIDIA GPU 和驱动程序的直接访问，支持多 GPU 和显示器等操作 |
| High-Speed Interconnect        | NVLink   | 高速互联技术     | 提供 GPU 与 GPU、GPU 与 CPU 之间的高速数据传输               |
| Optimus Technology             | Optimus  | 优驰技术         | 智能切换集成显卡与 NVIDIA GPU，以提供性能和延长电池续航时间  |
| PhysX Engine                   | PhysX    | 物理效果引擎     | 用于在 GPU 上实现逼真的物理效果，如碰撞、流体和软体动力学    |
| PostWorks Technology           | -        | 后处理技术       | 用于游戏和应用程序的后期视觉效果处理，如 TXAA 和 Bokeh 深度效果 |
| Scalable Link Interface        | SLI      | 可扩展连接接口   | 允许多个 NVIDIA GPU 在单个系统中协同工作，提高渲染能力       |
| Virtual GPU Technology         | vGPU     | 虚拟 GPU 技术    | 允许多个虚拟机共享同一物理 GPU 资源，优化虚拟化环境的图形性能 |
| GameWorks SDK                  | -        | 开发者工具 SDK   | 提供一系列工具和库，帮助开发者实现先进的视觉效果和物理模拟   |
| G-SYNC Display Synchronization | G-SYNC   | 显示同步技术     | 同步显示器刷新率与 GPU 输出，减少撕裂和卡顿，提供平滑的游戏体验 |
| GPU Boost Technology           | GPUBoost | GPU 性能提升技术 | 根据系统冷却能力动态提升 GPU 性能                            |

GPU 的核心架构及参数
CUDA Core：CUDA Core 是 NVIDIA GPU上的计算核心单元，用于执行通用的并行计算任务，是最常看到的核心类型。NVIDIA 通常用最小的运算单元表示自己的运算能力，CUDA Core 指的是一个执行基础运算的处理元件，我们所说的 CUDA Core 数量，通常对应的是 FP32 计算单元的数量。
Tensor Core：Tensor Core 是 NVIDIA Volta 架构及其后续架构（如Ampere架构）中引入的一种特殊计算单元。它们专门用于深度学习任务中的张量计算，如矩阵乘法和卷积运算。Tensor Core 核心特别大，通常与深度学习框架（如 TensorFlow 和 PyTorch）相结合使用，它可以把整个矩阵都载入寄存器中批量运算，实现十几倍的效率提升。
RT Core：RT Core 是 NVIDIA 的专用硬件单元，主要用于加速光线追踪计算。正常数据中心级的 GPU 核心是没有 RT Core 的，主要是消费级显卡才为光线追踪运算添加了 RTCores。RT Core 主要用于游戏开发、电影制作和虚拟现实等需要实时渲染的领域。

https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units

| Product       | Compute Capability | GPU Architecture | Tensor Cores | CUDA cores | FP32 (TFLOPS) | FP16 (TFLOPS) | INT8 (TOPS) | INT4 (TOPS) | Interconnect            | Memory Capacity | Memory Bandwidth |
| :------------ | ------------------ | ---------------- | ------------ | ------------- | -------------- | -------------- | ----------- | ----------- | ----------------------- | --------------- | ---------------- |
| GeForce RTX 4060 | 8.9                | Ada Lovelace     | 144          | 3072          | 12             | 48             | 96           | 192         | Gen4 PCIe               | 8 GB GDDR6      | 256 GB/s         |
| NVIDIA T4     | 7.5                | Turing           | 320          | 2560          | 8.1            | 65             | 130          | 260         | Gen3 x16 PCIe           | 16 GB GDDR6     | 320 GB/s         |
| NVIDIA A10    | 8.6                | Ampere           | 192          | 8192          | 31.2           | 125            | 250          | 500         | PCIe Gen4 64GB/s        | 24 GB GDDR6     | 600 GB/s         |
| NVIDIA **A100** | 8.0 | Ampere | 432 | 6912 | 19.5 | 312 | 624 | - | NVLink x 12 | 80GB HBM2e | 2039 GB/s |
| A800(A100) |  |  |  |  |  |  |  |  | NVLink x 8 |  |  |
| NVIDIA **L40S** | 8.9                | Ada Lovelace     | 568          | 18176         | 91.6           | 733            | 1466         | 733        | PCIe Gen4 x16: 64GB/s   | 48 GB GDDR6 with ECC | 864 GB/s        |
| L20(L40S)     |                    |                  | 432          | 14336         | 59.8           | 119.5         | 239          | -           | PCIe Gen4 x16 64GB/s    | 48GB            | 854 GB/s         |
| L2            |                    |                  | 288          | 9856          | 48.3           | 96.5           | 193          | -           | PCIe Gen4 x16 64GB/s    | 24GB            | 300 GB/s         |
| NVIDIA **H100** | 9.0                | Hopper           | 1024         | 16,896   | 67             | 990         | 3958         | -           | NVLink x 18          | 80GB HBM3       | 3.35 TB/s        |
| H800(H100) |                    |                  |              |            |               |               |             | -           | NVLink x 8           |        |         |
| H20(H800)     |                    |                  |              |          | 44             | 148            | 296          | -           | PCIe Gen5 x16 128GB/s   | 95GB HBM3       | 4 TB/s           |
| NVIDIA **H200** | 9.0 | Hopper | 1024 |  | 67 | 1979 | 3958 | 1,466 | SXM | 141GB HBM3 | 4.8 TB/s |


[GeForce RTX 4060](https://www.nvidia.com/en-us/geforce/laptops/compare/)
[NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[NVIDIA A10](<https://www.nvidia.com/en-us/data-center/products/a10-gpu/>
