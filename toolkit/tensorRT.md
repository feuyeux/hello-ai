# NVIDIA

## TensorRT

<https://developer.nvidia.com/tensorrt>

<https://github.com/NVIDIA/TensorRT>
<https://github.com/NVIDIA/TensorRT-LLM>

量化

简单来说，就是将连续的浮点数值映射到有限的离散值上。在TensorRT中，量化技术被用于将FP32(32位浮点数)转换为INT8(8位整数)，从而大大减少模型大小、降低内存占用，并提高推理速度。TensorRT支持使用对称均匀量化方案，即将量化值以有符号INT8表示，从量化到非量化值的转换仅通过一个乘法操作实现。

Flash Attention

## Triton Inference Server
