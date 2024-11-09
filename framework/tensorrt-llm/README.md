# TensorRT-LLM

A TensorRT Toolbox for Optimized Large Language Model Inference

<https://github.com/NVIDIA/TensorRT-LLM>

<https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/>

Many optimization techniques have risen to deal with this, from model optimizations like

- [kernel fusion](https://arxiv.org/abs/2307.08691) and
- [quantization](https://nvidia.github.io/TensorRT-LLM/precision.html) to runtime optimizations like C++ implementations,
- KV caching,
- [continuous in-flight batching](https://www.usenix.org/conference/osdi22/presentation/yu), and
- [paged attention](https://arxiv.org/pdf/2309.06180.pdf).
It can be difficult to decide which of these are right for your use case, and to navigate the interactions between these techniques and their sometimes-incompatible implementations.

That’s why [NVIDIA introduced TensorRT-LLM](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/), a comprehensive library for compiling and optimizing LLMs for inference. TensorRT-LLM incorporates all of those optimizations and more while providing an intuitive Python API for defining and building new models.

<https://github.com/QwenLM/Qwen/blob/refs%2Fheads%2Fmain/recipes%2Finference%2Ftensorrt%2FREADME.md>
