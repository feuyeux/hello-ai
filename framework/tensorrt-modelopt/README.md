# NVIDIA TensorRT Model Optimizer

A Library to Quantize and Compress Deep Learning Models for Optimized Inference on GPUs

<https://github.com/NVIDIA/TensorRT-Model-Optimizer>

Minimizing inference costs presents a significant challenge as generative AI models continue to grow in complexity and size. The **NVIDIA TensorRT Model Optimizer** (referred to as **Model Optimizer**, or **ModelOpt**) is a library comprising state-of-the-art model optimization techniques including [quantization](https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#quantization) and [sparsity](https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#sparsity) to compress models. It accepts a torch or [ONNX](https://github.com/onnx/onnx) model as inputs and provides Python APIs for users to easily stack different model optimization techniques to produce an optimized quantized checkpoint. Seamlessly integrated within the NVIDIA AI software ecosystem, the quantized checkpoint generated from Model Optimizer is ready for deployment in downstream inference frameworks like [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) or [TensorRT](https://github.com/NVIDIA/TensorRT). Further integrations are planned for [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for training-in-the-loop optimization techniques. For enterprise users, the 8-bit quantization with Stable Diffusion is also available on [NVIDIA NIM](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/).

## Quantization

Quantization is an effective model optimization technique for large models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality. Model Optimizer enables highly performant quantization formats including FP8, INT8, INT4, etc and supports advanced algorithms such as SmoothQuant, AWQ, and Double Quantization with easy-to-use Python APIs. Both Post-training quantization (PTQ) and Quantization-aware training (QAT) are supported.

## Sparsity

Sparsity is a technique to further reduce the memory footprint of deep learning models and accelerate the inference. Model Optimizer provides Python API `mts.sparsity()` to apply weight sparsity to a given model. `mts.sparsity()` supports [NVIDIA 2:4 sparsity pattern](https://arxiv.org/pdf/2104.08378) and various sparsification methods, such as [NVIDIA ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) and [SparseGPT](https://arxiv.org/abs/2301.00774).
