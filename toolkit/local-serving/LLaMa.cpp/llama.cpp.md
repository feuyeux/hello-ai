# LLaMa.cpp

<https://github.com/ggerganov/llama.cpp>

## build on windows

<https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md>
<https://cmake.org/download/>

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

```sh
export PATH="/d/coding/llama.cpp/build/bin/Release":$PATH

$ llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, VMM: yes
version: 3950 (f594bc80)
built with MSVC 19.41.34120.0 for x64
```

## Supported backends

| Backend                                                      | Target devices        |
| ------------------------------------------------------------ | --------------------- |
| [Metal](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#metal-build) | Apple Silicon         |
| [BLAS](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#blas-build) | All                   |
| [BLIS](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/BLIS.md) | All                   |
| [SYCL](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md) | Intel and Nvidia GPU  |
| [MUSA](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#musa) | Moore Threads MTT GPU |
| [CUDA](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cuda) | Nvidia GPU            |
| [hipBLAS](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#hipblas) | AMD GPU               |
| [Vulkan](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan) | GPU                   |
| [CANN](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cann) | Ascend NPU            |

## install on macos

```sh
brew install llama.cpp
```

## install llama-cpp-python on macos

<https://github.com/abetlen/llama-cpp-python>

```sh
pip install --upgrade --quiet  llama-cpp-python
```
