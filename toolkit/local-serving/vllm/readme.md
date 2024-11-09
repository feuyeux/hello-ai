# vllm

- Doc: <https://docs.vllm.ai/en/latest/>
- Code: <https://github.com/vllm-project/vllm>
- Paper: [Efficient Memory Management for Large Language Model Serving with **PagedAttention**](https://arxiv.org/abs/2309.06180)

## Installation

### WSL

```sh
conda create -y -n vllm_env python=3.10
conda activate vllm_env
pip install --upgrade pip
pip install vllm
```

```sh
# The latest version of vLLM needs v2.4.0, 
# but PyTorch has stopped building MacOS x86_64 binaries since torch v2.3.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## OpenAI-Compatible Server

```sh
vllm serve facebook/opt-125m
```

## Run code

```sh
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

```sh
## HuggingFace workaround
nano ~/anaconda3/envs/torch_trt/lib/python3.12/site-packages/huggingface_hub/constants.py
export HF_ENDPOINT="https://hf-mirror.com"
python /mnt/d/coding/hello-ai/tools/local_llms/vllm/hello.py
```
