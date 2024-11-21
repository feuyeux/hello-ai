# TensorRT

<https://github.com/NVIDIA/TensorRT>

<https://developer.nvidia.com/tensorrt>

NVIDIA® TensorRT™ is an ecosystem of APIs for high-performance deep learning inference. TensorRT includes an inference runtime and model optimizations that deliver low latency and high throughput for production applications. The TensorRT ecosystem includes

1. TensorRT,
2. TensorRT-LLM,
3. TensorRT Model Optimizer,
4. TensorRT Cloud.

[NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/index.html)

## Install TensorRT

### 0 check

```sh
$ uname -m && cat /etc/*release
x86_64
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.4 LTS"
PRETTY_NAME="Ubuntu 22.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.4 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy

$ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

$ uname -r
5.15.153.1-microsoft-standard-WSL2
```

### 1 cuda

[Ubuntu22.04 cuda12.5.1]<https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network>)

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

```sh
$ nvidia-smi
Sun Jul 21 17:32:58 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.85         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   46C    P8              2W /   95W |      41MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### 2 tensorrt

[TensorRT 10.2 GA for Ubuntu 22.04 and CUDA 12.0 to 12.5 DEB local repo Package](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.2.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.2.0-cuda-12.5_1.0-1_amd64.deb)

```sh
sudo dpkg -i /mnt/d/download/nv-tensorrt-local-repo-ubuntu2204-10.2.0-cuda-12.5_1.0-1_amd64.deb

sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.2.0-cuda-12.5/nv-tensorrt-local-012FC2A5-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get install tensorrt
```

```sh
$ dpkg-query -W tensorrt
tensorrt        10.2.0.19-1+cuda12.5
```

### 3 cuDNN

<https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network>

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
```

## samples

```sh
conda create --name trt python=3.12
conda activate trt
pip install --upgrade pip
```

<https://github.com/NVIDIA/TensorRT/tree/main/samples>

<https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html>
