# sampleOnnxMNIST

<https://github.com/NVIDIA/TensorRT/tree/main/samples/python/network_api_pytorch_mnist>

trains a convolutional model on the MNIST dataset and runs inference with a TensorRT engine.

- MNIST database(**M**odified **N**ational **I**nstitute of **S**tandards and **T**echnology database) is a large database of handwritten digits that is commonly used for training various image processing systems.
  - `mnist.py`: `D:\garden\anaconda3\envs\trt\Lib\site-packages\torchvision\datasets\mnist.py`
- CNN(Convolutional Neural Network):
  - The **Activation layer** implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `RELU`.
  - The **Convolution layer** computes a 2D (channel, height, and width) convolution, with or without bias.
  - The **MatrixMultiply layer** implements a matrix multiplication.
  - The bias of FullyConnected semantic can be added with an **ElementwiseLayer** of `SUM` operation.
  - The **Pooling layer** implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

```sh
conda activate trt

cd frameworks/tensorrt/network_api_pytorch_mnist
pip install Pillow>=10.0.0 pyyaml==6.0.1 requests==2.32.2 tqdm==4.66.4 numpy==1.26.4
pip install cuda-python==12.5.0 torch torchvision
pip install tensorrt_cu12_libs==10.2.0 tensorrt_cu12_bindings==10.2.0 tensorrt==10.2.0 --extra-index-url https://pypi.nvidia.com
python -c "import tensorrt; print(tensorrt.__version__)"
```

```sh
python sample.py
```

```sh
conda deactivate
```

----

- <https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#network_api_pytorch_mnist>
- <https://github.com/NVIDIA/TensorRT/tree/release/10.2/samples/python/network_api_pytorch_mnist>
