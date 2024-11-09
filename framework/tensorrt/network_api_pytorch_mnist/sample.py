import frameworks.tensorrt.common as common
import os
import sys

# 该示例使用 MNIST PyTorch 模型创建 TensorRT 推理引擎
import model
import numpy as np

import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

# 您可以设置日志记录器的严重性，以抑制消息（或显示更多消息）。
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    """
    保存模型数据常量的类。
    """
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    """
    根据提供的权重配置网络层。

    参数:
        network (trt.INetworkDefinition): TensorRT 网络定义
        weights (dict): 包含模型权重的字典
    """

    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE
    )

    def add_matmul_as_fc(net, input, outputs, w, b):
        """
        使用矩阵乘法添加全连接层

        参数:
            net (trt.INetworkDefinition): TensorRT 网络定义。
            input (trt.ITensor): 输入张量。
            outputs (int): 输出通道数。
            w (np.ndarray): 全连接层的权重。
            b (np.ndarray): 全连接层的偏置。

        返回:
            trt.IShuffleLayer: 重新调整形状后的输出层。
        """
        assert len(input.shape) >= 3
        m = 1 if len(input.shape) == 3 else input.shape[0]
        k = int(np.prod(input.shape) / m)
        assert np.prod(input.shape) == m * k
        n = int(w.size / k)
        assert w.size == n * k
        assert b.size == n

        input_reshape = net.add_shuffle(input)
        input_reshape.reshape_dims = trt.Dims2(m, k)

        filter_const = net.add_constant(trt.Dims2(n, k), w)
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        bias_const = net.add_constant(trt.Dims2(1, n), b)
        bias_add = net.add_elementwise(
            mm.get_output(0), bias_const.get_output(
                0), trt.ElementWiseOperation.SUM
        )

        output_reshape = net.add_shuffle(bias_add.get_output(0))
        output_reshape.reshape_dims = trt.Dims4(m, n, 1, 1)
        return output_reshape

    conv1_w = weights["conv1.weight"].cpu().numpy()
    conv1_b = weights["conv1.bias"].cpu().numpy()
    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=20,
        kernel_shape=(5, 5),
        kernel=conv1_w,
        bias=conv1_b,
    )
    conv1.stride_nd = (1, 1)

    pool1 = network.add_pooling_nd(
        input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2)
    )
    pool1.stride_nd = trt.Dims2(2, 2)

    conv2_w = weights["conv2.weight"].cpu().numpy()
    conv2_b = weights["conv2.bias"].cpu().numpy()
    conv2 = network.add_convolution_nd(
        pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b
    )
    conv2.stride_nd = (1, 1)

    pool2 = network.add_pooling_nd(
        conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride_nd = trt.Dims2(2, 2)

    fc1_w = weights["fc1.weight"].cpu().numpy()
    fc1_b = weights["fc1.bias"].cpu().numpy()
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 500, fc1_w, fc1_b)

    relu1 = network.add_activation(
        input=fc1.get_output(0), type=trt.ActivationType.RELU
    )

    fc2_w = weights["fc2.weight"].cpu().numpy()
    fc2_b = weights["fc2.bias"].cpu().numpy()
    fc2 = add_matmul_as_fc(
        network, relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b
    )

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))


def build_engine(weights):
    """
    构建并返回一个 TensorRT 引擎。

    参数:
        weights (dict): 包含模型权重的字典。

    返回:
        trt.ICudaEngine: TensorRT 引擎。
    """
    builder = trt.Builder(TRT_LOGGER)
    print("# 创建网络定义")
    # 使用 PyTorch 模型的权重填充网络"
    network = builder.create_network(0)
    # 根据提供的权重配置网络层
    populate_network(network, weights)

    print("# 创建构建配置-- 指定TensorRT应该如何优化模型")
    config = builder.create_builder_config()
    # 最大工作空间大小
    # 层实现通常需要一个临时工作空间，并且此参数限制了网络中任何层可以使用的最大大小
    # 如果提供的工作空间不足，TensorRT 可能无法找到层的实现
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

    print("# 构建和序列化TensorRT引擎")
    plan = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    # 从内存缓冲区反序列化引擎
    return runtime.deserialize_cuda_engine(plan)


def load_random_test_case(model, pagelocked_buffer):
    """
    随机选择一个图像作为测试用例，并将其复制到页锁定输入缓冲区。

    参数:
        model (model.MnistModel): MNIST 模型。
        pagelocked_buffer (np.ndarray): 页锁定输入缓冲区。

    返回:
        int: 测试用例的预期输出。
    """
    print("# [load_random_test_case] 随机选择一个图像作为测试用例。")
    img, expected_output = model.get_random_testcase()
    print("# [load_random_test_case] 复制到页锁定输入缓冲区")
    np.copyto(pagelocked_buffer, img)
    return expected_output


def main():
    """
    主函数，用于训练模型、构建引擎并对随机测试用例进行推理。
    """
    print("# 1 训练模型")
    mnist_model = model.MnistModel()
    mnist_model.learn()
    # 权重张量
    weights = mnist_model.get_weights()
    print("# 2 构建引擎")
    engine = build_engine(weights)
    print()
    print("# 3 分配缓冲区并创建流")
    # 要执行推理，您必须为输入和输出传递 TensorRT 缓冲区，TensorRT 要求您在 GPU 指针列表中指定。
    # 您可以使用为输入和输出张量提供的名称查询引擎，以在数组中找到正确的位置
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # 要执行推理需要额外的中间激活状态。这是通过 IExecutionContext 接口完成的
    context = engine.create_execution_context()

    print("# 4 执行推理")
    # common.do_inference 函数将返回一个输出列表 - 在这种情况下我们只有一个
    [output] = common.do_inference(
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    pred = np.argmax(output)
    case_num = load_random_test_case(
        mnist_model, pagelocked_buffer=inputs[0].host)
    common.free_buffers(inputs, outputs, stream)
    print("测试用例: " + str(case_num))
    print("预测: " + str(pred))


if __name__ == "__main__":
    main()
