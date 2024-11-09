<!-- markdownlint-disable MD033 -->

# 量化、剪枝和蒸馏：优化神经网络进行推理

视频地址：[Quantization vs Pruning vs Distillation: Optimizing NNs for Inference (youtube.com)](https://www.youtube.com/watch?v=UcwDgsMgTu4)

视频所属播放列表：[NLP Deep Dives - YouTube](https://www.youtube.com/playlist?list=PLc7il9kHHib2FOazrkCUBqGdQviq-thQu)

作者Bai Li的博客：<https://luckytoilet.wordpress.com/>

----

大家好，今天的视频的主题是如何压缩和优化模型。假设我们已经训练(trained)或微调(fine-tuned)了模型，现在准备部署(deploy)它供大家使用。但我们发现延迟(latency)太慢，想让模型更快。今天我将展示四种方法来加速我们的模型：

- 量化(Quantization)
- 剪枝(Pruning)
- 知识蒸馏(Knowledge distillation)
- 工程优化(Engineering optimizations)。

我叫Bai，一名机器学习工程师，也是自然语言处理的博士。

## 1 量化

让我们直接开始讨论量化。神经网络很大，占用很多空间，因为它们有数百万或数十亿的参数。默认情况下，当我们训练神经网络时，通常参数以FP32格式存储，这意味着**每个参数占用32位(32 bits)**。

量化的思想是的降低格式的精度，使参数占用更少的空间。例如FP16(16位浮点数)或INT8(8位整数)。如果我们以INT8格式存储所有参数，这意味着参数都被表示为0到255之间的整数，那么将比原始FP32格式的网络节省四倍的空间。

### 1.1 零点量化(Zero-point Quantization)

将浮点数转换为整数有几种方法，最常见的方法是称为零点量化。我将通过一个例子来说明这是如何工作的。

<img src="Zero-point_Quantization.png" alt="Zero-point_Quantization" width="550" />

之所以称之为零点量化，是因为原始矩阵(original matrix)中的所有零都被映射到量化版本的零。我们将看到为什么这对于稀疏神经网络(sparse neural network)很有用。接下来我们取最大绝对值元素(the maximum absolute value element)，并将其映射到`-128`或`127`。

在本例中，最大绝对值元素是`-51.5`，所以它被映射到`-128`。量化必须是线性变换(linear transformation)，所以一旦确定了两个元素，其余的元素也就确定了。最后，为了得到INT8表示，我们在每个元素上加128，使所有元素都是正数。

### 1.2 权重(Weight)与激活(Activation)量化

量化有两种方式：权重量化和激活量化。

<img src="Weight_vs_Activation_Quantization.png" alt="Weight_vs_Activation_Quantization" width="550" />

在**权重量化**中，神经网络的所有权重以INT8格式存储，在运行时将权重解量化(dequantize)为FP32，以便网络中的数据保持FP32格式。由于一切都是在FP32中完成的，它不会比原始模型更快，但仍然有用，因为它节省了空间。例如，在移动设备上，使模型缩小四倍是一个显著的改进。

另一方面，在**激活量化**中，我们将所有的输入转换为INT8，所有的计算也在INT8中执行。这样就比权重量化更快，因为在大多数硬件上INT8计算比FP32更快。但一个挑战是我们在量化模型时不知道神经网络的输入。因此，为了确定每层的缩放因子(scale factors)，我们需要一个**校准集(calibration set)**，代表我们在推理时预期看到的数据类型。如果我们遇到过静态或动态量化这些术语，这些指的是确定激活的缩放因子的不同方式。如果校准没有做好，我们会在网络中遇到剪切(clipping)，因为量化只能在特定范围内处理浮点数，任何超出范围的值都会剪切到最大或最小值。

查看我们打算进行推理的硬件规格，有助于确定使用哪种类型的量化。这里我找到了Nvidia A10 GPU的数据表(data sheet)，这是一款跑推理的主流GPU卡。根据规格表(specification sheet)，这款GPU的FP32性能是31 TeraFLOPS。而INT8性能要快得多，每秒250个张量运算。这要归功于它的张量核心能力(tensor core capabilities)。但并非所有GPU都有这种能力。所以在一些旧的GPU上，我们可能会发现FP32和INT8的性能一样。

<img src="A10.png" alt="A10" width="650" />

### 1.3 LLM.INT8: 混合分解(Mixed decompostition) (PTQ)

我们应该意识到另一个对量化有影响的是**异常值(outlier)**。最近的一篇 [LLM.INT8()](https://arxiv.org/abs/2208.07339) 的论文发现，在超过60亿参数的大型语言模型中，异常特征导致量化不起作用，模型的性能降至接近零。

<img src="quantization_doesnt_work.png" alt="quantization_doesnt_work" width="650" />

为了理解为什么会这样，请考虑如果我们的权重中有一个异常值会发生什么。桶将会变得非常大，因为介于最小值和最大值之间，只有256个桶来覆盖所有值，包括异常值。为了解决这个问题，他们提出了一个混合分解方案，其中异常值与大多数数据分开处理。

当我们运行较小的模型时，这不是必要的，但如果是对较大的语言模型进行量化，了解这一点很有用。

## 2 剪枝

现在让我们继续讨论第二种方法，剪枝。剪枝的基本思想是移除神经网络中的一些连接(connections)。这样我们就得到了所谓的**稀疏网络(sparse network)**。在矩阵计算(matrix computation)方面，矩阵中的许多值被设置为零，这使得存储更便宜，计算更快。

<img src="pruning.png" alt="pruning" width="550" />

### 2.1 幅度剪枝(Magnitude pruning) -- 非结构化剪枝(Unstructured pruning)

同样，我们可以使用许多不同的算法进行剪枝。在这个视频中，我只讨论最简单的一种——幅度剪枝。

<img src="magnitude_pruning.png" alt="magnitude_pruning" width="550" />

在幅度剪枝中，你首先选择一个剪枝因子X，它表示我们想要移除哪部分的连接。然后在网络的每一层，将绝对值最低的X的权重百分比设置为零。原因是，绝对值最低的权重，也就是最接近零的权重，对于网络的功能来说是最不重要的。通过移除一些连接，模型的准确度(accuracy)将经历一些退化(degradation)。所以作为可选的第三步，在保持移除的权重固定为零的同时，重新训练几次模型，以恢复一些准确度。

需要注意的是，仅仅将一些矩阵值设置为零实际上并不能节省空间或使其更快。因为零和非零值一样需要存储空间，处理时间也一样。所以当我们进行剪枝的时候，需要将其与某种**稀疏执行引擎(sparse execution engine)**结合起来，这种引擎可以利用稀疏化的神经网络结构。

让我来举例说明。通常，当你的GPU执行矩阵乘法(matrix multiplication)时，它会遍历两个矩阵的切片。每对切片累积(accumulates)一个外积矩阵(out-of-product matrix)。所有这些的和就是矩阵乘法。即使切片中有零，它也不影响这个操作需要多长时间。另一方面，专门设计用于乘以稀疏矩阵的算法，在稀疏矩阵乘法算法中有一个特殊的技巧，可以跳过(skips over)向量(vector)中的所有零条目，所以矩阵中的零越多，乘法就会越快。

<img src="sparse_matrix_mul.png" alt="sparse_matrix_mul" width="550" />

### 2.2 N:M 稀疏度(Sparsity) -- 结构化剪枝(structured pruning)

最后我将讨论结构化剪枝。如果你简单地从网络中移除连接，没有任何进一步的模式，那被称为非结构化剪枝。而结构化剪枝允许在设置为零的权重上施加更多的结构。一种结构化剪枝是**2-4结构稀疏度**模式。这意味着对于每四个连续矩阵值(consecutive matrix values)的块，只允许其中两个是非零的。这使得我们可以压缩格式存储矩阵，只存储非零值，以及表示值所在位置的索引。

<img src="2-4_structured_sparcity.png" alt=" 2-4_structured_sparcity" width="550" />

此外，Nvidia的张量核心GPU能够以更高的效率执行这种类型的结构化稀疏度。所以我们看到，对于剪枝神经网络，我们需要考虑硬件来设计剪枝算法。

使用哪种剪枝算法取决于我们打算部署神经网络的硬件上运行哪种类型的稀疏度。

## 3 蒸馏

使我们的模型更有效的第三种方法称为**知识蒸馏(knowledge distillation)**，有时也称为**模型蒸馏(model distillation)**。

那么什么是知识蒸馏呢？在知识蒸馏中，我们首先使用数据训练一个**教师网络(teacher network)**。教师网络训练完成后，我们开始训练**学生网络(student network)**来预测(predict)教师网络的输出。

为什么让学生预测教师网络的输出，会比仅仅从标签训练学生网络更有帮助呢？原因基本上是，教师网络的输出包含更多信息，因此学生网络从中学习更快更容易。假设我们正在训练一些分类模型，那么训练数据每训练实例只有一个标签。但是教师网络的输出给你提供了所有可能标签的概率分布，有更多的学习信息。

<img src="knowledge_distillation.png" alt="knowledge_distillation" width="550" />

知识蒸馏与其他优化模型方法相比有优点也有缺点。

<img src="distillation_advantagesNdisadvantages.png" alt="distillation_advantagesNdisadvantages" width="550" />

知识蒸馏的一个优点是可以修改学生模型的架构，使其与教师模型不同。例如，如果教师模型有12个transformer层，那么学生模型不一定必须有12个transformer层。它可能有6个、2个或其他，这种架构变化在量化或剪枝中是做不到的。因此，与其他我们见过的所有方法相比，知识蒸馏在速度上具有最大的潜在增益( potential gain)。

缺点是，它相对更难设置，因为我们需要设置训练数据，这可能是数十亿的token。如果教师模型是一个大模型，那么在其上运行推理可能是一个挑战。所以总的来说，知识蒸馏相对昂贵。根据我以前的经验，这可能需要从头开始训练教师模型所需的5-10%的总计算时间或GPU时间。

这里有一个例子。 **DistilBERT**是一个用知识蒸馏训练的模型，其中BERT是教师模型。在这个模型中，他们将BERT基础模型的大小减少了40%，同时保留了97%的准确度。作者告诉我们他们训练这个模型需要多少GPU以及训练了多长时间，DistilBERT在8个GPU上训练了大约90小时，总共大约700小时的GPU时间。相比之下，与BERT模型类似的**RoBERTa**模型需要在1000个GPU上训练一天，大约24,000小时的GPU时间，或者大约是20倍。所以我们在DistilBERT的例子中看到，使用知识蒸馏训练模型比从头开始(from scratch)训练要快得多，但仍然需要大量的计算。

## 4 工程优化(Engineering Optimizations)

最后一类优化，是将上述优化放到一起，我们将其归类为**工程优化**。

在某个时候，我们需要决定是在CPU上运行模型还是在GPU上运行。无论哪种情况，要使其高效运行都需要在硬件和软件之间进行一些集成。我的意思是，你的硬件可能具有快速运行模型的物理能力，但同时，软件需要知道如何使用硬件能力。

例如，向量化操作(vectorized operations)以**并行**方式运行大型矩阵相乘。GPU当然非常擅长这一点，但CPU实际上也可以使用一些较新的指令集，如AVX2和AVX512进行向量化操作。较新的CPU和GPU模型能够比全精度更快地执行降低精度和混合精度操作，如INT8格式，这对于快速运行量化模型的推理非常有用。

此外，一些GPU具有运行**稀疏内核(sparse kernels)**的硬件能力，这对于运行剪枝神经网络时获得速度增益是必要的。

<img src="pytorch_scaled-dot-product.png" alt="pytorch_scaled-dot-product" width="550" />

另一种优化是**融合内核fused kernels**。例如，PyTorch有一个叫做scaled dot product attention的函数，这个函数结合了通常在transformer架构中的一系列操作，但它执行得非常快。在transformer架构中，我们经常以固定顺序执行一系列串行操作。例如，查询矩阵和键矩阵相乘，然后取softmax，取平方根，然后应用dropout。如果我们将所有这些操作合并为一个单一的操作在GPU上执行，那么这比我们在PyTorch中顺序执行每个指令要快得多。

一种流行的实现方式叫做**FlashAttention**。它不仅将这些操作融合在一起，而且FlashAttention还进行了一些平铺(tiling)和一些根据gpu的内存层次结构进行的优化，以进一步减少执行操作所需的时间。我们可以在右侧的图表上看到，融合的Flash Attention比在PyTorch中简单地顺序执行所有操作要快得多。

<img src="FlashAttention.png" alt="FlashAttention" width="550" />

所有这些可能听起来有点令人生畏，但实际上并不复杂。因为所需要做的就是将训练好的模型转换成某种格式，由我们打算部署的硬件进行优化的推理引擎执行。

我们经常需要使用单独的框架进行训练和推理，因为**训练(training)**神经网络的要求通常与**推理(inference)**期间的要求大不相同。在**训练模型**时，我们需要一个库，它可以在训练期间做相关的事情，比如从磁盘加载(loading)和预处理(pre-processing)数据、执行梯度下降(gradient descent)和反向传播(back propagation)、运行评估(evaluations)、保存检查点(saving checkpoints)等。但在推理期间，这些都不是真正需要的。在**部署模型**进行推理时，要求往往有所不同。模型需要小而快，并且需要在可能与你训练模型不同的硬件上高效运行。因此，通常最好使用不同的库进行推理。

**ONNX Runtime**是两个最受欢迎的推理库之一，它可以在各种不同的硬件上运行存储为**ONNX**格式的模型。如果你喜欢TensorFlow生态系统，**TensorFlowLite**是另外的那一个。

<img src="trainingNinference.png" alt="trainingNinference" width="550" />

## 5 结论

让我们总结一下本视频中所涵盖的内容。

<img src="conclusion_table.png" alt="conclusion_table" width="550" />

首先是量化。量化使用较低精确的数据格式来减小模型大小和延迟。当我们将格式从FP32减少到INT8时，会减少4倍。结合使用低精度执行引擎执行低精度格式能够更快。一个缺点是它可能会导致准确度的损失，尽管希望不会太多。

剪枝是将网络的一些权重设置为零以节省空间和计算。为了使之有效，它需要一个能够执行稀疏神经网络的执行引擎。与量化类似，它可能会导致准确度的损失。

知识蒸馏是我们讨论到的，唯一可以修改模型架构的方法。因此，这种影响取决于因你我们如何修改架构，但其潜在的增益可能比其他任何方法都大得多。知识蒸馏的缺点是训练相对昂贵。

最后是工程优化。这些应该与上述所有方法结合使用。当你使用工程优化时，你应该期望准确度没有损失，因为输出应该是相同的。

<img src="trade-off.png" alt="trade-off" width="550" />

最终，所有这些方法都在**开发成本(development cost)**、**推理成本(inference cost)**和**模型准确度(model accuracy)**之间进行权衡(trade-off)。

量化和在一定程度上，模型剪枝，是减少模型延迟和推理成本的两种不太困难的方法。但对这两者来说，可能会稍微损失模型的准确度。

知识蒸馏有潜力进一步减少模型大小，但它也需要更复杂、更昂贵的训练，特别是对于使用大量GPU训练的大型模型。

感谢观看，我希望你将在你的项目中使用这些技术。如果你发现这个视频有帮助，请不要忘记点赞并订阅我的频道，并在我制作新的和有用的机器学习相关视频时得到通知。这将对我有很大帮助。

再见。
