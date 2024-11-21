# GGUF(GPT-Generated Unified Format)

由llama.cpp的创始人Georgi Gerganov定义发布的一种大模型文件格式。一个GGUF文件包括文件头、元数据键值对和张量信息等。这些组成部分共同定义了模型的结构和行为。

![gguf-spec.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png)

同时，GGUF支持多种数据类型，如整数、浮点数和字符串等。这些数据类型用于定义模型的不同方面，如结构、大小和参数。

GGUF文件具体的组成信息如下所示：

1. **文件头 (Header)**     
     - **作用**：包含用于识别文件类型和版本的基本信息。
     - **内容**：
       - `Magic Number`：一个特定的数字或字符序列，用于标识文件格式。
       - `Version`：文件格式的版本号，指明了文件遵循的具体规范或标准。
1. **元数据key-value对 (Metadata Key-Value Pairs)**
     - **作用**：存储关于模型的额外信息，如作者、训练信息、模型描述等。
     - **内容**：
       - `Key`：一个字符串，标识元数据的名称。
       - `Value Type`：数据类型，指明值的格式（如整数、浮点数、字符串等）。
       - `Value`：具体的元数据内容。
1. **张量计数器 (Tensor Count)**
       - **作用**：标识文件中包含的张量（Tensor）数量。
       - **内容**：
         - `Count`：一个整数，表示文件中张量的总数。
1. **张量信息 (Tensor Info)**
     - **作用**：描述每个张量的具体信息，包括形状、类型和数据位置。
     - **内容**：
       - `Name`：张量的名称。
       - `Dimensions`：张量的维度信息。
       - `Type`：张量数据的类型（如：浮点数、整数等）。
       - `Offset`：指明张量数据在文件中的位置。
1. **对齐填充 (Alignment Padding)**
    - **作用**：确保数据块在内存中正确对齐，有助于提高访问效率。
    - **内容**：通常是一些填充字节，用于保证后续数据的内存对齐。
1. **张量数据 (Tensor Data)**
    - **作用**：存储模型的实际权重和参数。
    - **内容**：
      - `Binary Data`：模型的权重和参数的二进制表示。
1. **端序标识 (Endianness)**
    - **作用**：指示文件中数值数据的字节顺序（大端或小端）。
    - **内容**：通常是一个标记，表明文件遵循的端序。
1. **扩展信息 (Extension Information)**
    - **作用**：允许文件格式未来扩展，以包含新的数据类型或结构。
    - **内容**：可以是新加入的任何额外信息，为将来的格式升级预留空间。

https://huggingface.co/docs/hub/en/gguf

