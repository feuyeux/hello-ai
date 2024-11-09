# Llama

<https://llama.meta.com>

## Llama 3.1论文精读

- [论文 The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783) 20240731
- [AI 论文精读列表](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=32744)
- [1. 导言](https://www.bilibili.com/video/BV1WM4m1y7Uh) 20240731
- [2. 预训练数据 哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1u142187S5) 20240805
- [3. 模型 哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Q4421Z7Tj) 20240813
- [4. 训练infra 哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1b4421f7fa) 20240828
- [5. 模型训练过程 哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1c8HbeaEXi) 20240904
- [从Llama 3报告出发的LLM基本技术整理](https://zhuanlan.zhihu.com/p/713794852)

### 1 Introduction 引言

### 2 General Overview 总体概述

### 3 Pre-Training 预训练

#### 3.1 Pre-Training Data 预训练数据

MinHash

PII(Personally Identifiable Information) and safety filtering

#### 3.2 Model Architecture 模型架构

GQA group  kvcache

#### 3.3 Infrastructure, Scaling, and Efficiency 基础设施、扩展和效率

Scaling Law

#### 3.4 Training Recipe 训练配方

### 4 Post-Training 后训练

- SFT(supervised finetuning)
- RS(rejection sampling)
- DPO(Direct preference optimization)

#### 4.1 Modeling 建模

#### 4.2 Post-training Data 后训练数据

#### 4.3 Capabilities 能力

##### 4.3.6 Factuality 事实性

### 5 Results 结果

#### 5.1 Pre-trained Language Model 预训练语言模型

#### 5.2 Post-trained Language Model 后训练语言模型

#### 5.3 Human Evaluations 人类评估

#### 5.4 Safety 安全性

### 6 Inference 推理

#### 6.1 Pipeline Parallelism 流水线并行

#### 6.2 FP8 Quantization FP8量化

### 7 Vision Experiments 视觉实验

#### 7.1 Data 数据

#### 7.2 Model Architecture 模型架构

#### 7.3 Model Scaling 模型缩放

#### 7.4 Pre-training 预训练

#### 7.5 Post-Training 后训练

#### 7.6 Image Recognition Results 图像识别结果

#### 7.7 Video Recognition Results 视频识别结果

### 8 Speech Experiments 语音实验

#### 8.1 Data 数据

#### 8.2 Model Architecture 模型架构

#### 8.3 Training Recipe 训练配方

#### 8.4 Speech Understanding Results 语音理解结果

#### 8.5 Speech Generation Results 语音生成结果

### 9 Related Work 相关工作

#### 9.1 Language 语言

#### 9.2 Multimodality 多模态

### 10 Conclusion 结论

### Contributors and Acknowledgements 贡献者和致谢

#### Core Contributors 核心贡献者

#### Contributors 贡献者

#### Acknowledgements 致谢

### References 参考文献
