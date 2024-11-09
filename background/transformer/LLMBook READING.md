# LLMBook

语言模型的四个发展阶段

1 统计语言模型（Statistical Language Model, SLM）

使用马尔可夫假设（Markov Assumption）来建立语言序列的预测模型 𝑛-gram

2神经语言模型（Neural Language Model, NLM）

循环神经网络（Recurrent Neural Networks, RNN）

分布式词表示（Distributed Word Representation）又称为“词嵌入”（Word Embedding）。使用低维稠密向量来表示词汇的语义，这种基于隐含语义特征表示的语言建模方法为自然语言处理任务提供了一种较为通用的解决途径。word2vec 是一个具有代表性的词嵌入学习模型，它构建了一个简化的浅层神经网络来学习分布式词表示，所学习到的词嵌入可以用作后续任务的语义特征提取器

3 预训练语言模型（Pre-trained Language Model, PLM）

- ELMo 是一个早期的代表性预训练语言模型，提出使用大量的无标注数据训练双向 LSTM（Bidirectional LSTM, biLSTM）网络，预训练完成后所得到的 biLSTM 可以用来学
习上下文感知的单词表示
- 谷歌提出了基于自注意力机制（Self-Attention）的 Transformer 模型，通过自注意力机制建模长程序列关系。基于 Transformer 架构，谷歌进一步提出了预训练语言模型 BERT，采用了仅有编码器的 Transformer 架构，并通过在大规模无标注数据上使用专门设计的预训练任务来学习双向语言模型
- OpenAI也迅速采纳了 Transformer 架构，将其用于 GPT-1 的训练。与 BERT 模型不同的是，GPT-1 采用了仅有解码器的 Transformer 架构，以及基于下一个词元预测的预训练任务进行模型的训练

编码器架构被认为更适合去解决自然语言理解任务（如完形填空等）
解码器架构更适合解决自然语言生成任务（如文本摘要等）。

4 大语言模型（Large Language Model, LLM）

研究人员发现，通过规模扩展（如增加模型参数规模或数据规模）通常会带来下游任务的模型性能提升，这种现象通常被称为“扩展法则”（Scaling Law）。这些大规模的预训练语言模型在解决复杂任务时表现出了与小型预训练语言模型（例如 330M 参数的 BERT 和 1.5B 参数的 GPT-2）不同的行为。例如，GPT-3 可以通过“上下文学习”（In-Context Learning, ICL）的方式来利用少样本数据解决下游任务，而 GPT-2 则不具备这一能力。这种大模型具有但小模型不具有的能力通常被称为“涌现能力”（Emergent Abilities）。为了区别这一能力上的差异，学术界将这些大型预训练语言模型命名为“大语言模型”（Large Language Model, LLM）

“顿悟”（Grokking）

OpenAI 从参数、数据、算力三个方面深入地研究了规模扩展对于模型性能所带来的影响，建立了定量的函数关系，称之为“扩展法则”（Scaling Law）

“3 H 对齐标准”，即 Helpfulness（有用性）、Honesty（诚实性）和 Harmlessness（无害性）

<https://github.com/RUCAIBox/LLMSurvey>

指令微调很难教会大语言模型预训练阶段没有学习到的知识与能力，它主要起到了对于模型能力的激发作用，而不是知识注入作用

3D 并行策略实际上是三种常用的并行训练技术的组合，即数据并行（Data Parallelism）、流水线并行（Pipeline Parallelism）和张量并行（Tensor Parallelism）。
