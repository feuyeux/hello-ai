<!-- markdownlint-disable MD033 MD045 -->

# Agent：P（感知）→ P（规划）→ A（行动）

- 感知（Perception）是指Agent从环境中收集信息并从中提取相关知识的能力。
- 规划（Planning）是指Agent为了某一目标而作出的决策过程。
- 行动（Action）是指基于环境和规划做出的动作。

<img src="agent_1.png" style="width:500px" />

类 LangChain 中的各种概念

- Models，也就是我们熟悉的调用大模型API。

- Prompt Templates，在提示词中引入变量以适应用户输入的提示模版。

- Chains，对模型的链式调用，以上一个输出为下一个输入的一部分。

- Agent，能自主执行链式调用，以及访问外部工具。

- Multi-Agent，多个Agent共享一部分记忆，自主分工相互协作。

<img src="agent_2.png" style="width:400px" />

Agent框架

- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
  - <https://docs.agpt.co/>
  - <https://github.com/Significant-Gravitas/Nexus/wiki>

- [GPT-Engineer](https://github.com/AntonOsika/gpt-engineer)
- [MetaGPT](https://github.com/geekan/MetaGPT)
  - <https://docs.deepwisdom.ai/main/zh/guide/get_started/introduction.html>

<https://www.breezedeus.com/article/ai-agent-part1>

<https://www.breezedeus.com/article/ai-agent-part2>

<https://www.breezedeus.com/article/ai-agent-part3>
