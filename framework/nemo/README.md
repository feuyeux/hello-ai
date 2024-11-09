# NVIDIA NeMo Framework

<https://github.com/NVIDIA/NeMo>

<https://www.nvidia.com/en-us/ai-data-science/products/nemo/>

NVIDIA NeMo™ is an end-to-end platform for developing custom generative AI—including large language models (LLMs), multimodal, vision, and [speech AI](https://www.nvidia.com/en-us/ai-data-science/solutions/speech-ai/) —anywhere. Deliver enterprise-ready models with precise data curation, cutting-edge customization, retrieval-augmented generation (RAG), and accelerated performance.

[NVIDIA NeMo Framework Developer Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest)

![nemo-llm-mm-stack](nemo-llm-mm-stack.png)

1 [Data Curation](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html) 数据监管

**NeMo Curator** /kjʊəˈreɪtə(r)/ is a Python library that includes a suite of data-mining modules. These modules are optimized for GPUs and designed to scale, making them ideal for curating natural language data to train LLMs. With NeMo Curator, researchers in Natural Language Processing (NLP) can efficiently extract high-quality text from extensive raw web data sources.

**Model Training and Customization**

NeMo Framework provides a comprehensive set of tools for the **efficient training and customization** of [LLMs](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/index.html#llm-index) and [Multimodal models](https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/index.html#multimodal-index). This includes the setup of the compute cluster, data downloading, and model hyperparameter selection. Each model and task come with default configurations that are regularly tested. However, these configurations can be adjusted to train on new datasets or test new model hyperparameters. For customization, NeMo Framework supports not only fully Supervised Fine-Tuning (SFT), but also a range of Parameter Efficient Fine-Tuning (PEFT) techniques. These techniques include Ptuning, LoRA, Adapters, and IA3. They typically achieve nearly the same accuracy as SFT, but at a fraction of the computational cost.

2 [Model Alignment](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/index.html)

Part of the framework, NeMo-Aligner is a **scalable toolkit for efficient model alignment**. NeMo-Aligner, a component of NeMo Framework, is a scalable toolkit designed for effective model alignment. The toolkit supports [Supervised Finetuning (SFT)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/sft.html#model-alignment-by-supervised-fine-tuning-sft) and other state-of-the-art (SOTA) 使用最先进技术的 model alignment algorithms such as [SteerLM](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/steerlm.html), [DPO](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html), and [Reinforcement Learning from Human Feedback (RLHF)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/rlhf.html). These algorithms enable users to align language models to be more safe and helpful.

3 [Launcher](https://docs.nvidia.com/nemo-framework/user-guide/latest/launcherguide/index.html#launcherguide-index)

NeMo Launcher streamlines your experience with the NeMo Framework by providing an **intuitive interface for constructing comprehensive workflows**. This allows for effective organization and management of experiments across different environments. Based on the Hydra framework, NeMo Launcher enables users to easily create and modify hierarchical configurations using both configuration files and command-line arguments. It simplifies the process of initiating large-scale training, customization, or alignment tasks. These tasks can be run locally (supporting single node), on [NVIDIA Base Command Manager](https://www.nvidia.com/en-in/data-center/base-command/) (Slurm), or on cloud providers such as AWS, Azure, and Oracle Cloud Infrastructure (OCI). This is all made possible through Launcher scripts, eliminating the need for writing any code.

4 [Model Inference](https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#deploy-nemo-framework-models)

NeMo Framework seamlessly integrates with **enterprise-level model deployment** tools through [NVIDIA NIM](https://www.nvidia.com/en-gb/launchpad/ai/generative-ai-inference-with-nim). This integration is powered by NVIDIA TensorRT-LLM and NVIDIA Triton Inference Server, ensuring **optimized and scalable inference**.

----

- <https://github.com/NVIDIA/NeMo>
- <https://github.com/NVIDIA/NeMo-Curator>
- <https://github.com/NVIDIA/NeMo-Aligner>
- <https://github.com/NVIDIA/NeMo-Framework-Launcher>
