# 文言文SFT

这位博主也是个兴致勃然的家伙，他基于Llama3-8B进行和微调和量化，生成了自己的文言文大模型。

他将这个大模型跑在LM Studio上，将王小波作品中的一句话翻译成了文言文。

![](2024-09-30_10.33.20.png)

视频地址：<https://www.youtube.com/watch?v=Tq6qPw8EUVg>

1. 语料：<https://github.com/NiuTrans/Classical-Modern>
2. 训练集生成脚本：<https://gist.github.com/lanesky/6092906644c36d16ad39df3ac6d623d2>
3. 文言文模型微调和量化脚本：<https://colab.research.google.com/drive/1Ne068_p8ZbJt93DdXT2zlxcTIYUuE_zT?usp=sharing>
4. 预训练模型：Llama 3.1 (8B)
5. 工具
   - Colab: <https://colab.research.google.com/>
   - HuggingFace: <https://huggingface.co/>
   - unsloth: <https://github.com/unslothai/unsloth>
   - llama.cpp: <https://github.com/ggerganov/llama.cpp>
   - LM studio: <https://lmstudio.ai/>

博主老哥思路清晰、卷面工整，完整的流程示意如下

![](drawio.png)
