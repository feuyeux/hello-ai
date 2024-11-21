# BERT base model

<https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment>

## install

```sh
python -m venv bert-base
source bert-base/bin/activate

# windows
python.exe -m pip install --upgrade pip
# macos
pip install --upgrade pip
```

```sh
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:36:51_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.5, V12.5.40
Build cuda_12.5.r12.5/compiler.34177558_0
```

```sh
# https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U transformers
pip install numpy==1.26.4
```

## Run (Feature -> LLM -> Label)

```sh
python emotion_classification.py
```

```sh
|torch: 2.4.0+cu124 |cuda is available: True |transformers: 4.44.0 |numpy: 1.26.4
Sentence: I hate road trips, they are always exhausting.
Predicted Emotion: Very negative
Predicted Probability: 0.4253181517124176

Sentence: I'm not sure about camping, it sounds uncomfortable.
Predicted Emotion: Negative
Predicted Probability: 0.4905439615249634

Sentence: A trip sounds okay, but I don't have any strong feelings about it.
Predicted Emotion: Neutral
Predicted Probability: 0.7353528738021851

Sentence: A weekend getaway sounds fun!
Predicted Emotion: Positive
Predicted Probability: 0.5042614936828613

Sentence: I love outdoor adventures with friends it's going to be amazing!”
Predicted Emotion: Very positive
Predicted Probability: 0.7617835402488708
```
