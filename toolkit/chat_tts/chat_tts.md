<!-- markdownlint-disable MD033 -->
# ChatTTS

[ChatTTS](https://chattts.com/) - Text-to-Speech for Conversational Scenarios

<https://github.com/libukai/Awesome-ChatTTS>

## Install on WSL

```sh
sudo apt-get install -y build-essential libfst-dev
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
```

update requirements.txt

```properties
Cython
numpy<2.0.0
numba
torch>=2.1.0
torchaudio
tqdm
vector_quantize_pytorch
transformers>=4.41.1
vocos
IPython
gradio
pybase16384
pynini==2.1.6.post1; sys_platform == 'linux'
WeTextProcessing; sys_platform == 'linux'
nemo_text_processing; sys_platform == 'linux'
av
pydub
```

```sh
conda create -n chattts
conda activate chattts
pip install -r requirements.txt
pip install PySoundFile
```

## Install on macOS

```sh
$ conda create -n chattts python=3.8 -y
$ conda activate chattts
$ conda env list
$ which python
/Users/hanl5/miniconda3/envs/chattts/bin/python
$ pip install -r requirements.txt
```

## Run

```sh
conda activate chattts
python examples/cmd/run.py "Welcome, my friend." "欢迎你，我的朋友。"
```

![run_example](run_example.png)

## Sample

```python
import asyncio
import time
import torch
import torchaudio
import ChatTTS

chat = ChatTTS.Chat()
chat.load(compile=False)

def hello(text_input, audio_output='output.wav'):
    # sampled speaker
    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=.2,
    )
    # For sentence level manual control.
    # use oral_(0-9), laugh_(0-2), break_(0-7)
    # to generate special token in text to synthesize.
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_6][laugh_1][break_5]',
        top_P=0.5,
        top_K=20,
    )
    wavs = chat.infer(
        text_input,
        skip_refine_text=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )
    audio_tensor = torch.from_numpy(wavs[0])
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    torchaudio.save(audio_output, audio_tensor, 24000, format='wav')

texts = [
    "这是一个[lbreak][laugh]为对话应用设计的文本转语音模型。可以精确控制诸如[lbreak]，[laugh]笑声[laugh]，停顿[uv_break]，以及语调等[lbreak]韵律元素。",
]

async def process_text(text, output_name):
    output_file = f"{output_name}_{int(time.time())}.wav"
    print(f"Processing: {text} {output_file}")
    start_time = time.time()
    hello(text, output_file)
    end_time = time.time()
    print(f"Done: {end_time - start_time} seconds")

async def main():
    tasks = []
    for text in texts:
        for j in range(3):
            tasks.append(
                process_text(text, f"hello_chat_tts_output_{j}")
            )
    await asyncio.gather(*tasks)

asyncio.run(main())
```

```sh
conda activate chattts
python hello_chat_tts.py
```

```sh
Processing: 这是一个[lbreak]为对话应用设计的[laugh]文本转语音模型。可以精确控制诸如[lbreak] 笑声 [laugh] 停顿[uv_break]以及语调等韵律元素。 hello_chat_tts_output_0_1728917169.wav
code:   0%|                                                                                                                                                            | 1/2048(max) [00:00,  2.36it/s]We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
code:  19%|█████████████████████████████▎                                                                                                                            | 390/2048(max) [00:10, 37.56it/s]
Done: 11.80508542060852 seconds
Processing: 这是一个[lbreak]为对话应用设计的[laugh]文本转语音模型。可以精确控制诸如[lbreak] 笑声 [laugh] 停顿[uv_break]以及语调等韵律元素。 hello_chat_tts_output_1_1728917181.wav
code:  20%|███████████████████████████████▏                                                                                                                          | 414/2048(max) [00:06, 59.81it/s]
Done: 6.972497463226318 seconds
Processing: 这是一个[lbreak]为对话应用设计的[laugh]文本转语音模型。可以精确控制诸如[lbreak] 笑声 [laugh] 停顿[uv_break]以及语调等韵律元素。 hello_chat_tts_output_2_1728917188.wav
code:  21%|████████████████████████████████▎                                                                                                                         | 429/2048(max) [00:07, 60.24it/s]
Done: 7.176044464111328 seconds
```

## Audio Samples
<audio controls src="hello_chat_tts_output_1_1728918488.wav" title="output_0"></audio>
<audio controls src="hello_chat_tts_output_0_1728918475.wav" title="output_1"></audio>
<audio controls src="hello_chat_tts_output_2_1728918098.wav" title="output_2"></audio>
