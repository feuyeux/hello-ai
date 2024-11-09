import asyncio
import time

import torch
import torchaudio

import ChatTTS

chat = ChatTTS.Chat()
chat.load(compile=False)  # Set to True for better performance


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
    # Assuming `wavs` is a list of numpy arrays
    wavs = chat.infer(
        text_input,
        skip_refine_text=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )
    # Convert the first element of `wavs` to a PyTorch tensor
    audio_tensor = torch.from_numpy(wavs[0])
    # Ensure the tensor is 2D
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    # Save the audio tensor to a file
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
