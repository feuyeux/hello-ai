# gTTS

```sh
python -m venv gtts_venv
source gtts_venv/bin/activate
pip install --upgrade pip
pip install gTTS
```

```sh
gtts-cli 'hello' --output hello.mp3
open hello.mp3
```

hello.py

```python
from gtts import gTTS
tts_en = gTTS('hello', lang='en')
tts_fr = gTTS('bonjour', lang='fr')
tts_cn = gTTS('你好', lang='zh-CN')

with open('hello_bonjour.mp3', 'wb') as f:
    tts_en.write_to_fp(f)
    tts_fr.write_to_fp(f)
    tts_cn.write_to_fp(f)
```

```sh
python hello.py 
open hello_bonjour.mp3
```
