# https: // py-googletrans.readthedocs.io/en/latest/
from googletrans import Translator
from gtts import gTTS
import os
import speech_recognition as spr
import time

translator = Translator()
recognizer = spr.Recognizer()
mc = spr.Microphone()

chinese_text = None

try:
    with mc as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Speak to translation")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        audio = recognizer.listen(source, timeout=2)

    recognizer.pause_threshold = 1.0
    chinese_text = recognizer.recognize_google(audio, language='zh-CN')
    print("chinese_text: ", chinese_text)
except spr.UnknownValueError:
    print("无法理解音频")
except spr.RequestError:
    print(f"无法从服务中获取结果")
except spr.WaitTimeoutError:
    print("识别超时")
except spr.TranscriptionNotReady:
    print("识别未准备好")
except Exception as e:
    print(f"发生错误: {e}")

if chinese_text is None:
    chinese_text = """
    爸爸和我在妈妈的厨房里给姐姐做早餐。
    """

languages = {
    'en': 'en',
    'de': 'de',
    'fr': 'fr',
    'es': 'es',
    'ru': 'ru',
    'el': 'el',
    'ar': 'ar',
    'hi': 'hi',
    'ja': 'ja',
    'ko': 'ko'
}

for lang_code, lang_dest in languages.items():
    translated = translator.translate(
        chinese_text, src='zh-cn', dest=lang_dest)
    tts_text = translated.text
    print(f"Translation[{lang_dest}]: {tts_text}")
    # print(f"Translation: {tts_text} Pronunciation: {translated.pronunciation}")
    tts = gTTS(tts_text, lang=lang_code, slow=True)
    tts.save(f"tts_result_{lang_code}.mp3")
    time.sleep(1)

os.system("start tts_result_en.mp3")
