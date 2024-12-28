# Hello Translator

```sh
conda create -n trans python=3.10
conda activate trans
python -m ensurepip --upgrade
pip install -r requirements.txt
```

```sh
conda activate trans
python hello_trans.py
```

[ISO 639 language codes](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes)

|语言|编码|
|----|----|
|中文|zh|
|英语|en|
|德语|de|
|法语|fr|
|西班牙语|es|
|俄语|ru|
|希腊语|el|
|阿拉伯语|ar|
|印地语|hi|
|日语|ja|
|韩语|ko|

```python
print(googletrans.LANGUAGES)
```
