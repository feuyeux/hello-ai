# translation transformer

<https://pytorch.org/tutorials/beginner/translation_transformer.html>

```sh
# translation_transformer.py

python3 -m venv transformer_env

source transformer_env/bin/activate

pip3 install torch torchtext spacy portalocker
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

python translation_transformer.py

# run .ipynb
pip3 install matplotlib

deactivate
```
