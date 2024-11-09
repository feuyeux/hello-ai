# MUSIC

<https://github.com/hf-lin/ChatMusician>

```sh
python3 -m venv music_env
source music_env/bin/activate
```

## chatMusician

```sh
cd $HOME/coding/feuyeux/ChatMusician
pip install -r requirements.txt

brew install abcmidi

pip install gradio

python model/infer/chatmusician_web_demo.py -c "m-a-p/ChatMusician" --server_port 8888
```

## musicLM

```sh
pip install musiclm-pytorch

```
