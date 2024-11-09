#!/bin/bash
python3 -m venv trans_env
source trans_env/bin/activate
pip3 install torch transformers sentencepiece protobuf ipywidgets ipykernel
pip3 install accelerate
pip install bitsandbytes

#
pip install --upgrade embedchain
pip install --upgrade "embedchain[dataloaders]"
