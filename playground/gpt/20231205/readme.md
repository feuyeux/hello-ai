# Generating article using AI and translate in any language (GPT2 and MBart)

<https://www.linkedin.com/pulse/generating-article-using-ai-translate-any-language-gpt2-kundi/>

```sh
# macos
python3 -m venv gpt_env

source gpt_env/bin/activate
pip3 install torch transformers sentencepiece ipywidgets protobuf

# windows
python -m venv gpt_env

gpt_env/Scripts/activate.bat
python -m pip install --upgrade pip
pip install torch transformers sentencepiece ipywidgets protobuf

touch hello.ipynb

deactivate
```

[MBart and MBart-50](https://huggingface.co/docs/transformers/model_doc/mbart)

[Translation Task](https://huggingface.co/docs/transformers/tasks/translation)

<https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt>

Languages covered

1. 阿拉伯语 (ar_AR) Arabic (ar_AR)
1. 捷克语 (cs_CZ) Czech (cs_CZ)
1. 德语 (de_DE) German (de_DE)
1. 英语 (en_XX) English (en_XX)
1. 西班牙语 (es_XX) Spanish (es_XX)
1. 爱沙尼亚语 (et_EE) Estonian (et_EE)
1. 芬兰语 (fi_FI) Finnish (fi_FI)
1. 法语 (fr_XX) French (fr_XX)
1. 古吉拉特语 (gu_IN) Gujarati (gu_IN)
1. 印地语 (hi_IN) Hindi (hi_IN)
1. 意大利语 (it_IT) Italian (it_IT)
1. 日语 (ja_XX) Japanese (ja_XX)
1. 哈萨克语 (kk_KZ) Kazakh (kk_KZ)
1. 韩语 (ko_KR) Korean (ko_KR)
1. 立陶宛语 (lt_LT) Lithuanian (lt_LT)
1. 拉脱维亚语 (lv_LV) Latvian (lv_LV)
1. 缅甸语 (my_MM) Burmese (my_MM)
1. 尼泊尔语 (ne_NP) Nepali (ne_NP)
1. 荷兰语 (nl_XX) Dutch (nl_XX)
1. 罗马尼亚语 (ro_RO) Romanian (ro_RO)
1. 俄语 (ru_RU) Russian (ru_RU)
1. 僧伽罗语 (si_LK) Sinhala (si_LK)
1. 土耳其语 (tr_TR) Turkish (tr_TR)
1. 越南语 (vi_VN) Vietnamese (vi_VN)
1. 中文 (zh_CN) Chinese (zh_CN)
1. 南非荷兰语 (af_ZA) Afrikaans (af_ZA)
1. 阿塞拜疆语 (az_AZ) Azerbaijani (az_AZ)
1. 孟加拉语 (bn_IN) Bengali (bn_IN)
1. 波斯语 (fa_IR) Persian (fa_IR)
1. 希伯来语 (he_IL) Hebrew (he_IL)
1. 克罗地亚语 (hr_HR) Croatian (hr_HR)
1. 印度尼西亚语 (id_ID) Indonesian (id_ID)
1. 格鲁吉亚语 (ka_GE) Georgian (ka_GE)
1. 高棉语 (km_KH) Khmer (km_KH)
1. 马其顿语 (mk_MK) Macedonian (mk_MK)
1. 马拉雅拉姆语 (ml_IN) Malayalam (ml_IN)
1. 蒙古语 (mn_MN) Mongolian (mn_MN)
1. 马拉地语 (mr_IN) Marathi (mr_IN)
1. 波兰语 (pl_PL) Polish (pl_PL)
1. 普什图语 (ps_AF) Pashto (ps_AF)
1. 葡萄牙语 (pt_XX) Portuguese (pt_XX)
1. 瑞典语 (sv_SE) Swedish (sv_SE)
1. 斯瓦希里语 (sw_KE) Swahili (sw_KE)
1. 泰米尔语 (ta_IN) Tamil (ta_IN)
1. 泰卢固语 (te_IN) Telugu (te_IN)
1. 泰语 (th_TH) Thai (th_TH)
1. 塔加洛语 (tl_XX) Tagalog (tl_XX)
1. 乌克兰语 (uk_UA) Ukrainian (uk_UA)
1. 乌尔都语 (ur_PK) Urdu (ur_PK)
1. 科萨语 (xh_ZA) Xhosa (xh_ZA)
1. 加利西亚语 (gl_ES) Galician (gl_ES)
1. 斯洛文尼亚语 (sl_SI) Slovene (sl_SI)
