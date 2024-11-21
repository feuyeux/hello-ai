import warnings
import transformers
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
warnings.filterwarnings('ignore')

print(
    "|torch:", torch.__version__,
    "|cuda is available:", torch.cuda.is_available(),
    "|transformers:", transformers.__version__,
    "|numpy:", np.__version__
)

# https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
tokenizer = BertTokenizer.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment', num_labels=5)
emotion_labels = ["Very negative", "Negative",
                  "Neutral", "Positive", "Very positive"]


def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    return inputs


sentences = ["I hate road trips, they are always exhausting.",
             "I'm not sure about camping, it sounds uncomfortable.",
             "A trip sounds okay, but I don't have any strong feelings about it.",
             "A weekend getaway sounds fun!",
             "I love outdoor adventures with friends it's going to be amazing!”"]


def predict(sentence):
    inputs = preprocess_text(sentence)
# 使用模型进行预测
    with torch.no_grad():
        outputs = model(**inputs)
# 获取预测结果
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).tolist()[0]
    predicted_emotion = emotion_labels[predicted_class]
    predicted_probability = predictions[0][predicted_class].item()
# 打印结果
    print(f"Sentence: {sentence}")
    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Predicted Probability: {predicted_probability}\n")


for sentence in sentences:
    predict(sentence)
