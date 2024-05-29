import os
import json
import pandas as pd
from src.classifiers import FastAnswerClassifier
from transformers import T5Tokenizer, T5ForConditionalGeneration



t5_tokenizer = T5Tokenizer.from_pretrained('ai-forever/ruT5-large')
t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join("data", 'models_bss')).to("cuda")


classifier = FastAnswerClassifier(t5_model, t5_tokenizer)
