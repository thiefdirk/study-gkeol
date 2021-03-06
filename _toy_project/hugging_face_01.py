import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, pipeline, BertTokenizerFast, TFGPT2LMHeadModel, GPT2LMHeadModel
import pandas as pd
import numpy as np
from konlpy.tag import Mecab  
# from keras.preprocessing.sequence import pad_sequences # 0채워서 와꾸맞춰줌

tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")  #("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2", pad_token_id=0)

# generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
datasets = pd.read_csv("D:\study_data\_temp/marrage.csv")

print(tokenizer.morphs(datasets))

# datasets = pd.DataFrame(datasets)

# # x_data = datasets["topic"]
# y_data = datasets["quote"]

# x_data = str(x_data)
# y_data = str(y_data)

text = '결혼'
input_ids= tokenizer.encode(datasets, return_tensors='pt')
input_ids = input_ids[:,1:]

outputs = model.generate(input_ids, max_length=30)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output)
# output=generator(x_data, max_length = 300)
