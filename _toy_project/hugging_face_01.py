import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")

datasets = pd.read_csv("D:\study_data\_temp/marrage.csv", names=["topic", "quote"])

print(datasets)

datasets = pd.DataFrame(datasets)

print(datasets.head(3))

x_data = datasets["topic"]
y_data = datasets["quote"]
print(type(x_data), type(y_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_data = np.array(x_data)
y_data = np.array(y_data)
print(type(x_data), type(y_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_data = str(x_data)
y_data = str(y_data)


# 긍정 1, 부정 0
# labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)

x_data = tokenizer(x_data)
y_data = tokenizer(y_data)

print(x_data)
print(y_data)

def main():
    trainer : Trainer(
        model,
        train_datasets=y_data,
        eval_datasets=y_data(shuffle=False)
    )
    trainer.train()
    
input_ids = tokenizer.encode(y_data, return_tensors='pt')

gen_ids = model.generate(input_ids,
                           max_length=15,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)

generated = tokenizer.decode(gen_ids[0])
print(generated)
