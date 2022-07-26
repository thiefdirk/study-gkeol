import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer, pipeline, BertTokenizerFast, TFGPT2LMHeadModel, GPT2LMHeadModel
import pandas as pd
import numpy as np

tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")  #("skt/kogpt2-base-v2")

model = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2", pad_token_id=0)

# generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
datasets = pd.read_csv("D:\study_data\_temp/marrage.csv", names=["topic", "quote"])

datasets = pd.DataFrame(datasets)

x_data = datasets["topic"]
y_data = datasets["quote"]

x_data = np.array(x_data)
y_data = np.array(y_data)

print(type(x_data[0]), type(y_data[0])) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

X = torch.tensor(x_data, dtype=torch.float32).to("cuda:0")
Y = torch.tensor(y_data, dtype=torch.int64).to("cuda:0")

ds = TensorDataset(X, Y)

# x_data = str(x_data)
# datasets = dict.fromkeys(datasets,0)

train_dataloader = data.DataLoader(datasets)

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataloader,
    tokenizer=tokenizer
)

trainer.train()
# x_data = datasets["topic"]
# y_data = datasets["quote"]

# x_data = str(x_data)
# y_data = str(y_data)

text = '결혼'
input_ids= tokenizer.encode(text, return_tensors='pt')
input_ids = input_ids[:,1:]

outputs = model.generate(input_ids, max_length=30)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output)
# output=generator(x_data, max_length = 300)
