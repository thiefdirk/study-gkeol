import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, pipeline, BertTokenizerFast, TFGPT2LMHeadModel, GPT2LMHeadModel
import pandas as pd
import numpy as np

save_directory = "saved"


tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")  #("skt/kogpt2-base-v2")

model = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2", pad_token_id=0)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer = BertTokenizerFast.from_pretrained(save_directory)  #("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained(save_directory, pad_token_id=0)