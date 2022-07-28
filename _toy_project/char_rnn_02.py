# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from konlpy.tag import Okt 
import nltk
from nltk.tokenize import word_tokenize
# test=tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# print(test)
okt=Okt()  
path = 'D:\study_data\_temp/'

datasets = pd.read_csv("/Users/gkeol/Documents/study-gkeol/marrage_orgn.csv",
                   encoding="utf-8", names=["topic", "quote"])


datasets['topic'] = datasets['topic'].str.strip()

print(f'Length of text: {len(datasets)} characters')

print(datasets[:250])

train = [(quote,topic)  for index,
         (quote,topic)  in datasets.iterrows()]
print(okt.morphs(datasets))
print(okt.pos(datasets))  
print(okt.nouns(datasets))  
allword = set(word for sentence in train
              for word in word_tokenize(sentence[0]))

