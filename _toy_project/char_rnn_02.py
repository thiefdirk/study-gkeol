# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from konlpy.tag import Okt 
# test=tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# print(test)
okt=Okt()  
path = 'D:\study_data\_temp/'

datasets = pd.read_csv("/Users/gkeol/Documents/study-gkeol/marrage.csv")

print(f'Length of text: {len(datasets)} characters')

print(datasets[:250])

vocab = sorted(set(datasets))

print(f'{len(vocab)} unique characters')

print(okt.morphs(datasets[1]))
print(okt.pos(datasets))  
print(okt.nouns(datasets))  

