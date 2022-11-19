# 3-dim dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
# randam np.array (1316664, 1)

a = pd.DataFrame(np.random.rand(48, 1))

# 7배수번째 데이터들 추출후 5508번째 행까지 추출

b = a.iloc[::7, :] # ::7은 7의 배수번째 데이터 추출 

# lstm stateful = True 모델

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(1, batch_input_shape=(1, 7, 1), stateful=True))
model.add(tf.keras.layers.Dense(1))