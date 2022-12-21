import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try_num = 46


data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)

test_out_data = test_out_data.drop(test_out_data.columns[3], axis=1)
out_data = out_data.drop(out_data.columns[3], axis=1)



data = data.reset_index(drop=True)
out_data = out_data.reset_index(drop=True)
test_out_data = test_out_data.reset_index(drop=True)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]
test_out_data = test_out_data.drop(index)

test_out_data_np = test_out_data.to_numpy()

# test_out_data = pd.DataFrame(test_out_data_np, columns=['temp', 'hum', 'sun','ground', 'wind', 'rain', 'snow'])



test_sets_path = 'C:/Users/kogas/Desktop/ai_factory/_data/dataset/test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)
# print(test_out_data)



for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

# print(data)
# print(test_0, test_0.shape)


data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)
# print(data)

test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
test_sets['year'] = test_sets['TIME'].dt.year
test_sets['month'] = test_sets['TIME'].dt.month
test_sets['day'] = test_sets['TIME'].dt.day
test_sets['hour'] = test_sets['TIME'].dt.hour
test_sets['weekday'] = test_sets['TIME'].dt.weekday
pred_index = test_sets['TIME'].values
test_sets = test_sets.drop(['TIME'], axis=1, inplace=False)

x = data
test_sets = test_sets.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','TI_MEAN'], axis=1)
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y
x = pd.concat([x, out_data], axis=1)
x = x.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','TI_MEAN'], axis=1)



# test_sets = pd.concat([test_sets, test_out_data], axis=1, ignore_index=True)

# outdata = x[['temp', 'hum', 'sun', 'ground', 'wind', 'rain', 'snow']]
# target = y[['PIA205B-02A_MIN']]

# outdata_add = sm.add_constant(outdata, has_constant = 'add')

# multi_model = sm.OLS(target, outdata)
# fitted_multi_model = multi_model.fit()
# fitted_multi_model.summary()





 

x = x.to_numpy()
y = y.to_numpy()
test_sets = test_sets.to_numpy()
# test_sets = test_sets.append(test_out_data)
test_sets = np.concatenate([test_sets,test_out_data], axis=1)

print(x.shape) 
print(y.shape)  
print(test_sets.shape)  
print(test_out_data.shape) 

sc = StandardScaler()

x = sc.fit_transform(x)

test_sets = sc.transform(test_sets)



def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

def split_test(seq, size):
    aaa=[]
    for i in range(5508):
        subset = seq[i*size : (i + 1)*size]
        aaa.append([item for item in subset])
    return np.array(aaa)

size = 6 


x = split_x(x, size)
x = x[:-1,:]
y = y[6:,:]
test_sets = split_test(test_sets, size)
print(x.shape)
print(y.shape)
print(test_sets)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=False, random_state=66)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=6, padding='same', input_shape=(6, 17)))
model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=2, padding='same'))
# model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=6, padding='same'))
# model.add(Flatten())
model.add(LSTM(units=128, return_sequences=False))
model.add(tf.keras.layers.LayerNormalization())
model.add(Dense(2))


# model = load_model(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/model_save_{try_num}.csv')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath=f'./_save/ModelCheckPoint/ai_fac_MCP{try_num}.hdf5')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, mode='auto', verbose=1)

model.compile(loss='mae', optimizer='adam')
history = model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_data=(x_test, y_test), verbose=1, callbacks=[es,reduce_lr])

model.save(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/model_save_{try_num}.csv')

loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)

submission_predict = model.predict(test_sets)

# print(submission_predict)

print('loss : ', round(loss,4))  #0.0147
# print('y_predict : ', y_predict)
# print('submission_predict : ', submission_predict)


submission = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_sample.csv', header=0, index_col=0)
# print(submission)
submission.iloc[:, 0:] = submission_predict
# submission = pd.DataFrame(submission_predict, columns=pred_col.columns, index=pred_index)

submission.to_csv(f'C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_{try_num}th.csv')

import matplotlib.pyplot as plt

def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "loss")