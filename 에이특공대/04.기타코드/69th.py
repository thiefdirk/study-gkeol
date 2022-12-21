import pstats
from re import VERBOSE
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Input
from tensorflow.python.keras.models import load_model, Model
# from tensorflow.python.keras.metrics import MeanAbs
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try_num = 69


data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)





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
# x = x.drop(['PIA205B-02A_MIN', 'PIA205B-02A_MAX'], axis=1)   #종속변수
x = x.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','FI_SUM'], axis=1)   #독립변수
test_sets = test_sets.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','FI_SUM'], axis=1)
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y
x = pd.concat([x, out_data], axis=1)

# print(x.shape) 
# print(y.shape)  

x = x.to_numpy()
y = y.to_numpy()
test_sets = test_sets.to_numpy()
# test_sets = test_sets.append(test_out_data)
test_sets = np.concatenate([test_sets,test_out_data], axis=1)



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

# model = Sequential()
# model.add(Conv1D(filters=128, kernel_size=2, padding='same', input_shape=(6, 18)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=2, padding='same'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same'))
# model.add(Flatten())
# model.add(Dense(50, activation='selu'))
# model.add(Dense(30, activation='selu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='elu'))
# model.add(Dense(2))


input_layer = Input((6,18))
Conv1D_1 = Conv1D(filters=128, kernel_size=2, padding='same')(input_layer)
MaxP_1 = MaxPooling1D(pool_size=2)(Conv1D_1)
Conv1D_2 = Conv1D(filters=64, kernel_size=2, padding='same')(MaxP_1)
MaxP_2 = MaxPooling1D(pool_size=2)(Conv1D_2)
Conv1D_3 = Conv1D(filters=32, kernel_size=2, padding='same')(MaxP_2)

# gap
GAP = GlobalAveragePooling1D()(Conv1D_3)
Dense_1_1 = Dense(50, activation = 'selu')(GAP)
# Dense_1_2 = Dense(30, activation = 'selu')(Dense_1_1)
# Dense_1_3 = Dense(32, activation = 'relu')(Dense_1_2)
# Dense_1_4 = Dense(16, activation = 'relu')(Dense_1_3)
# Dense_1_5 = Dense(8, activation = 'elu')(Dense_1_4)
GAPout = Dense(2, name=('gapout'))(Dense_1_1)
# gmp
GMP = GlobalMaxPooling1D()(Conv1D_3)
Dense_2_1 = Dense(50, activation = 'selu')(GMP)
# Dense_2_2 = Dense(30, activation = 'selu')(Dense_2_1)
# Dense_2_3 = Dense(32, activation = 'relu')(Dense_2_2)
# Dense_2_4 = Dense(16, activation = 'relu')(Dense_2_3)
# Dense_2_5 = Dense(8, activation = 'elu')(Dense_2_4)
GMPout = Dense(2, name=('gmpout'))(Dense_2_1)

model = Model(inputs=input_layer, outputs=[GAPout, GMPout])


# model.load_model(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/model_save_{try_num}.pth')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath=f'./_save/ModelCheckPoint/ai_fac_MCP{try_num}.hdf5',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=100, factor=0.5, mode='auto', verbose=1)

model.compile(loss={'gapout':'mae','gmpout':'mae'}, optimizer='adam', loss_weights=[0.5,0.5])
history = model.fit(x_train, y_train, epochs=3000, batch_size=256, validation_split=0.2, verbose=1, callbacks=[es,reduce_lr,cp])

model.save(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/model_save_{try_num}.pth')


model.load_weights(f'./_save/ModelCheckPoint/ai_fac_MCP{try_num}.hdf5')

# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
# score = r2_score(y_test, y_predict)

# submission_predict = model.predict(test_sets)
gap_predict, gmp_predict = model.predict(test_sets)

# print(submission_predict)

# print('loss : ', round(loss,4))  
# print('y_predict : ', y_predict)
# print('submission_predict : ', submission_predict)


submission = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_sample.csv', header=0, index_col=0)

submission.iloc[:, 0:] = gap_predict

submission.to_csv(f'C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_gap_{try_num}th.csv')

submission.iloc[:, 0:] = gmp_predict

submission.to_csv(f'C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_gmp_{try_num}th.csv')

import matplotlib.pyplot as plt

def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "loss")