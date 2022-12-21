import pstats
import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
import autokeras as ak
from tensorflow.python.keras.losses import mean_absolute_error

print(tf.test.is_gpu_available())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try_num = 40


data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
x_reshaped = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/x_reshaped.csv', header=0)
test_sets_reshaped = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/test_sets_reshaped.csv', header=0)

test_sets_path = 'C:/Users/kogas/Desktop/ai_factory/_data/dataset/test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)

for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

# print(data)
# print(test_0, test_0.shape)


# data['TIME'] = pd.to_datetime(data['TIME'])
# data['year'] = data['TIME'].dt.year
# data['month'] = data['TIME'].dt.month
# data['day'] = data['TIME'].dt.day
# data['hour'] = data['TIME'].dt.hour
# data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)
print(data)

# test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
# test_sets['year'] = test_sets['TIME'].dt.year
# test_sets['month'] = test_sets['TIME'].dt.month
# test_sets['day'] = test_sets['TIME'].dt.day
# test_sets['hour'] = test_sets['TIME'].dt.hour
# test_sets['weekday'] = test_sets['TIME'].dt.weekday
# pred_index = test_sets['TIME'].values
test_sets = test_sets.drop(['TIME'], axis=1, inplace=False)

data.info()
test_sets.info()
# data.astype(float).dtypes

x = data
x = data.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S'], axis=1)
test_sets = test_sets.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S'], axis=1)
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y

print(x) 
print(x.shape) 
print(y.shape)

# x = x.to_numpy()
# y = y.to_numpy()
# test_sets = test_sets.to_numpy()
# x_df3_ = x_df3_.to_numpy()

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


# x = split_x(x, size)
# x = x[:-1,:]
# y = y[6:,:]
# test_sets = split_test(test_sets, size)
print(x.shape)
print(y.shape)
print(test_sets.shape)


def np_to_df2(np_array):# 109722, 6, 12 -> 109722, 72
    df = pd.DataFrame(np_array[0].reshape(1,72))
    for i in range(1, len(np_array)):
        df = df.append(pd.DataFrame(np_array[i].reshape(1,72)))
    return df

def np_to_df3(np_array):# 109722, 6, 12 -> 109722, 12 (모든행 더해서 6으로 나눔)
    df = pd.DataFrame(np.reshape(np_array[0].sum(axis=0)/6, (1,7)))
    print(np_array[0].shape)
    print(df.shape)
    for i in range(1, len(np_array)):
        df = pd.concat([df, pd.DataFrame(np.reshape(np_array[i].sum(axis=0)/6, (1,7)))], axis=0)
        print(df.shape)
    return df

def df_to_np(df):
    np_array = np.array(df.iloc[0:12])
    for i in range(1, len(df//12)):
        np_array = np.append(np_array, np.array(df.iloc[i*12:(i+1)*12]), axis=0)
        print(np_array.shape)
    return np_array.reshape(len(df)//12,12)

# print(x.shape)
# print(test_sets.shape)

# x.to_csv('./x_reshaped.csv', index=False)
# test_sets.to_csv('./test_sets_reshaped.csv', index=False)


from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y, train_size=0.85, shuffle=False, random_state=66)

# model = Sequential()
# model.add(Conv1D(filters=128, kernel_size=2, padding='same', input_shape=(6, 12)))
# model.add(MaxPooling1D(pool_size=2))
# # model.add(Conv1D(filters=64, kernel_size=2, padding='same'))
# # model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same'))
# model.add(Flatten())
# model.add(Dense(50, activation='selu'))
# model.add(Dense(30, activation='selu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='elu'))
# model.add(Dense(2))

model = ak.TimeseriesForecaster(lookback=6, loss=mean_absolute_error, seed=777, predict_from=1)

# model.fit(x,y, validation_split=0.2, verbose=1, batch_size=6)

submission_predict = model.predict(test_sets)

submission_predict_6 = submission_predict[::6, :]

print(submission_predict.shape)
submission_predict =np.concatenate([np.reshape(submission_predict[0],(1,2)), submission_predict_6], axis=0)

print(submission_predict.shape)



# model.fit(x_reshaped, y)


# model = load_model(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/model_save_{try_num}.csv')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# es = EarlyStopping(monitor='val_loss', patience=70, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath=f'./_save/ModelCheckPoint/ai_fac_MCP{try_num}.hdf5')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=40, factor=0.5, mode='auto', verbose=1)

# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=500, batch_size=256, validation_data = [x_test, y_test], verbose=1, callbacks=[es,reduce_lr])

# model.save(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/model_save_{try_num}.csv')

# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
# score = r2_score(y_test, y_predict)

# submission_predict = model.predict(test_sets_reshaped)

# print('loss : ', round(loss,4))  #loss :  0.013
# print('y_predict : ', y_predict)
# print('submission_predict : ', submission_predict)


submission = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_sample.csv', header=0, index_col=0)
# print(submission)
submission.iloc[:, 0:] = submission_predict
# submission = pd.DataFrame(submission_predict, columns=pred_col.columns, index=pred_index)

submission.to_csv(f'C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_{try_num}th.csv')