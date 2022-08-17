import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = 'C:\study\_data\dacon_chung'

# input = []
# target = []
# break_num_input = 0
# break_num_target = 0

# while break_num_input < 60:
#     try:
#         datasets = pd.read_csv(path + '/train_input/CASE_' + '%02d' %(break_num_input) + '.csv')
#         input.append(datasets)
#         break_num_input += 1
#         print(np.array(input).shape)        
#     except:
#         break_num_input += 1
#         continue
    
# while break_num_target < 60:
#     try:
#         datasets = pd.read_csv(path + '/train_target/CASE_' + '%02d' %(break_num_target) + '.csv', index_col=0)
#         target.append(datasets['rate'])
#         break_num_target += 1
#         print(np.array(target).shape)
#     except:
#         break_num_target += 1
#         continue

# save_path = 'C:\study\_data\dacon_chung'
# joblib.dump(input, save_path + '/input.pkl')
# joblib.dump(target, save_path + '/target.pkl')


# print(input)
# print(target)

input = np.array(joblib.load(path + '/input.pkl'))
target = np.array(joblib.load(path + '/target.pkl'))

input = input.reshape(1607, 1440, 37)

print(input)
print(target)
target = pad_sequences(target, padding='post')
print(target.shape)

#2. 모델링

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, MaxPooling1D, GRU

model = Sequential()
model.add(Flatten(input_shape=(target.shape[1], 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(47))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(input, target, epochs=100, batch_size=37, verbose=1)

#4. 평가, 예측
loss, mae = model.evaluate(input, target, batch_size=37)
print('loss : ', loss)
print('mae : ', mae)

break_num_test = 0

