import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Input
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.client import device_lib
import seaborn as sns

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('./train.csv', header=0)
out_data = pd.read_csv('./out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('./test_out_data.csv', header=0, index_col=0)

data = data.reset_index(drop=True)
out_data = out_data.reset_index(drop=True)
test_out_data = test_out_data.reset_index(drop=True)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]
test_out_data = test_out_data.drop(index)

test_out_data_np = test_out_data.to_numpy()


test_sets_path = './test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)

for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)

test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
test_sets['year'] = test_sets['TIME'].dt.year
test_sets['month'] = test_sets['TIME'].dt.month
test_sets['day'] = test_sets['TIME'].dt.day
test_sets['hour'] = test_sets['TIME'].dt.hour
test_sets['weekday'] = test_sets['TIME'].dt.weekday
test_sets = test_sets.drop(['TIME'], axis=1, inplace=False)

data = pd.concat([data, out_data], axis=1)

'''

# 데이터 기본정보 확인
print(data.info())
print(data.isnull().sum())

# 이상치 확인

green_diamond = dict(markerfacecolor='r', marker='s')
plt.boxplot(data,flierprops=green_diamond)
plt.title("outlier box plot")
plt.xticks(range(1,len(data.columns)+1),data.columns, fontsize=10, rotation=20)
plt.show()

# PCA

from sklearn.decomposition import PCA

for i in range(1,25):
    pca = PCA(n_components=i)
    principalcomponents = pca.fit_transform(x)

    print(f'{i}PCA',pca.explained_variance_ratio_)
    print(f'{i}PCA',sum(pca.explained_variance_ratio_))
    print()


# 파생피처 상관관계
df_corr = data.corr()
df_corr = df_corr.apply(lambda x: round(x,2))

ax = sns.heatmap(df_corr, annot=True, linewidth=.5)
plt.show()

'''


x = data
x = x.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','FI_SUM'], axis=1)   #독립변수
test_sets = test_sets.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','FI_SUM'], axis=1)
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y

print(x.shape) # (109728, 18)
print(y.shape) # (109728, 2)

x = x.to_numpy()
y = y.to_numpy()
test_sets = test_sets.to_numpy()
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
test_sets = split_test(test_sets, size)
x = x[:-1,:]
y = y[6:,:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=False, random_state=66)

input_layer = Input((6,18))
Conv1D_1 = Conv1D(filters=128, kernel_size=2, padding='same')(input_layer)
MaxP_1 = MaxPooling1D(pool_size=2)(Conv1D_1)
Conv1D_2 = Conv1D(filters=64, kernel_size=2, padding='same')(MaxP_1)
MaxP_2 = MaxPooling1D(pool_size=2)(Conv1D_2)
Conv1D_3 = Conv1D(filters=32, kernel_size=2, padding='same')(MaxP_2)

# gap
GAP = GlobalAveragePooling1D()(Conv1D_3)
Dense_1_1 = Dense(50, activation = 'selu')(GAP)
Dense_1_2 = Dense(30, activation = 'selu')(Dense_1_1)
Dense_1_3 = Dense(32, activation = 'relu')(Dense_1_2)
Dense_1_4 = Dense(16, activation = 'relu')(Dense_1_3)
Dense_1_5 = Dense(8, activation = 'elu')(Dense_1_4)
GAPout = Dense(2, name=('gapout'))(Dense_1_5)

# gmp
GMP = GlobalMaxPooling1D()(Conv1D_3)
Dense_2_1 = Dense(50, activation = 'selu')(GMP)
Dense_2_2 = Dense(30, activation = 'selu')(Dense_2_1)
Dense_2_3 = Dense(32, activation = 'relu')(Dense_2_2)
Dense_2_4 = Dense(16, activation = 'relu')(Dense_2_3)
Dense_2_5 = Dense(8, activation = 'elu')(Dense_2_4)
GMPout = Dense(2, name=('gmpout'))(Dense_2_5)

model = Model(inputs=input_layer, outputs=[GAPout, GMPout])


# model.load_model('./model_save_67.pth')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./ai_fac_MCP67_.hdf5',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=100, factor=0.5, mode='auto', verbose=1)

model.compile(loss={'gapout':'mae','gmpout':'mae'}, optimizer='adam', loss_weights=[0.5,0.5])

history = model.fit(x_train, y_train, epochs=3000, batch_size=256, validation_split=0.2, verbose=1, callbacks=[es,reduce_lr,cp])

model.save('./model_save_67.pth')

model.load_weights('./ai_fac_MCP67.hdf5')

gap_predict, gmp_predict = model.predict(test_sets)

loss, gap_loss, gmp_loss = model.evaluate(x_test,y_test)

print(loss)
print(gap_loss)
print(gmp_loss)

submission = pd.read_csv('./submission_sample.csv', header=0, index_col=0)

submission.iloc[:, 0:] = gap_predict

submission.to_csv('./submission_gap_67th.csv')

# submission.iloc[:, 0:] = gmp_predict

# submission.to_csv('./submission_gmp_67th.csv')

