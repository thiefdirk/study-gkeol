from warnings import filters
# import DenseNet264
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, Conv2DTranspose, UpSampling2D, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import matplotlib.pyplot as plt
import pickle as pkl
import joblib
start = time.time()
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################


#1. 데이터

x_train = np.load('d:/study_data/_save/_npy/keras49_9_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_9_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_9_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_9_test_y.npy')
test_set = np.load('d:/study_data/_save/_npy/keras49_9_test_set.npy')



x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
test_set_noised = test_set + np.random.normal(0, 0.1, size=test_set.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # np.clip는 최소값과 최대값을 지정해주는 함수입니다.
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
test_set_noised = np.clip(test_set_noised, a_min=0, a_max=1)

print(x_train.shape) # (3316, 100, 100, 3)
print(x_train_noised.shape) # (3316, 100, 100, 3)
print(test_set.shape) # (1, 100, 100, 3)
print(test_set_noised.shape) # (1, 100, 100, 3)
print(x_train) # (831, 100, 100, 3)
print(x_train_noised) # (831, 100, 100, 3)
print(test_set) # (1, 100, 100, 3)

#2. 모델


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 3))) 
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(hidden_layer_size/4, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    model.summary()

    return model

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 3)))
#     model.add(MaxPooling2D((2, 2), padding='same'))
#     model.add(Conv2D(hidden_layer_size/4, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2DTranspose(hidden_layer_size/2, (7, 7), activation='relu', padding='valid'))
#     model.add(Conv2DTranspose(hidden_layer_size/2, (7, 7), activation='relu', padding='valid'))
#     model.add(Conv2DTranspose(hidden_layer_size/2, (10, 10), activation='relu', padding='valid'))
#     model.add(Conv2DTranspose(hidden_layer_size/2, (10, 10), activation='relu', padding='valid'))
#     model.add(Conv2DTranspose(hidden_layer_size/2, (11, 11), activation='relu', padding='valid'))
#     model.add(Conv2DTranspose(3, (11, 11), activation='sigmoid', padding='valid'))
#     model.summary()
#     return model

model = autoencoder(hidden_layer_size=100)


es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=100, batch_size=180, validation_split=0.2, callbacks=[es, lr], verbose=1)

path = 'D:\study_data\_save/_xg/'
# model = pkl.load(open(path + 'a08_men_women_model.pkl', 'rb'))
# pkl.dump(model, open(path+'a08_men_women_model.pkl', 'wb')) # 저장, wb : write binary
# joblib.dump(model, path+'a08_men_women_model.dat')

#save model


output = model.predict(x_test_noised)
output_test = model.predict(test_set_noised)

random_test = np.random.randint(x_test.shape[0], size=5)
print(random_test)

for x,i in enumerate(random_test):
    ax = plt.subplot(3,6,x+1)    
    if x <= 4:
        plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3,6,x+7)
    if x <= 4:
        plt.imshow(x_test_noised[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3,6,x+13)    
    if x <= 4:
        plt.imshow(output[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
ax = plt.subplot(3,6,6)
plt.imshow(test_set[0])
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(3,6,12)
plt.imshow(test_set_noised[0])
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(3,6,18)
plt.imshow(output_test[0])
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.tight_layout() # 그래프 간격 조정
plt.show()
    

# loss :  0.6230305433273315
# acc :  0.86908358335495

# loss :  0.6827794313430786
# val_loss :  0.976878821849823
# accuracy :  0.9928355813026428
# val_accuracy :  0.759036123752594
# loss:  [0.5758450031280518, 0.6827794313430786]
# 성별 :  [0]
# time : 27.231499195098877
