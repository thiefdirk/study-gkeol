from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_test.shape) # (40000,)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 40000 # 증폭
randindx = np.random.randint(x_train.shape[0], size = augument_size)
print(randindx,randindx.shape) # (40000,)
print(np.max(randindx), np.min(randindx)) # 59997 2
print(type(randindx)) # <class 'numpy.ndarray'>

x_augumented = x_train[randindx].copy()
print(x_augumented,x_augumented.shape) # (40000, 28, 28)
y_augumented = y_train[randindx].copy()
print(y_augumented,y_augumented.shape) # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
# 이과정을 통해 x_augumented 만 변형을 해서 [0] 을붙어 뽑아내준다

print(x_augumented)
print(x_augumented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)


####################원핫인코더###################
df1 = pd.DataFrame(y_train)
df2 = pd.DataFrame(y_test)
print(df1)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y_train = oh.fit_transform(df1)
y_test = oh.transform(df2)
print('====================================')
print(y_train.shape)
print('====================================')
print(y_test.shape)
################################################

#### 모델구성 ####


#2. 모델구성
model = Sequential()
input1 = Input(shape=(28,28,1))
conv2D_1 = Conv2D(100,3, padding='same')(input1)
MaxP1 = MaxPooling2D()(conv2D_1)
drp1 = Dropout(0.2)(MaxP1)
conv2D_2 = Conv2D(200,2,
                  activation='relu')(drp1)
MaxP2 = MaxPooling2D()(conv2D_2)
drp2 = Dropout(0.2)(MaxP2)
conv2D_3 = Conv2D(200,2, padding='same',
                  activation='relu')(drp2)
MaxP3 = MaxPooling2D()(conv2D_3)
drp3 = Dropout(0.2)(MaxP3)
flatten = Flatten()(drp3)
dense1 = Dense(200)(flatten)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('relu')(batchnorm1)
drp4 = Dropout(0.2)(activ1)
dense2 = Dense(100)(drp4)
batchnorm2 = BatchNormalization()(dense2)
activ2 = Activation('relu')(batchnorm2)
drp5 = Dropout(0.2)(activ2)
dense3 = Dense(100)(drp5)
batchnorm3 = BatchNormalization()(dense3)
activ3 = Activation('relu')(batchnorm3)
drp6 = Dropout(0.2)(activ3)
output1 = Dense(10, activation='softmax')(drp6)
model = Model(inputs=input1, outputs=output1)   

# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

###############################로드모델###################################
# model = load_model(load_filepath + '0708_1757_0018-0.2908.hdf5')
#########################################################################

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=40, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=200, batch_size=128,
                 validation_split=0.3,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
df3 = pd.DataFrame(y_predict)
y_predict = oh.transform(df3)


acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

# loss :  [0.27665308117866516, 0.9068999886512756]
# acc스코어 :  0.9069

# loss :  [0.27819696068763733, 0.9103000164031982]
# acc스코어 :  0.9103