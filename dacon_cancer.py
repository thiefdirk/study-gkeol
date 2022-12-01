import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import cv2



train_data = pd.read_csv('D:\open/train.csv', encoding='utf-8', index_col=0)
test_data = pd.read_csv('D:\open/test.csv', encoding='utf-8', index_col=0)

# 결측치 확인
print(train_data.isnull().sum())

# '암의 장경' 컬럼 결측지 0

train_data['암의 장경'] = train_data['암의 장경'].fillna(train_data['암의 장경'].mean())
test_data['암의 장경'] = test_data['암의 장경'].fillna(test_data['암의 장경'].mean())
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

scaler = StandardScaler()
le = LabelEncoder()

print(train_data.shape) # (1000, 28)
print(test_data.shape) # (250, 26)

train_data, val_data,train_label , val_label = train_test_split(train_data.drop('N_category', axis=1), train_data['N_category'], test_size=0.2, random_state=42, stratify=train_data['N_category'])

print(train_data.head())
print(train_label.head())
print(val_label.head())
print(val_data.head())


for col in train_data.columns:
    if col in ignore_cols:
        continue
    if col in numeric_cols:
        train_data[col] = scaler.fit_transform(train_data[col].values.reshape(-1,1)) 
        val_data[col] = scaler.transform(val_data[col].values.reshape(-1,1))
        test_data[col] = scaler.transform(test_data[col].values.reshape(-1,1))
    else:
        train_data[col] = le.fit_transform(train_data[col])
        val_data[col] = le.transform(val_data[col])
        test_data[col] = le.transform(test_data[col])
    


#import image from D:\open\train_imgs/
train_imgs = []
for i in range(len(train_data)):
    img = cv2.imread('D:\open'+train_data['img_path'][i])
    # resize
    img = np.array(img)/255
    img = cv2.resize(img, (512, 512))
    
    train_imgs.append(np.array(img))
    print(train_imgs)
    
np.save('C:\study\_data\dacon_cancer/train_imgs.npy', train_imgs)
# train_imgs = np.load('C:\study\_data\dacon_cancer/train_imgs.npy')
        
    

mask_imgs = [] # grayscale
for i in range(len(train_data)):
    if train_data['mask_path'][i] == None:
        mask_imgs.append(np.zeros((int(img.size[0]/2),int(img.size[1]/2))))
    else: # grayscale
        img = cv2.imread('D:\open'+train_data['mask_path'][i], cv2.IMREAD_GRAYSCALE)
        img = np.array(img)/255
        img = cv2.resize(img, (512, 512))
        
        mask_imgs.append(np.array(img))
        
np.save('C:\study\_data\dacon_cancer/mask_imgs.npy', mask_imgs)
# mask_imgs = np.load('C:\study\_data\dacon_cancer/mask_imgs.npy')
        
test_imgs = []
for i in range(len(test_data)):
    img = cv2.imread('D:\open'+test_data['img_path'][i])
    img = np.array(img)/255        
    img = cv2.resize(img, (512, 512))

    test_imgs.append(np.array(img))
    
np.save('C:\study\_data\dacon_cancer/test_imgs.npy', test_imgs)
# test_imgs = np.load('C:\study\_data\dacon_cancer/test_imgs.npy')

exit()
print(train_data.shape) # (1000, 28)
print(test_data.shape) # (250, 26)

train_img, val_img = train_test_split(train_imgs, test_size=0.2, random_state=42, stratify=train_data['N_category'])
train_mask, val_mask = train_test_split(mask_imgs, test_size=0.2, random_state=42, stratify=train_data['N_category'])


train_data = train_data.drop(['ID', 'img_path', 'mask_path'], axis=1)
test_data = test_data.drop(['ID', 'img_path'], axis=1)
    
# multi input model(img, data)

# img model(unet with pretrained weight)
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Dropout, Add, Conv2DTranspose, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def unet(input_size=(512,512,3)): # input = (512,512,3), output = (512,512,1)
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    return y 

def get_wnet():
    inputs = Input((512, 512, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = aspp_block(pool4, 512)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    down10 = concatenate([Conv2D(32, (3, 3), activation='relu', padding='same')(conv9), conv9], axis=3)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(down10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = BatchNormalization()(conv10)    
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)

    down11 = concatenate([Conv2D(64, (3, 3), activation='relu', padding='same')(pool10), conv8], axis=3)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(down11)
    conv11 = BatchNormalization()(conv11)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)
    conv11 = BatchNormalization()(conv11)   
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    down12 = concatenate([Conv2D(128, (3, 3), activation='relu', padding='same')(pool11), conv7], axis=3)
    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(down12)
    conv12 = BatchNormalization()(conv12)
    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv12)    
    conv12 = BatchNormalization()(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)

    down13 = concatenate([Conv2D(256, (3, 3), activation='relu', padding='same')(pool12), conv6], axis=3)
    conv13 = Conv2D(256, (3, 3), activation='relu', padding='same')(down13)
    conv13 = BatchNormalization()(conv13)
    conv13 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv13)  
    conv13 = BatchNormalization()(conv13)    
    pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    conv14 = aspp_block(pool13, 512)
    
    up15 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv14), conv13], axis=3)
    conv15 = Conv2D(256, (3, 3), activation='relu', padding='same')(up15)
    conv15 = BatchNormalization()(conv15)    
    conv15 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv15)
    conv15 = BatchNormalization()(conv15) 
    
    up16 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv15), conv12], axis=3)
    conv16 = Conv2D(128, (3, 3), activation='relu', padding='same')(up16)
    conv16 = BatchNormalization()(conv16)     
    conv16 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv16)
    conv16 = BatchNormalization()(conv16)      

    up17 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv16), conv11], axis=3)
    conv17 = Conv2D(64, (3, 3), activation='relu', padding='same')(up17)
    conv17 = BatchNormalization()(conv17)      
    conv17 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv17)
    conv17 = BatchNormalization()(conv17)  
    
    up18 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv17), conv10], axis=3)
    conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(up18)
    conv18 = BatchNormalization()(conv18)      
    conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv18)
    conv18 = BatchNormalization()(conv18)    
    
    conv18 = aspp_block(conv18, 32)
    
    conv19 = Conv2D(1, (1, 1), activation='sigmoid')(conv18)

    model = Model(inputs=[inputs], outputs=[conv19])
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics='accuracy')
    return model


# data model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def data_model(input_size=(25,)): # input = (25,), output = (1,)
    inputs = Input(input_size)
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# train model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils import plot_model
import os

# merge model
def merge_model(): # input = (512, 512, 3), (25,), output = (1)
    
    model1 = get_wnet()
    model2 = data_model()
    
    inputs1 = Input((512, 512, 3))
    inputs2 = Input((25,))
    
    x1 = model1(inputs1)
    x2 = model2(inputs2)
    
    # (512, 512, 1) -> f
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    gm1 = GlobalMaxPooling2D()(conv1)
    concat1 = concatenate([gm1, x2])
    dense1 = Dense(32, activation='relu')(concat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(16, activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(8, activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(4, activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(2, activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(1, activation='sigmoid')(dense1)
    
    model = Model(inputs=[inputs1, inputs2], outputs=[dense1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['f1_score'])
    
    return model

# train
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils import plot_model
import os

def train():
    model = merge_model()
    plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()
    
    # # load data
    # train_data = np.load('train_data.npy')
    # train_label = np.load('train_label.npy')
    # val_data = np.load('val_data.npy')
    # val_label = np.load('val_label.npy')
    
    # train
    checkpoint = ModelCheckpoint('model.h5', monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, verbose=1)
    callbacks_list = [checkpoint, early]
    model.fit([train_img, train_data], [train_mask,train_label], validation_data=([val_img, val_data], [val_mask,val_label]), epochs=100, batch_size=8, callbacks=callbacks_list)
    
    # save model
    model.save('model.h5')
    
    return model

from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# test
def test():
    model = load_model('model.h5', custom_objects={'f1_score': f1_score})
    model.summary()
    
    # load data
    # test_data = np.load('test_data.npy')
    # test_label = np.load('test_label.npy')
    
    # test
    pred = model.predict([test_imgs, test_data], batch_size=8)
    submisson_pred = pd.read_csv('D:\open/sample_submission.csv')
    submisson_pred['N_category'] = pred
    submisson_pred.to_csv('submission.csv', index=False)
    
    return model

if __name__ == '__main__':
    train()
    test()

