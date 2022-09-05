# DNN으로 구성
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
tf.compat.v1.set_random_seed(123)
tf.compat.v1.disable_eager_execution()

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. # astype : 형변환, float32 : 32비트 실수형
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])


#hidden layer 1
w1 = tf.compat.v1.get_variable('weight1', shape=[2,2,1,128]) # 2,2,1,64 : 2,2 : 커널사이즈, 1 : 컬러, 64 : 필터갯수
#model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,3,128,64], name='weight2'))

#hidden layer 3
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,3,64,32], name='weight3'))

#DNN layer
w4 = tf.compat.v1.get_variable('weight4',shape = [512,100], initializer=tf.keras.initializers.GlorotUniform())
b1 = tf.Variable(tf.compat.v1.random_normal([100]), name='bias1')

#Output layer
w5 = tf.compat.v1.get_variable('weight5',shape = [100,10], initializer=tf.keras.initializers.GlorotUniform())
b2 = tf.Variable(tf.compat.v1.random_normal([10]), name='bias2')

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') # strides : 1,1,1,1 : 1,1 : 스트라이드, # padding='VALID' : 패딩사용안함
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ksize : 커널사이즈, strides : 스트라이드, padding : 패딩
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='VALID') # strides : 1,1,1,1 : 1,1 : 스트라이드, # padding='VALID' : 패딩사용안함
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ksize : 커널사이즈, strides : 스트라이드, padding : 패딩
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='VALID') # strides : 1,1,1,1 : 1,1 : 스트라이드, # padding='VALID' : 패딩사용안함
L3 = tf.nn.elu(L3)
# F1 = tf.compat.v1.layers.flatten(L3)
F1 = tf.reshape(L3, [-1, L3.shape[1]*L3.shape[2]*L3.shape[3]])
print(L3) # Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)
L4 = tf.nn.selu(tf.matmul(F1, w4) + b1)
L4 = tf.nn.dropout(L4, rate=0.3) # rate : 0.3
hypothesis = tf.matmul(L4, w5) + b2


# hypothesis = tf.nn.softmax(tf.matmul(F1, w4)) # tf.nn : 신경망 관련 함수



# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 31
batch_size = 100
total_batch = int(len(x_train)/batch_size)
for epochs in range(epoch) :
    next_batch = 0
    avg_loss = 0
    for i in range(total_batch) :
        batch_x = x_train[next_batch:next_batch+batch_size]
        batch_y = y_train[next_batch:next_batch+batch_size]
        next_batch += batch_size
        _, loss_val = sess.run([train, loss], feed_dict={x: batch_x, y: batch_y})
        avg_loss += loss_val/total_batch
        prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1)) 
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    if epochs % 10 == 0 :
        print(epochs + 1, avg_loss, acc)

        
# 4. 평가, 예측
prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1)) # arg_max : 최대값의 인덱스를 반환, argmax(hypothesis, 1) : 행단위로 최대값의 인덱스를 반환
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

sess.close()
# acc :  0.5455
