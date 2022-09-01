# model = XGBClassifier(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
tf.compat.v1.set_random_seed(123)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#1. 데이터
datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
y_data = pd.get_dummies(y_data).values


#2. 모델구성
#input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66, stratify=y_data)


#hidden layer 1
w1 = tf.compat.v1.Variable(tf.random_normal([4,20], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random_normal([20], name='bias1'))

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.random_normal([20,10], name='weight1'))
b2 = tf.compat.v1.Variable(tf.random_normal([10], name='bias1'))

#output layer
w3 = tf.compat.v1.Variable(tf.random_normal([10,3], name='weight2'))
b3 = tf.compat.v1.Variable(tf.random_normal([3], name='bias2'))


hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1) # sigmoid : 0~1 사이의 값으로 변환
hidden_layer2 = tf.sigmoid(tf.matmul(hidden_layer1, w2) + b2) # sigmoid : 0~1 사이의 값으로 변환
hypothesis = tf.nn.softmax(tf.matmul(hidden_layer2, w3) + b3) # tf.nn : 신경망 관련 함수




loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))



# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss)

#3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2001
for epochs in range(epoch) :
    cost_val, hy_val, _, w_val, b_val = sess.run([loss, hypothesis, train, w3, b3],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 100 == 0 :
        print(epochs, 'loss : ', cost_val, '\n')
        
# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = np.where(y_pred > 0.5, 1, 0)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()
# acc :  1.0