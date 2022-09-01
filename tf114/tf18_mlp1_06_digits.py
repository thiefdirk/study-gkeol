from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype, load_digits
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

tf.set_random_seed(66)

# 1. 데이터
sess = tf.compat.v1.Session()
datasets = load_digits()
# x_data = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# y_data = pd.DataFrame(datasets.target, columns=['target'])
x_data = datasets.data
y_data = datasets.target
# y_data = y_data.reshape(-1, 1)
print(y_data.shape) # (569, 1)

y_data = pd.get_dummies(y_data).values

print(y_data.shape) # (569, 1)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66, stratify=y_data)


x=tf.compat.v1.placeholder(tf.float32, shape=[None,64])
y=tf.compat.v1.placeholder(tf.float32, shape=[None,10])

#hidden layer 1
w1 = tf.compat.v1.Variable(tf.random_normal([64,20], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random_normal([20], name='bias1'))

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.random_normal([20,15], name='weight1'))
b2 = tf.compat.v1.Variable(tf.random_normal([15], name='bias1'))

#output layer
w3 = tf.compat.v1.Variable(tf.random_normal([15,10], name='weight2'))
b3 = tf.compat.v1.Variable(tf.random_normal([10], name='bias2'))


hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1) # sigmoid : 0~1 사이의 값으로 변환
hidden_layer2 = tf.sigmoid(tf.matmul(hidden_layer1, w2) + b2) # sigmoid : 0~1 사이의 값으로 변환
hypothesis = tf.nn.softmax(tf.matmul(hidden_layer2, w3) + b3) # tf.nn : 신경망 관련 함수



loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.matmul(x, w) + b, labels=y)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2001
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 100 == 0 :
        print(epochs, 'loss : ', cost_val, '\n')
        
# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()
# acc :  0.9305555555555556