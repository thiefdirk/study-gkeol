from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

tf.set_random_seed(66)

# 1. 데이터
sess = tf.compat.v1.Session()
datasets = fetch_covtype()
# x_data = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# y_data = pd.DataFrame(datasets.target, columns=['target'])
x_data = datasets.data
y_data = datasets.target
# y_data = y_data.reshape(-1, 1)
print(y_data.shape) # (569, 1)

y_data = pd.get_dummies(y_data).values

print(y_data.shape) # (569, 1)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66, stratify=y_data)


x=tf.compat.v1.placeholder(tf.float32, shape=[None,54])
y=tf.compat.v1.placeholder(tf.float32, shape=[None,7])

w=tf.Variable(tf.compat.v1.random_normal([54,7]), name='weight')
b=tf.Variable(tf.compat.v1.random_normal([1,7]), name='bias')


# w=tf.Variable(tf.zeros([30,1]), name='weight')
# b=tf.Variable(tf.zeros([1]), name='bias')

# w=tf.Variable(tf.ones([30,1]), name='weight')
# b=tf.Variable(tf.ones([1]), name='bias')

# w dtype change to float64
# w = tf.cast(w, tf.float64)
# b = tf.cast(b, tf.float64)
# print(x_data[0:1])
# print(x_data[0:1].shape)


# print(tf.matmul(x_data[0:1], w))
# print(sess.run(tf.matmul(x_data[0:1], w)))


# 2. 모델구성


hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.matmul(x, w) + b, labels=y)

# log1p : log(1+x) 계산
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2001
for epochs in range(epoch) :
    cost_val, hy_val, _, w_val, b_val = sess.run([loss, hypothesis, train, w, b],
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
# acc :  1.0