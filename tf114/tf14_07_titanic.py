from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

tf.set_random_seed(66)

# 1. 데이터
sess = tf.compat.v1.Session()
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)

x_data = train_set.drop(['Survived'], axis=1)
y_data = train_set['Survived']

print(x_data.shape) # (891, 10)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66, stratify=y_data)
sess.run(tf.compat.v1.global_variables_initializer())

x=tf.compat.v1.placeholder(tf.float32, shape=[None,10])
y=tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w=tf.Variable(tf.compat.v1.random_normal([10,1]), name='weight')
b=tf.Variable(tf.compat.v1.random_normal([1]), name='bias')


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


hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
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
y_pred = np.where(y_pred > 0.5, 1, 0)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()
