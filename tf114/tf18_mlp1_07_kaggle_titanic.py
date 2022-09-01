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





train_set = train_set.fillna({"Embarked": "C"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

x_data = train_set.drop(['Survived'], axis=1)
y_data = train_set['Survived']

y_data = y_data.values

print(y_data.shape) # (891,)

y_data = y_data.reshape(-1,1)

print(y_data.shape) # (891,)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66, stratify=y_data)
sess.run(tf.compat.v1.global_variables_initializer())

x=tf.compat.v1.placeholder(tf.float32, shape=[None,8])
y=tf.compat.v1.placeholder(tf.float32, shape=[None,1])

#hidden layer 1
w1 = tf.compat.v1.Variable(tf.random_normal([8,20], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random_normal([20], name='bias1'))

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.random_normal([20,15], name='weight1'))
b2 = tf.compat.v1.Variable(tf.random_normal([15], name='bias1'))

#output layer
w3 = tf.compat.v1.Variable(tf.random_normal([15,1], name='weight2'))
b3 = tf.compat.v1.Variable(tf.random_normal([1], name='bias2'))


hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1) # sigmoid : 0~1 사이의 값으로 변환
hidden_layer2 = tf.sigmoid(tf.matmul(hidden_layer1, w2) + b2) # sigmoid : 0~1 사이의 값으로 변환
hypothesis = tf.sigmoid(tf.matmul(hidden_layer2, w3) + b3) # tf.nn : 신경망 관련 함수


# 2. 모델구성


# hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
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
y_pred = np.where(y_pred > 0.5, 1, 0)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()
# acc :  0.8268156424581006