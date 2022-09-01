import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
tf.compat.v1.set_random_seed(123)


#1. 데이터
x_data = [[0,0], [0,1], [1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

#2. 모델구성
#input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

#hidden layer 1
w1 = tf.compat.v1.Variable(tf.random_normal([2,20], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random_normal([20], name='bias1'))

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.random_normal([20,10], name='weight1'))
b2 = tf.compat.v1.Variable(tf.random_normal([10], name='bias1'))

#output layer
w3 = tf.compat.v1.Variable(tf.random_normal([10,1], name='weight2'))
b3 = tf.compat.v1.Variable(tf.random_normal([1], name='bias2'))


hidden_layer1 = tf.matmul(x, w1) + b1 # sigmoid : 0~1 사이의 값으로 변환
hidden_layer2 = tf.sigmoid(tf.matmul(hidden_layer1, w2) + b2) # sigmoid : 0~1 사이의 값으로 변환
hypothesis = tf.sigmoid(tf.matmul(hidden_layer2, w3) + b3) # sigmoid : 0~1 사이의 값으로 변환

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) # binary_crossentropy


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss)

#3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 500
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if epochs % 20 == 0 :
        print(epochs, 'loss : ', cost_val, '\n', hy_val)
        




y_pred = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32)) # 0.5보다 크면 1, 작으면 0, cast: 형변환


print('예측값 : ', y_pred)

acc = accuracy_score(y_data, y_pred)
print('acc : ', acc)

sess.close()