import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)
from sklearn.metrics import r2_score, mean_absolute_error
#1. 데이터

x1_data = [73., 93., 89., 96., 73.] # 국어
x2_data = [80., 88., 91., 98., 66.] # 영어
x3_data = [75., 93., 90., 100., 70.] # 수학
y_data = [152., 185., 180., 196., 142.] # 환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)

y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.Variable(tf.random.normal([1]), name='weight1')
w2 = tf.Variable(tf.random.normal([1]), name='weight2')
w3 = tf.Variable(tf.random.normal([1]), name='weight3')

b = tf.Variable(tf.random.normal([1]), name='bias')


#2. 모델 구성

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b


loss = tf.reduce_mean(tf.square(hypothesis - y))

# lr = 0.000089
# gradient = tf.reduce_mean((hypothesis - y) * x1) # gradient : 기울기
# descent = w1 - lr * gradient # descent : 하강
# w1_update = w1.assign(descent) # assign : 할당

# gradient = tf.reduce_mean((hypothesis - y) * x2) # gradient : 기울기
# descent = w2 - lr * gradient # descent : 하강
# w2_update = w2.assign(descent) # assign : 할당

# gradient = tf.reduce_mean((hypothesis - y) * x3) # gradient : 기울기
# descent = w3 - lr * gradient # descent : 하강
# w3_update = w3.assign(descent) # assign : 할당

# gradient = tf.reduce_mean((hypothesis - y)) # gradient : 기울기
# descent = b - lr * gradient # descent : 하강
# b_update = b.assign(descent) # assign : 할당

# w1_history = []
# w2_history = []
# w3_history = []

# loss_history = []

# b_history = []


# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# for step in range(1001):
#     _, _, _, b_val, loss_val, w1_val, w2_val, w3_val = sess.run([w1_update, w2_update, w3_update, b_update, loss, w1, w2, w3], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
#     print(step, '\t' , b_val, '\t' ,loss_val, '\t' ,w1_val, '\t', w2_val, '\t', w3_val  ) # \t : tab
#     w1_history.append(w1_val[0])
#     w2_history.append(w2_val[0])
#     w3_history.append(w3_val[0])
#     b_history.append(b_val[0])
#     loss_history.append(loss_val)
    
# y_pred = x1 * w1_val + x2 * w2_val + x3 * w3_val + b_val

# print('예측값 : ', sess.run(y_pred, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data}))




# r2 = r2_score(y_data, sess.run(y_pred, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data}))
# print('R2 : ', r2)

# mae = mean_absolute_error(y_data, sess.run(y_pred, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data}))
# print('MAE : ', mae)

# # 예측값 :  [149.59848 186.07285 180.36487 194.50705 144.84207]
# # R2 :  0.9917501449609749
# # MAE :  1.63485107421875

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000099)
train = optimizer.minimize(loss)

#3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 3001
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if epochs % 20 == 0 :
        print(epochs, 'loss : ', cost_val, '\n', hy_val)
        
sess.close()

y_pred = hy_val

sess = tf.compat.v1.Session()

print('예측값 : ', y_pred)



r2 = r2_score(y_data, y_pred)
print('R2 : ', r2)

mae = mean_absolute_error(y_data, y_pred)
print('MAE : ', mae)

sess.close()