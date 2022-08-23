import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1, 2, 3]
y_train = [1, 2, 3]
x_test = [4, 5, 6]
y_test = [4, 5, 6]


x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x) # gradient : 기울기
descent = w - lr * gradient # descent : 하강
update = w.assign(descent) # assign : 할당  

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_val, w_val = sess.run([update, loss, w], feed_dict={x: x_train, y: y_train})
    print(step, '\t' ,loss_val, '\t' ,w_val) # \t : tab
    w_history.append(w_val[0])
    loss_history.append(loss_val)
    
y_pred = x_test * w_val

print("y_pred : ", y_pred)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("r2 : ", r2)
print("mae : ", mae)





sess.close()


    
