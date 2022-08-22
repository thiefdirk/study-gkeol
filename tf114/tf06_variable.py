import tensorflow as tf
sess = tf.compat.v1.Session()
# tf.variable() : 변수를 만드는 함수이다.

x = tf.Variable([2], dtype=tf.float32) # 변수, float32 : 32비트 실수
y = tf.Variable([3], dtype=tf.float32) # 변수

init = tf.compat.v1.global_variables_initializer() # 변수 초기화
sess.run(init)

sess.run(x+y)

