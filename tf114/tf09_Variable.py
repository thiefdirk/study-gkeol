import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32) # random_normal : 정규분포
print(변수)

#1. 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수) # 변수를 실행시키는 것
print("aaa : ", aaa) # aaa :  [-1.5080816]
sess.close()

#2. 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess, ) # 변수.eval() : 변수를 실행시키는 함수
print("bbb : ", bbb) # bbb :  [-1.5080816]
sess.close()

#3. 초기화 세번째
sess = tf.compat.v1.InteractiveSession() # InteractiveSession : 세션을 열어주는 함수
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval() # 변수.eval() : 변수를 실행시키는 함수
print("ccc : ", ccc) # ccc :  [-1.5080816]
sess.close()