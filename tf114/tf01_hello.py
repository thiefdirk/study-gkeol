import tensorflow as tf
import sys

# 파이썬 버전확인
print("Python version: {}".format(sys.version))
# 텐서플로우 버전확인
print("TensorFlow version: {}".format(tf.__version__))

# print("Hello, TensorFlow!")

hello = tf.constant('hello world!')
print(tf.constant('hello world!'))
print(hello)

# sess = tf.Session() # session : 실행환경
sess = tf.compat.v1.Session()
print(sess.run(hello))

# 텐서플로는 출력할때 sess.run()을 사용해야 한다.

# constant() 함수는 상수를 만드는 함수이다.
# variable() 함수는 변수를 만드는 함수이다.
# placeholder() 함수는 변수를 만드는 함수이다.