import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x)) # exp : 지수함수

softmax2 = lambda x: np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)

y = softmax2(x)

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()



# elu, selu, reaky relu
# 3_2, 3_3, 3_4