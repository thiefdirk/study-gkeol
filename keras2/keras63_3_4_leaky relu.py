import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.1): # alpha : 기울기
    return np.where(x>=0, x, alpha*x)

leaky_relu2 = lambda x, alpha=0.1: np.where(x>=0, x, alpha*x)

x = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1씩 증가
print(x, len(x))

y = leaky_relu2(x)

plt.plot(x, y, 'k-') # k- : 검은색 실선
plt.grid()
plt.show()

# elu, selu, leaky relu
# 3_2, 3_3, 3_4