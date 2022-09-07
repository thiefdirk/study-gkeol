import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1): # alpha : 기울기
    return np.where(x>=0, x, alpha * (np.exp(x)-1)) # np.where(조건, 참, 거짓)

elu2 = lambda x, alpha=1: np.where(x>=0, x, alpha * (np.exp(x)-1))

x = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1씩 증가
print(x, len(x))

y = elu2(x)

plt.plot(x, y, 'k-') # k- : 검은색 실선
plt.grid()
plt.show()

# elu, selu, reaky relu
# 3_2, 3_3, 3_4