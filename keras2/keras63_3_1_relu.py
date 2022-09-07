import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x) # 0과 x 중 큰 값을 반환

relu2 = lambda x: np.maximum(0, x) # 0과 x 중 큰 값을 반환

x = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1씩 증가
print(x, len(x))

y = relu2(x)

plt.plot(x, y, 'k-') # k- : 검은색 실선
plt.grid()
plt.show()

# elu, selu, reaky relu
# 3_2, 3_3, 3_4