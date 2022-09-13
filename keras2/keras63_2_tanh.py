import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x) 

tanh2 = lambda x: np.tanh(x)

x = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1씩 증가
print(x, len(x))

y = tanh2(x)

plt.plot(x, y, 'k-') # k- : 검은색 실선
plt.grid()
plt.show()