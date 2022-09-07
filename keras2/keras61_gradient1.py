import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6

x = np.linspace(-1, 6, 100) # linspace : -1부터 6까지 100개의 구간으로 나눔
print(x, len(x))

y = f(x)

plt.plot(x, y, 'k-') # k- : 검은색 실선
plt.plot(2, 2, 'sk') # sk : 검은색 사각형
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()