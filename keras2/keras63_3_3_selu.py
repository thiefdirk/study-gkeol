import numpy as np
import matplotlib.pyplot as plt

def selu(x, scale=1.0507, alpha=1.67326): # scale : 1.0507, alpha : 1.67326
    return scale * (np.maximum(0,x) + np.minimum(0,alpha * (np.exp(x)-1)))

selu2 = lambda x, scale=1.0507, alpha=1.67326: scale * (np.maximum(0,x) + np.minimum(0,alpha * (np.exp(x)-1)))
 


x = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1씩 증가
print(x, len(x))

y = selu2(x)

plt.plot(x, y, 'k-') # k- : 검은색 실선
plt.grid()
plt.show()

# elu, selu, reaky relu
# 3_2, 3_3, 3_4