# 비 지도학습

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
print(keras.__version__) # 2.9.0
(x_train, _), (x_test, _) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], -1) # (70000, 784)
x_columns = pd.DataFrame(x)
print(len(x_columns.columns)) # 784


# for i in range(len(x_columns.columns)) :
#     (x_train, _), (x_test, _) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)

#     x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)

#     x = x.reshape(x.shape[0], -1)
    
#     pca = PCA(n_components=i+1) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
#     x = pca.fit_transform(x) # x를 pca로 변환한다.
#     pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
#     cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.
#     print('n_components=', i+1, ':', cumsum[-1]) # 중요도를 이용해 주요하지 않은 변수를 제거한다.
#     print(np.argmax(cumsum >= 0.95)+1)
#     print(np.argmax(cumsum >= 0.99)+1)
#     print(np.argmax(cumsum >= 0.999)+1)
#     print(np.argmax(cumsum+1))

pca = PCA(n_components=783) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = pca.fit_transform(x) # x를 pca로 변환한다.
pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.
print('n_components=', 783, ':') # 중요도를 이용해 주요하지 않은 변수를 제거한다.
print(np.argmax(cumsum >= 0.95)+1) #154
print(np.argmax(cumsum >= 0.99)+1) #331
print(np.argmax(cumsum >= 0.999)+1) #486
print(np.argmax(cumsum+1)) #712


###############################################
# [실습]
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95
# 0.99
# 0.999
# 1.0
# 힌트 np.argmax(pca.explained_variance_ratio_)
# pca에 3차원 안들어감 2차원만 들어감
# reshape로 차원을 바꿔준다.
###############################################

