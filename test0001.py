# 3-dim dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
# randam np.array (1316664, 1)

a = pd.DataFrame(np.random.rand(48, 1))
b = pd.DataFrame(np.random.rand(48, 1))

# 데이터프레임에서 7과 8의 배수행 index 추출

index = a.index[(a.index % 7 == 0) | (a.index % 8 == 0)]

# 추출한 index를 이용하여 데이터프레임에서 행 삭제

a = a.drop(index)

# concat np.array

c = np.concatenate((a, b), axis=1)
