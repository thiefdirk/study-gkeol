# 3-dim dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df_corr = x.corr() # df_corr : 3-dim dataframe
mask = np.array(df_corr)
mask[np.tril_indices_from(mask)] = False  # np.tril_indices_from(mask) : 아래쪽 삼각형만 True로 만들어라

plt.figure(figsize= (10,8))

sns.heatmap(df_corr, 
            cmap = sns.light_palette("red", as_cmap=True),
            annot = True, 
            mask = mask)

plt.show()

# 컬럼 19개 데이터 뒤에서 4번째 컬럼 드랍

x = x.drop(x.columns[15], axis=1)