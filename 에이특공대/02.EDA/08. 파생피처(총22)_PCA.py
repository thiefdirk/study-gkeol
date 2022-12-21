import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# try_num = 

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)

data = data.reset_index(drop=True)
out_data = out_data.reset_index(drop=True)
test_out_data = test_out_data.reset_index(drop=True)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]
test_out_data = test_out_data.drop(index)
test_out_data_np = test_out_data.to_numpy()

test_sets_path = 'C:/Users/kogas/Desktop/ai_factory/_data/dataset/test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)


for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)

data['bog']=data['FY_SUM'] + data['FIA_SUM']
data['lp_ti'] = data['FY_SUM'] + data['LP_TOTAL']

test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
test_sets['year'] = test_sets['TIME'].dt.year
test_sets['month'] = test_sets['TIME'].dt.month
test_sets['day'] = test_sets['TIME'].dt.day
test_sets['hour'] = test_sets['TIME'].dt.hour
test_sets['weekday'] = test_sets['TIME'].dt.weekday
pred_index = test_sets['TIME'].values
test_sets = test_sets.drop(['TIME'], axis=1, inplace=False)

test_sets['bog']=test_sets['FY_SUM'] + test_sets['FIA_SUM']
test_sets['lp_ti'] = test_sets['FY_SUM'] + test_sets['LP_TOTAL']

x = data
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y
x = pd.concat([x, out_data], axis=1)


# x = x.to_numpy()
# y = y.to_numpy()
# print(y.shape)

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

def split_test(seq, size):
    aaa=[]
    for i in range(5508):
        subset = seq[i*size : (i + 1)*size]
        aaa.append([item for item in subset])
    return np.array(aaa)

size = 6 


x = split_x(x, size)
x = x[:-1,:]
y = y[6:,:]
test_sets = split_test(test_sets, size)


print(x.shape)

from sklearn.decomposition import PCA

for i in range(1,25):
    pca = PCA(n_components=i)
    principalcomponents = pca.fit_transform(x)

    print(f'{i}PCA',pca.explained_variance_ratio_)
    print(f'{i}PCA',sum(pca.explained_variance_ratio_))
    print()
'''
1PCA [0.97584357]
1PCA 0.9758435700706553

2PCA [0.97584357 0.01423971]
2PCA 0.9900832791338326

3PCA [0.97584357 0.01423971 0.00500856]
3PCA 0.9950918362677423

4PCA [0.97584357 0.01423971 0.00500856 0.00208142]
4PCA 0.9971732537922519

5PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04]
5PCA 0.9981424531170079

6PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04]
6PCA 0.9989749568864692

7PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04]
7PCA 0.9994717694251549

8PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04]
8PCA 0.9996070513861058

9PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04]
9PCA 0.99973185758296

10PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05]
10PCA 0.9998232948095942

11PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05]
11PCA 0.9998631117690007

12PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05]
12PCA 0.9999006394034858

13PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05]
13PCA 0.9999259616146463

14PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05]
14PCA 0.9999478023554378

15PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05]
15PCA 0.9999642126698098

16PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05]
16PCA 0.9999795155092637

17PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05]
17PCA 0.9999916597330286

18PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06]
18PCA 0.9999949395113503

19PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06 3.13576075e-06]
19PCA 0.9999980752721046

20PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06 3.13576075e-06 1.04681761e-06]
20PCA 0.9999991220897103

21PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06 3.13576075e-06 1.04681761e-06
 8.69916022e-07]
21PCA 0.9999999920057324

22PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06 3.13576075e-06 1.04681761e-06
 8.69916022e-07 7.99426771e-09]
22PCA 1.0000000000000002

23PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06 3.13576075e-06 1.04681761e-06
 8.69916022e-07 7.99426771e-09 2.15964120e-32]
23PCA 1.0000000000000002

24PCA [9.75843570e-01 1.42397091e-02 5.00855713e-03 2.08141752e-03
 9.69199325e-04 8.32503769e-04 4.96812539e-04 1.35281961e-04
 1.24806197e-04 9.14372266e-05 3.98169594e-05 3.75276345e-05
 2.53222112e-05 2.18407408e-05 1.64103144e-05 1.53028395e-05
 1.21442238e-05 3.27977832e-06 3.13576075e-06 1.04681761e-06
 8.69916022e-07 7.99426771e-09 2.15964120e-32 6.55105325e-34]
24PCA 1.0000000000000002


'''    