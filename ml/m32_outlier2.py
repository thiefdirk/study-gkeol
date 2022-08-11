import numpy as np
import pandas as pd
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa) # 행과 열을 바꿔줌
print(aaa)
print(aaa[:,0])
print(aaa[:,1])


def outlier(data_out, i) : 
    quartile_1, q2, quartile_3 = np.percentile(data_out[:,i],
                                               [25, 50, 75]) # 25%와 75%의 사분위수를 구함, np.percentile()는 정렬된 데이터를 입력받아 사분위수를 구함
    print('1사분위수 : ', quartile_1)
    print('50%사분위수 : ', q2)
    print('3사분위수 : ', quartile_3)
    iqr = quartile_3 - quartile_1 # 사분위수를 구함
    print('IQR : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # 1.5배 사분위수를 구함
    upper_bound = quartile_3 + (iqr * 1.5) # 1.5배 사분위수를 구함
    print('최소값 : ', lower_bound)
    print('최대값 : ', upper_bound)
    return np.where(((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))) # 최소값과 최대값 이상의 값을 찾아서 반환함

outliers_loc1 = outlier(aaa,0) # 최소값과 최대값 이상의 값을 찾아서 반환함
print('최소값과 최대값 이상의 값을 찾아서 반환함 1 : ', outliers_loc1)

outliers_loc2 = outlier(aaa,1) # 최소값과 최대값 이상의 값을 찾아서 반환함
print('최소값과 최대값 이상의 값을 찾아서 반환함 2 : ', outliers_loc2)

import matplotlib.pyplot as plt
plt.boxplot(aaa) # 산점도 그림을 그리는 함수
plt.show()

# 1사분위수 :  4.0
# 50%사분위수 :  7.0
# 3사분위수 :  10.0
# IQR :  6.0
# 최소값 :  -5.0
# 최대값 :  19.0
# 최소값과 최대값 이상의 값을 찾아서 반환함 1 :  (array([ 0, 12], dtype=int64),)


# 1사분위수 :  200.0
# 50%사분위수 :  400.0
# 3사분위수 :  600.0
# IQR :  400.0
# 최소값 :  -400.0
# 최대값 :  1200.0
# 최소값과 최대값 이상의 값을 찾아서 반환함 2 :  (array([6], dtype=int64),)