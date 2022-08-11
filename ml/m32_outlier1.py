import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
# bbb = np.array([47, 60, 64, 70, 75, 80, 83, 90])

def outlier(data_out) : 
    quartile_1, q2, quartile_3 = np.percentile(data_out,
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
    return np.where((data_out > upper_bound) | (data_out < lower_bound)) # 최소값과 최대값 이상의 값을 찾아서 반환함

outliers_loc = outlier(aaa) # 최소값과 최대값 이상의 값을 찾아서 반환함
print('최소값과 최대값 이상의 값을 찾아서 반환함 : ', outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa) # 산점도 그림을 그리는 함수
plt.show()

# 1사분위수 :  4.0
# 50%사분위수 :  7.0
# 3사분위수 :  10.0
# IQR :  6.0
# 최소값 :  -5.0
# 최대값 :  19.0
# 최소값과 최대값 이상의 값을 찾아서 반환함 :  (array([ 0, 12], dtype=int64),)