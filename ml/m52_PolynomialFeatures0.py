import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3) # arange(8) = [0,1,2,3,4,5,6,7]
print(x)

print(x.shape) # (4, 2)

# polynomialFeatures : 다항식 변환
pf = PolynomialFeatures(degree=2) # degree=2 : 2차항까지 만들어라, include_bias=False : 절편을 만들지 말아라

x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape) # (4, 6)