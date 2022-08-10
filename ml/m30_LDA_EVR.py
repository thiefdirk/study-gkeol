import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer, fetch_california_housing, fetch_covtype, load_wine, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 1. 데이터
# datasets = load_iris() # (150, 4) -> (150, 2)
# datasets = load_breast_cancer() # (569, 30) -> (569, 1)
# datasets = load_wine() # (178, 13) -> (178, 2)
# datasets = load_digits() # (1797, 64) -> (1797, 9)
datasets = fetch_covtype() # (581012, 54) -> (581012, 6)

x = datasets.data
y = datasets.target
print(x.shape) # (150, 4)

lda = LDA()
lda.fit(x, y)
x = lda.transform(x)
print(x.shape) # (150, 4)


lda_EVR = lda.explained_variance_ratio_
cumsum = np.cumsum(lda_EVR)
print(cumsum)

