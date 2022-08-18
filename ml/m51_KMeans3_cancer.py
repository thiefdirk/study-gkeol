# KMeans 가장 대표적인 비지도 학습 알고리즘

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import accuracy_score
# pca import
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

datasets = load_breast_cancer()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

print(df.head())

scaler = StandardScaler()
scaler.fit(df)
df = scaler.transform(df)


pca = PCA(n_components=2)
pca.fit(df)
df = pca.transform(df)

kmeans = KMeans(n_clusters=2, random_state=12) # n_clusters=3 : 3개의 군집으로 나눈다
kmeans.fit(df)

results = kmeans.labels_
print(results)
print(datasets.target)

acc = accuracy_score(datasets.target, results)

print('acc: ', acc)
# acc:  0.14586994727592267

# standardScaler, pca, random_state=12
# acc:  0.9068541300527241