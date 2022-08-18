# KMeans 가장 대표적인 비지도 학습 알고리즘

import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import accuracy_score


datasets = load_wine()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

print(df.head())

kmeans = KMeans(n_clusters=3, random_state=1234) # n_clusters=3 : 3개의 군집으로 나눈다
kmeans.fit(df)

results = kmeans.labels_
print(results)
print(datasets.target)

acc = accuracy_score(datasets.target, results)

print('acc: ', acc)

# acc:  0.702247191011236