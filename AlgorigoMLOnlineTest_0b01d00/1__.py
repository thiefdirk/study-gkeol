# EDA

# import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('./AlgorigoMLOnlineTest_0b01d00/data/train.csv', index_col=0)
test_data = pd.read_csv('./AlgorigoMLOnlineTest_0b01d00/data/test.csv', index_col=0)
unlabel_data = pd.read_csv('./AlgorigoMLOnlineTest_0b01d00/data/unlabeled.csv', index_col=0)

print(train_data.info())
print(test_data.info())
print(unlabel_data.info())

print(train_data.head())

# labelencoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
unlabed_cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']  

for col in cat_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])
    
for col in unlabed_cat_cols:
    unlabel_data[col] = le.fit_transform(unlabel_data[col])
    
# fillna(0)

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
unlabel_data = unlabel_data.fillna(0)
    
x_train = train_data.drop(['satisfaction'], axis=1)
y_train = train_data['satisfaction']
x_test = test_data.drop(['satisfaction'], axis=1)
y_test = test_data['satisfaction']
    
print(train_data.head())
print(test_data.head())
print(unlabel_data.head())


corr = x_train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':8})
plt.xticks(rotation=50)
plt.show()

corr2 = unlabel_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr2, annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':8})
plt.xticks(rotation=30)
plt.show()




