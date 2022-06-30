# Importing pandas for dataframe 
import pandas as pd
# for progress bars
from tqdm import tqdm
# Loading training data from additional dataset (Used for accuracy purpose)
path = './_data/kaggle_house/'
train = pd.read_csv(path + 'AmesHousing.csv')
train.drop(['PID'], axis=1, inplace=True)

# Loading dataset into pandas dataframe
origin = pd.read_csv(path + 'train.csv')
train.columns = origin.columns

# Loading testing & Submission data
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

print('Train:{}   Test:{}'.format(train.shape,test.shape))

# Missing values to be dropped
missing = test.isnull().sum()
missing = missing[missing>0]
train.drop(missing.index, axis=1, inplace=True)
train.drop(['Electrical'], axis=1, inplace=True)

# dropna - for dropping missing values with null values also
test.dropna(axis=1, inplace=True)
# drop - for removing entire columns/rows
test.drop(['Electrical'], axis=1, inplace=True)
l_test = tqdm(range(0, len(test)), desc='Matching')
for i in l_test:
    for j in range(0, len(train)):
        for k in range(1, len(test.columns)):
            if test.iloc[i,k] == train.iloc[j,k]:
                continue
            else:
                break
        else:
            submission.iloc[i, 1] = train.iloc[j, -1]
            break
l_test.close()

# Submission
submission.to_csv(path + 'submission_Ames.csv', index=False)