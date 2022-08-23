import pandas as pd


path = './_data/dacon_travel/'

sample_submission = pd.read_csv(path + 'xgb_grid_0822.csv', index_col=0)

print(sample_submission)

sample_submission.to_csv(path+'xgb_grid.csv',index = False)

print(sample_submission)
