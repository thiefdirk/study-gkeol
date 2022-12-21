import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)

x = data
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]


# for i in ['ZIH120-02', 'PRESSURE-S', 'FI_SUM', 'FY_SUM', 'TI_MEAN', 'LP_TOTAL', 'FIA_SUM', 'STN-MFR-S']:
#     print(i)
#     x = data[i].values
#     print('covariance: {:.2f}'.format(np.cov(x,y)[0,1]))
#     print('correlation: {:.2f}'.format(stats.pearsonr(x,y)[0]))
#     print('p-value: {:.4f}'.format(stats.pearsonr(x,y)[1]))
#     print('/n')

rho, p_val = stats.spearmanr(x['ZIH120-02'],y)
print("p-value : {}".format(rho,p_val))