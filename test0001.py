# 3-dim dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# integerarray weekday
holiday_list = [datetime.date(2022,1,31), datetime.date(2022,1,22)]
data = pd.DataFrame(datetime.date(2022,1,31), columns=['TIME'])

for i in range(len(data['TIME'])):
    if data['TIME'][i] == holiday_list:
        data['weekday'][i] = 1
    else:
        data['weekday'][i] = 0
        
print(data)