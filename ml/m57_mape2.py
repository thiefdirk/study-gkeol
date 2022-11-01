import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# y_true = np.array([100.,2.])
# y_pred = np.array([200.,102.])

y_true = np.array([100.,200.])
y_pred = np.array([200.,300.])

mae = mean_absolute_error(y_true, y_pred)

print(mae) 

mape = mean_absolute_percentage_error(y_true, y_pred)

print(mape)
