import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

y_true = np.array([1., 2., 3.])
y_pred = np.array([2.,4.,2.])

mae = mean_absolute_error(y_true, y_pred)

print(mae) 

mape = mean_absolute_percentage_error(y_true, y_pred)

print(mape)

mape_tf = keras.metrics.mean_absolute_percentage_error(y_true, y_pred)

print(mape_tf.numpy())