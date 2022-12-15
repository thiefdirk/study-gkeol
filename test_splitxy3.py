import numpy as np
a = np.array(range(1, 11))
###
def split_xy3(dataset, time_step, y_columns):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_step
        y_end_number = x_end_number + y_columns - 1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)