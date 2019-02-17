
import numpy as np


def calc_running_avg(x):

    x_len = len(x)

    running_avg = np.empty(x_len)
    for t in range(x_len):
        running_avg[t] = x[max(0, t - 100):(t + 1)].mean()

    return running_avg
