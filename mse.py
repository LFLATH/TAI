import numpy as np

def mse(val_true, val_pred):
    return np.mean(np.power(val_true - val_pred, 2))

def mse_p(val_true, val_pred):
    return 2 * (val_pred - val_true) / np.size(val_true)

def rmse(true, pred):
    return np.sqrt(np.mean(np.power(true - pred, 2)))
def drmse(true, pred):
    return 1/(2 * (true-pred) / np.size(true))

drmse(1,40000)

mse_p(1,4)
rmse(1,4)