import numpy as np

def mse(val_true, val_pred):
    return np.mean(np.power(val_true - val_pred, 2))

def dmse(val_true, val_pred):
    return 2 * (val_pred - val_true) / np.size(val_true)

def rmse(true, pred):
    return np.sqrt(np.mean(np.power(true - pred, 2)))
def drmse(true, pred):
    return 1/(2 * (pred-true) / np.size(true))

