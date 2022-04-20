import numpy as np

def mse(val_true, val_pred):
    return np.mean(np.power(val_true - val_pred, 2))

def mse_p(val_true, val_pred):
    return 2 * (val_pred - val_true) / np.size(val_true)