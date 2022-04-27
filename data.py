import numpy as np
from numpy import genfromtxt

ds = np.genfromtxt(r"C:\Users\Sweet\source\repos\TAI\TAI\Test_Images\iris.csv", float, delimiter=',', skip_header=1)

data = ds[:,0:4]
true = ds[:,4]