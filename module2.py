import cv2
import PIL
import numpy as np

img = cv2.imread("C:/users/Sweet/source/repos/assets/test1.jpg")

hist = np.histogram(img.flatten(),256,[0,256])[0]