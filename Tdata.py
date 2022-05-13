import os#Allows us to access pictures and files with Python
from Image_Processing.HSVConverter import convertHue
from Image_Processing.ImageFormatter import *#Import the method from ImageFormatter
import numpy as np
from numpy import genfromtxt

true = np.genfromtxt("Test_Images/tomatoeimages/timagesTrue.csv", delimiter=',')

folder = 'Test_Images/tomatoeimages'

imgarray = np.zeros((784))
step = 1
for i in os.listdir(folder):
    f = os.path.join(folder, i)
    if os.path.isfile(f):
        extension = os.path.splitext(f)[1]
        if extension == ".jpg":
            image = Image.open(f)
            img = format_Image(image)
            a = convertHue(img)
            imgarray = np.vstack((imgarray, a))
            #print(step)
            step = step+1
        else:
            print("skipped because not a jpg")

imgarray = imgarray[1:81]

X = np.reshape(imgarray, (80, 784, 1))
Y = np.reshape(true, (80, 1, 1))

print(X.shape)