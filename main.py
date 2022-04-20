import os#Allows us to access pictures and files with Python
from HSVConverter import convertHue
from ImageFormatter import *#Import the method from ImageFormatter
import numpy as np
from dense import Dense
from tan import Hypertan
from mse import mse, mse_p
from network import train, predict
'''
img_folder = 'Test_Images'#Defines the folder with the test images
for images in os.listdir(img_folder):#Loops through these images
    img = Image.open("Test_Images/"+images)#Opens the Images
    format_Image(img)#Formats the Images

results= numpy.array([1, 1])
'''
img1 = Image.open('Test_Images/t1.jpg') #Loads the image
img1 = convertHue(img1)
img2 = Image.open('Test_Images/t2.jpg') #Loads the image
img2 = convertHue(img2)
imgarray = numpy.array([img1, img2])
print(np.shape(img1))
print(np.shape(img2))

'''

X = np.array([1, 2, 3, 4, 50, 60, 70])
Y = np.array([1, 1, 1, 1, 0, 0, 0])

network = [
    Dense(784, 10), Changed to 784 since 784 hsv values, changed to 10 because 10 nodes in the first layer
    Hypertan(),
    Dense(10, 1), Changed to w10 because 10 input nodes, output to 1 output node 
    Hypertan()
]

# train
train(network, mse, mse_p, X, Y, epochs=10000, learning_pace=0.1)
for x in X:
    pred_num = predict(network, x)
    if pred_num < 0.5:
        print("Greater than 10")
    else:
        print("Less than 10")
'''