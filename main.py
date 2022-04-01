import os#Allows us to access pictures and files with Python
from Network import *
from HSVConverter import convertHue

from ImageFormatter import *#Import the method from ImageFormatter
'''
img_folder = 'Test_Images'#Defines the folder with the test images
for images in os.listdir(img_folder):#Loops through these images
    img = Image.open("Test_Images/"+images)#Opens the Images
    format_Image(img)#Formats the Images
'''
results= [1, 1]
img1 = Image.open('Test_Images/t1.jpg') #Loads the image
img1 = convertHue(img1)
img2 = Image.open('Test_Images/t2.jpg') #Loads the image
img2 = convertHue(img2)
imgarray = [img1, img2]
network = Network()
print(network.train(imgarray, results))


