import os#Allows us to access pictures and files with Python
from Network import train
from HSVConverter import convertHue

from ImageFormatter import *#Import the method from ImageFormatter
'''
img_folder = 'Test_Images'#Defines the folder with the test images
for images in os.listdir(img_folder):#Loops through these images
    img = Image.open("Test_Images/"+images)#Opens the Images
    format_Image(img)#Formats the Images
'''
result= [1]
img = Image.open('Test_Images/t1.jpg') #Loads the image
img = convertHue(img)

print(train(img, result))


