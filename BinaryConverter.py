from ImageFormatter import *
from numpy import asarray
img = Image.open('Test_Images/tomato1.jpg') #Loads the image
img = format_Image(img)#Formats the Image
data = asarray(img)#Makes image into an array of rgb values
print(data)#Prints the data

