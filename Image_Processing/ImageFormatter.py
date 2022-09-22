from PIL import Image
import numpy

def format_Image(img):#Turned the file into a method
    simg = img.resize((28,28),resample=Image.NEAREST)#Resize the image to 28x28 pixels using the Nearest Function from PIL
    return simg  
